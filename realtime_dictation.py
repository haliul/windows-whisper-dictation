import whisper
import sounddevice as sd
import numpy as np
import pyautogui
import pyperclip
import torch
import threading
import queue
import time
from scipy.signal import butter, lfilter

# --- Настройки ---
MODEL_TYPE = "large"  # "tiny", "base", "small", "medium", "large"
LANGUAGE = "ru"       # Укажи язык для более точного распознавания, если нужно
SAMPLE_RATE = 16000   # Whisper ожидает 16kHz
CHANNELS = 1
BLOCK_DURATION_MS = 100 # Длительность одного блока аудио для анализа (в миллисекундах)
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000) # Размер блока в сэмплах

# Настройки VAD (Voice Activity Detection)
THRESHOLD_RMS = 0.015       # Порог RMS для определения речи (подбирается экспериментально)
SILENCE_DURATION_S = 1.2  # Сколько секунд тишины считать окончанием фразы
MIN_VOICE_DURATION_S = 0.3 # Минимальная длительность голосового сегмента для обработки
MAX_RECORD_DURATION_S = 30 # Максимальная длительность одной записи перед принудительной обработкой

# Очереди для обмена данными между потоками
audio_queue = queue.Queue()
text_queue = queue.Queue()

# Глобальные флаги и переменные
is_running = True
model = None
last_speech_time = time.time()
active_recording_buffer = []
is_speaking_flag = False


def rms(data):
    """Вычисляет RMS аудиоданных."""
    return np.sqrt(np.mean(data**2))

def bandpass_filter(data, lowcut=300.0, highcut=3000.0, fs=SAMPLE_RATE, order=5):
    """Простой полосовой фильтр для выделения голосовых частот."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    try:
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y
    except Exception: # Если что-то пошло не так с частотами
        return data


def audio_capture_thread():
    """Поток для захвата аудио с микрофона."""
    global last_speech_time, active_recording_buffer, is_speaking_flag

    print("Поток захвата аудио запущен.")
    
    stream_callback_data = {
        "last_speech_time": time.time(),
        "active_recording_buffer": [],
        "is_speaking_currently": False,
        "speech_start_time": 0
    }

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Статус stream: {status}", flush=True)
        
        # Преобразование в float32, если это не так (Whisper этого ожидает)
        processed_data = indata.astype(np.float32).flatten()

        # Фильтрация (опционально, может улучшить VAD)
        # processed_data = bandpass_filter(processed_data)

        current_rms = rms(processed_data)
        
        now = time.time()

        if current_rms > THRESHOLD_RMS:
            stream_callback_data["last_speech_time"] = now
            if not stream_callback_data["is_speaking_currently"]:
                print("Обнаружена речь...")
                stream_callback_data["is_speaking_currently"] = True
                stream_callback_data["speech_start_time"] = now
                stream_callback_data["active_recording_buffer"] = [processed_data.copy()] # Начинаем новый буфер
            else:
                stream_callback_data["active_recording_buffer"].append(processed_data.copy())
            
            # Принудительная обработка, если запись слишком длинная
            if (now - stream_callback_data["speech_start_time"]) > MAX_RECORD_DURATION_S and stream_callback_data["active_recording_buffer"]:
                print("Запись слишком длинная, принудительная обработка.")
                full_audio = np.concatenate(stream_callback_data["active_recording_buffer"])
                audio_queue.put(full_audio)
                stream_callback_data["active_recording_buffer"] = []
                stream_callback_data["is_speaking_currently"] = False


        elif stream_callback_data["is_speaking_currently"]: # Была речь, а теперь тишина
            if (now - stream_callback_data["last_speech_time"]) > SILENCE_DURATION_S:
                print("Обнаружена тишина после речи, обрабатываем.")
                if stream_callback_data["active_recording_buffer"] and \
                   (now - stream_callback_data["speech_start_time"]) >= MIN_VOICE_DURATION_S :
                    full_audio = np.concatenate(stream_callback_data["active_recording_buffer"])
                    audio_queue.put(full_audio)
                stream_callback_data["active_recording_buffer"] = []
                stream_callback_data["is_speaking_currently"] = False
            else: # Тишина, но еще не достаточно долгая, продолжаем накапливать, если вдруг это пауза в речи
                 stream_callback_data["active_recording_buffer"].append(processed_data.copy())


    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', blocksize=BLOCK_SIZE, callback=callback):
            print(f"Начинаю прослушивание микрофона (модель: {MODEL_TYPE}, порог RMS: {THRESHOLD_RMS}, тишина: {SILENCE_DURATION_S}с)...")
            while is_running:
                sd.sleep(100) # Держим поток живым
    except Exception as e:
        print(f"Ошибка в потоке захвата аудио: {e}")
        print("Возможно, проблема с микрофоном или его настройками.")
        print("Убедитесь, что микрофон подключен и выбран как устройство по умолчанию.")
        print("Также проверьте разрешения на доступ к микрофону для Python/терминала.")
        text_queue.put(None) # Сигнал для основного потока о завершении


def whisper_transcription_thread():
    """Поток для транскрипции аудио с помощью Whisper."""
    global model, is_running
    print("Поток транскрипции запущен.")
    
    try:
        print("Загрузка модели Whisper (это может занять время)...")
        # Проверка CUDA и установка fp16
        use_fp16 = not torch.cuda.is_available() # False если CUDA есть, True если CPU
        if torch.cuda.is_available():
            print(f"CUDA доступна. Используем GPU. fp16={use_fp16}")
        else:
            print(f"CUDA НЕ доступна. Используем CPU. fp16={use_fp16}. Это будет медленно для модели '{MODEL_TYPE}'.")
        
        model = whisper.load_model(MODEL_TYPE, device=None) # device=None -> автоопределение (CUDA if available)
        print(f"Модель Whisper '{MODEL_TYPE}' загружена.")
    except Exception as e:
        print(f"Критическая ошибка: Не удалось загрузить модель Whisper: {e}")
        print("Убедитесь, что PyTorch установлен с поддержкой CUDA, если вы хотите использовать GPU.")
        print("Попробуйте модель поменьше (например, 'base' или 'small'), если проблема в ресурсах.")
        is_running = False # Останавливаем все, если модель не загрузилась
        text_queue.put(None) # Сигнализируем основному потоку
        return

    while is_running:
        try:
            audio_data_np = audio_queue.get(timeout=1) # Ждем данные 1 секунду
            if audio_data_np is None: # Сигнал о завершении
                break
            if not is_running: break # Дополнительная проверка

            print(f"Получен аудиофрагмент длиной {len(audio_data_np)/SAMPLE_RATE:.2f}с для транскрипции.")
            
            # Транскрипция
            # Убедимся, что данные в нужном формате (float32)
            if audio_data_np.dtype != np.float32:
                 audio_data_np = audio_data_np.astype(np.float32)
            
            # Нормализация (Whisper ожидает аудио в диапазоне -1 до 1, хотя обычно справляется и так)
            # audio_data_np = audio_data_np / np.max(np.abs(audio_data_np)) if np.any(audio_data_np) else audio_data_np


            transcription_options = {"language": LANGUAGE, "fp16": use_fp16}
            if LANGUAGE is None: # Если язык не указан, Whisper попытается его определить
                del transcription_options["language"]

            result = model.transcribe(audio_data_np, **transcription_options)
            
            recognized_text = result["text"].strip()

            if recognized_text: # Только если что-то распознано
                print(f"Распознано: {recognized_text}")
                text_queue.put(recognized_text)
            else:
                print("Ничего не распознано или пустая строка.")
            
            audio_queue.task_done()

        except queue.Empty:
            continue # Таймаут, просто продолжаем цикл
        except Exception as e:
            if is_running: # Не печатаем ошибку, если нас останавливают
                print(f"Ошибка в потоке транскрипции: {e}")
            time.sleep(0.1) # Небольшая пауза в случае ошибки


def main():
    """Основная функция."""
    global is_running
    print("Запуск диктовки. Нажмите Ctrl+C для выхода.")
    print(f"Убедитесь, что активным является текстовое поле, куда нужно вводить текст.")
    time.sleep(2) # Даем время переключиться на нужное окно

    capture_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    transcribe_thread = threading.Thread(target=whisper_transcription_thread, daemon=True)

    capture_thread.start()
    transcribe_thread.start()

    try:
        while is_running:
            try:
                text_to_type = text_queue.get(timeout=0.5) # Ждем текст недолго
                if text_to_type is None: # Сигнал о критической ошибке из потоков
                    print("Получен сигнал о завершении из потока. Выход.")
                    is_running = False
                    break
                
                if text_to_type:
                    full_text_to_type = text_to_type + " "
                    print(f"Распознанный текст для ввода: '{full_text_to_type}'")
                    try:
                        print(f"Копирую в буфер: '{full_text_to_type}'")
                        pyperclip.copy(full_text_to_type)
                        print("Текст скопирован в буфер обмена.")
                        time.sleep(0.1) # Даем буферу обмена мгновение
                        
                        print("Попытка вставки через Ctrl+V (keyDown/keyUp)...")
                        pyautogui.keyDown('ctrlleft') # Нажимаем левый Ctrl
                        time.sleep(0.05) # Небольшая пауза
                        pyautogui.keyDown('v')      # Нажимаем V
                        time.sleep(0.05) # Держим обе клавиши
                        pyautogui.keyUp('v')        # Отпускаем V
                        time.sleep(0.05) # Небольшая пауза
                        pyautogui.keyUp('ctrlleft')  # Отпускаем левый Ctrl
                        print("Команды keyDown/keyUp для Ctrl+V отправлены.")
                        # time.sleep(0.2) # Пауза после вставки, возможно, не нужна здесь

                    except Exception as e:
                        print(f"Ошибка при использовании буфера обмена: {e}")
                        print("Попробую посимвольный ввод как запасной вариант...")
                        # Очистим буфер, чтобы избежать случайной вставки чего-то не того позже
                        try: 
                            pyperclip.copy("") 
                        except: pass

                        for char_idx, char_val in enumerate(full_text_to_type):
                            # Этот print может быть слишком частым, но полезен для отладки
                            # print(f"  Запасной ввод: символ {char_idx+1}/{len(full_text_to_type)}: '{char_val}'") 
                            pyautogui.typewrite(char_val, interval=0.05)
                        print("Запасной посимвольный ввод завершен.")
                    
                text_queue.task_done()

            except queue.Empty:
                if not capture_thread.is_alive() and model is None: # Если модель не загрузилась и поток захвата упал
                    print("Поток захвата аудио не активен и модель не загружена. Выход.")
                    is_running = False
                elif not capture_thread.is_alive() and model is not None:
                     print("Поток захвата аудио не активен. Проверьте микрофон. Выход.")
                     is_running = False
                if not transcribe_thread.is_alive() and model is not None: # Если модель загружена, но поток транскрипции упал
                    print("Поток транскрипции не активен. Выход.")
                    is_running = False
                continue
            except Exception as e:
                print(f"Ошибка в основном цикле: {e}")
                time.sleep(0.1)


    except KeyboardInterrupt:
        print("\nCtrl+C нажат. Завершаю работу...")
    finally:
        is_running = False
        print("Остановка потоков...")
        
        # Даем потокам шанс завершиться корректно
        if capture_thread.is_alive():
            # Для sounddevice нет простого способа прервать InputStream из другого потока,
            # кроме как через is_running, который он проверяет в sd.sleep().
            # Главное, чтобы callback больше не вызывался.
            pass

        if transcribe_thread.is_alive():
            audio_queue.put(None) # Отправляем сигнал завершения потоку транскрипции
            transcribe_thread.join(timeout=2)

        print("Программа завершена.")


if __name__ == "__main__":
    # Небольшая задержка перед стартом, чтобы успеть переключить фокус
    # на нужное текстовое поле после запуска скрипта из терминала.
    print("Скрипт запущен. У вас есть 3 секунды, чтобы сделать активным нужное текстовое поле.")
    time.sleep(3)
    main() 