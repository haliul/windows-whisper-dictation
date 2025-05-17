# Windows Real-time Dictation with Whisper

This is a Python script for real-time dictation on Windows. It uses the OpenAI Whisper model to convert speech to text and automatically inputs the recognized text into the active text field.

## Features

*   **Real-time Speech Recognition**: Captures audio from the microphone and transcribes it on the fly.
*   **High-Quality Recognition**: Uses OpenAI's Whisper models (defaults to the `large` model).
*   **Automatic Text Input**: Recognized text is automatically typed into the active window.
*   **Voice Activity Detection (VAD)**: Starts recording upon detecting speech and processes audio after a period of silence.
*   **GPU Support (CUDA)**: Automatically utilizes NVIDIA GPUs for accelerated recognition if available.
*   **Configurable Parameters**: Microphone sensitivity threshold, silence duration, and other parameters can be adjusted in the script.
*   **Convenient Launch**: Includes a `.bat` file for quick startup.

## Requirements

*   Windows
*   Python 3.8+
*   Microphone
*   NVIDIA GPU with CUDA support (recommended for fast performance with the `large` model, but the script will also run on CPU)
*   FFmpeg (may be required by Whisper; it's recommended to install it and add it to PATH)

## Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <YOUR_REPOSITORY_FOLDER>
    ```

2.  **Create and activate a Python virtual environment:**
    In the project directory (e.g., `D:\WINSPER`):
    ```powershell
    python -m venv winsper_env
    .\winsper_env\Scripts\Activate.ps1
    ```
    For `cmd.exe`:
    ```cmd
    python -m venv winsper_env
    .\winsper_env\Scripts\activate.bat
    ```

3.  **Install the required dependencies:**
    While in the activated virtual environment, run:
    ```bash
    pip install openai-whisper sounddevice pyautogui torch torchvision torchaudio scipy pyperclip
    ```
    To use an NVIDIA GPU with CUDA (e.g., CUDA 12.1), it's recommended to install PyTorch separately:
    ```bash
    pip uninstall torch torchvision torchaudio -y # Uninstall existing ones, if any
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    Replace `cu121` with your CUDA version if necessary (you can find it using the `nvidia-smi` command).

## Usage

1.  **Via `.bat` file (recommended):**
    *   Simply double-click the `run_dictation.bat` file.
    *   A command prompt window will open, and the script will start.
    *   After startup, you will have 3 seconds to focus the desired text field.
    *   To stop the script, press `Ctrl+C` in the command prompt window.

2.  **Directly via Python:**
    *   Ensure the `winsper_env` virtual environment is activated.
    *   Run the command:
        ```bash
        python realtime_dictation.py
        ```
    *   After startup, you will have 3 seconds to focus the desired text field.
    *   To stop the script, press `Ctrl+C`.

## Configuration

Key parameters can be changed at the beginning of the `realtime_dictation.py` file:

*   `MODEL_TYPE`: Whisper model type (e.g., `"large"`, `"medium"`, `"base"`). `large` offers the highest quality but requires more resources.
*   `LANGUAGE`: Recognition language (e.g., `"ru"` for Russian, `"en"` for English, `None` for auto-detection).
*   `THRESHOLD_RMS`: Sensitivity threshold for detecting speech (adjust experimentally, e.g., `0.015`).
*   `SILENCE_DURATION_S`: Seconds of silence to consider the end of a phrase (e.g., `1.2`).
*   `MIN_VOICE_DURATION_S`: Minimum duration of a voice segment to be processed (e.g., `0.3`).

## Troubleshooting

*   **`ModuleNotFoundError`**: Ensure you have activated the virtual environment and installed all dependencies using the command in the "Installation" section.
*   **CUDA not available / slow performance**:
    *   Ensure you have NVIDIA drivers and CUDA Toolkit installed (if required for your PyTorch version).
    *   Verify PyTorch is installed with support for your CUDA version (see "Installation" section).
    *   Check the script's output: it should indicate if CUDA is being used.
*   **Microphone not working / no sound**:
    *   Check if the microphone is connected and selected as the default recording device in Windows.
    *   Verify microphone access permissions for Python/terminal in Windows settings.
*   **Problems with Cyrillic input (or other non-ASCII characters)**:
    *   The script uses a copy-paste method (emulating `Ctrl+V`), which is usually reliable for Unicode. If issues occur, ensure the target application correctly handles pasting from the clipboard.
    *   The active keyboard layout might affect input emulation. The script attempts to work around this by using clipboard functions.
*   **`ffmpeg` not found**: Whisper may require `ffmpeg`. Download it from the official website, extract it, and add the path to the `bin` folder (containing `ffmpeg.exe`) to your system's PATH variable.

## Dependencies

Main dependencies (installed via pip):

*   `openai-whisper`
*   `sounddevice`
*   `numpy`
*   `pyautogui`
*   `pyperclip`
*   `torch` (preferably with CUDA support)
*   `scipy`

Happy dictating!

---

# Windows Real-time Dictation with Whisper (Русская версия)

Это Python-скрипт для диктовки в реальном времени под Windows. Он использует модель OpenAI Whisper для преобразования речи в текст и автоматически вводит распознанный текст в активное текстовое поле.

## Возможности

*   **Распознавание речи в реальном времени**: Захватывает аудио с микрофона и транскрибирует его "на лету".
*   **Высокое качество распознавания**: Использует модели Whisper от OpenAI (по умолчанию настроена `large` модель).
*   **Автоматический ввод текста**: Распознанный текст автоматически печатается в активное окно.
*   **Определение голосовой активности (VAD)**: Начинает запись при обнаружении речи и обрабатывает аудио после периода тишины.
*   **Поддержка GPU (CUDA)**: Автоматически использует GPU NVIDIA для ускорения распознавания, если доступно.
*   **Настраиваемые параметры**: Порог чувствительности микрофона, длительность тишины и другие параметры можно настроить в скрипте.
*   **Удобный запуск**: Включает `.bat` файл для быстрого запуска.

## Требования

*   Windows
*   Python 3.8+
*   Микрофон
*   Видеокарта NVIDIA с поддержкой CUDA (рекомендуется для быстрой работы `large` модели, но скрипт будет работать и на CPU)
*   FFmpeg (может потребоваться для Whisper, рекомендуется установить и добавить в PATH)

## Установка

1.  **Клонируйте репозиторий (или скачайте файлы):**
    ```bash
    git clone <URL_ВАШЕГО_РЕПОЗИТОРИЯ>
    cd <ПАПКА_РЕПОЗИТОРИЯ>
    ```

2.  **Создайте и активируйте виртуальное окружение Python:**
    В директории проекта (например, `D:\WINSPER`):
    ```powershell
    python -m venv winsper_env
    .\winsper_env\Scripts\Activate.ps1
    ```
    Для `cmd.exe`:
    ```cmd
    python -m venv winsper_env
    .\winsper_env\Scripts\activate.bat
    ```

3.  **Установите необходимые зависимости:**
    Находясь в активированном виртуальном окружении, выполните:
    ```bash
    pip install openai-whisper sounddevice pyautogui torch torchvision torchaudio scipy pyperclip
    ```
    Для использования GPU NVIDIA с CUDA (например, CUDA 12.1), рекомендуется установить PyTorch отдельно:
    ```bash
    pip uninstall torch torchvision torchaudio -y # Удалить существующие, если есть
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    Замените `cu121` на вашу версию CUDA, если необходимо (узнать можно командой `nvidia-smi`).

## Использование

1.  **Через `.bat` файл (рекомендуется):**
    *   Просто дважды щелкните по файлу `run_dictation.bat`.
    *   Откроется окно командной строки, скрипт запустится.
    *   После запуска у вас будет 3 секунды, чтобы сделать активным нужное текстовое поле.
    *   Для остановки скрипта нажмите `Ctrl+C` в окне командной строки.

2.  **Напрямую через Python:**
    *   Убедитесь, что виртуальное окружение `winsper_env` активировано.
    *   Выполните команду:
        ```bash
        python realtime_dictation.py
        ```
    *   После запуска у вас будет 3 секунды, чтобы сделать активным нужное текстовое поле.
    *   Для остановки скрипта нажмите `Ctrl+C`.

## Настройка

Основные параметры можно изменить в начале файла `realtime_dictation.py`:

*   `MODEL_TYPE`: Тип модели Whisper (например, `"large"`, `"medium"`, `"base"`). `large` - самое высокое качество, но требует больше ресурсов.
*   `LANGUAGE`: Язык распознавания (например, `"ru"` для русского, `"en"` для английского, `None` для автоопределения).
*   `THRESHOLD_RMS`: Порог чувствительности для определения начала речи (подбирается экспериментально, например, `0.015`).
*   `SILENCE_DURATION_S`: Сколько секунд тишины считать окончанием фразы (например, `1.2`).
*   `MIN_VOICE_DURATION_S`: Минимальная длительность голосового сегмента для обработки (например, `0.3`).

## Возможные проблемы и их решение

*   **Ошибка `ModuleNotFoundError`**: Убедитесь, что вы активировали виртуальное окружение и установили все зависимости (см. раздел "Установка").
*   **CUDA не доступна / медленная работа**:
    *   Убедитесь, что у вас установлены драйверы NVIDIA и CUDA Toolkit (если требуется для вашей версии PyTorch).
    *   Убедитесь, что PyTorch установлен с поддержкой вашей версии CUDA (см. раздел "Установка").
    *   Проверьте вывод скрипта: он должен сообщить, используется ли CUDA.
*   **Микрофон не работает / нет звука**:
    *   Проверьте, подключен ли микрофон и выбран ли он устройством записи по умолчанию в Windows.
    *   Проверьте разрешения на доступ к микрофону для Python/терминала в настройках Windows.
*   **Проблемы с вводом кириллицы (или других не-ASCII символов)**:
    *   Скрипт использует метод копирования-вставки (эмуляция `Ctrl+V`), который обычно надежен для Unicode. Если возникают проблемы, убедитесь, что целевое приложение корректно обрабатывает вставку из буфера обмена.
    *   Активная раскладка клавиатуры может влиять на эмуляцию ввода. Скрипт пытается это обойти, используя функции буфера обмена.
*   **`ffmpeg` не найден**: Whisper может требовать `ffmpeg`. Скачайте его с официального сайта, распакуйте и добавьте путь к папке `bin` (где находится `ffmpeg.exe`) в системную переменную PATH.

## Зависимости

Основные зависимости (устанавливаются через pip):

*   `openai-whisper`
*   `sounddevice`
*   `numpy`
*   `pyautogui`
*   `pyperclip`
*   `torch` (желательно с поддержкой CUDA)
*   `scipy`

Приятной диктовки! 