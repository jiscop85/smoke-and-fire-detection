Fire and Smoke Detection System with YOLOv8
This project implements a real-time fire and smoke detection system using YOLOv8 and Flask. It is capable of detecting fire and smoke through a connected webcam.

Table of Contents
Features

Prerequisites

Installation

Usage

File Structure

Troubleshooting

Important Notes

Features
Real-time fire and smoke detection using YOLOv8.

Web-based interface built with Flask.

Utilizes webcam for live video feed.

Plays an alert sound upon detection.

Prerequisites
Before you begin, ensure you have the following:

Python 3.8 or higher: The project is developed with Python 3.8+.

Webcam: A connected and functional webcam.

Minimum 4GB RAM: Recommended for smooth operation.

Windows 10 or higher: The current setup is optimized for Windows. Minor adjustments might be needed for other operating systems.

Installation
Follow these steps to set up the project:

Install Python:

Download and install the latest version of Python from the official Python website.

Crucially, make sure to select "Add Python to PATH" during installation.

Download the Project:

Download the project files from its repository.

Extract the files to a suitable folder on your system.

Create a Virtual Environment:
Open your terminal (Command Prompt or PowerShell) and execute the following commands:

# Navigate to your project folder
cd path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment (on Windows)
.\venv\Scripts\activate

Install Required Libraries:
With your virtual environment activated, install all necessary libraries using pip:

pip install -r requirements.txt

Download YOLOv8 Model:

If you have your own trained YOLOv8 model (best.pt), place it in the following path: YOLOv8-Fire-and-Smoke-Detection/runs/detect/train/weights/best.pt.

Otherwise, the system will automatically download and use the default YOLOv8n model upon first run.

Usage
Once the installation is complete, follow these steps to run the application:

Ensure Virtual Environment is Activated:

# On Windows
.\venv\Scripts\activate

Run the Application:
Execute the main Python script:

python app.py

Access the Web Interface:

Open your web browser.

Navigate to http://localhost:5000.

Click the "Start" button on the webpage to initiate the detection system.

File Structure
project/
│
├── app.py                     # Main application script
├── requirements.txt           # List of required Python libraries
├── Fire-Truck-Sound.mp3       # Alert sound file
│
├── templates/
│   └── index.html             # Web page template
│
└── static/
    └── style.css              # Stylesheet for the web page

Troubleshooting
Here are solutions to common issues you might encounter:

Webcam is not working:

Ensure your webcam is correctly connected and powered on.

Close any other applications that might be using the webcam.

Try restarting your system.

Library installation errors:

Verify that Python is installed correctly and added to your system's PATH.

Try upgrading pip and reinstalling the requirements:

pip install --upgrade pip
pip install -r requirements.txt

Web page does not open:

Confirm that port 5000 is not being used by another application.

Ensure you are entering the full address: http://localhost:5000.

Important Notes
For local network access, you can replace localhost with your system's IP address.

To change the default port, you can modify the 5000 value in the app.py file.

For improved performance or specific camera settings, you might need to adjust webcam configurations within the app.py file.


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------

# سیستم تشخیص دود و آتش با YOLOv8

این پروژه یک سیستم تشخیص دود و آتش با استفاده از YOLOv8 و Flask است که قابلیت تشخیص آتش را از طریق دوربین وب‌کم دارد.

## پیش‌نیازها

1. پایتون نسخه 3.8 یا بالاتر
2. دوربین وب‌کم متصل به سیستم
3. حداقل 4GB RAM
4. ویندوز 10 یا بالاتر (برای سیستم‌عامل‌های دیگر نیاز به تغییرات جزئی در کد دارد)

## مراحل نصب

1. **نصب پایتون:**
   - از [سایت رسمی پایتون](https://www.python.org/downloads/) آخرین نسخه را دانلود و نصب کنید
   - حتماً گزینه "Add Python to PATH" را هنگام نصب انتخاب کنید

2. **دانلود پروژه:**
   - پروژه را دانلود کنید
   - فایل‌ها را در یک پوشه مناسب استخراج کنید

3. **ایجاد محیط مجازی:**
   در ترمینال (Command Prompt یا PowerShell) دستورات زیر را اجرا کنید:
   ```bash
   # رفتن به پوشه پروژه
   cd path/to/project

   # ایجاد محیط مجازی
   python -m venv venv

   # فعال‌سازی محیط مجازی در ویندوز
   .\venv\Scripts\activate
   ```

4. **نصب کتابخانه‌های مورد نیاز:**
   ```bash
   pip install -r requirements.txt
   ```

5. **دانلود مدل YOLOv8:**
   - اگر مدل آموزش‌دیده خود را دارید، آن را در مسیر `YOLOv8-Fire-and-Smoke-Detection/runs/detect/train/weights/best.pt` قرار دهید
   - در غیر این صورت، سیستم به طور خودکار از مدل پیش‌فرض YOLOv8n استفاده خواهد کرد

## نحوه اجرا

1. **اطمینان از فعال بودن محیط مجازی:**
   ```bash
   # در ویندوز
   .\venv\Scripts\activate
   ```

2. **اجرای برنامه:**
   ```bash
   python app.py
   ```

3. **دسترسی به رابط کاربری:**
   - مرورگر خود را باز کنید
   - به آدرس `http://localhost:5000` بروید
   - روی دکمه "شروع" کلیک کنید تا سیستم شروع به کار کند

## ساختار فایل‌ها

```
project/
│
├── app.py                    # فایل اصلی برنامه
├── requirements.txt          # لیست کتابخانه‌های مورد نیاز
├── Fire-Truck-Sound.mp3     # فایل صوتی هشدار
│
├── templates/
│   └── index.html           # قالب صفحه وب
│
└── static/
    └── style.css            # فایل استایل
```

## رفع مشکلات متداول

1. **دوربین کار نمی‌کند:**
   - مطمئن شوید دوربین به درستی متصل است
   - برنامه‌های دیگر که از دوربین استفاده می‌کنند را ببندید
   - سیستم را ری‌استارت کنید

2. **خطای نصب کتابخانه‌ها:**
   - مطمئن شوید پایتون به درستی نصب شده است
   - دستور زیر را اجرا کنید:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```

3. **صفحه وب باز نمی‌شود:**
   - مطمئن شوید پورت 5000 آزاد است
   - آدرس را به صورت کامل وارد کنید: `http://localhost:5000`

## نکات مهم

- برای استفاده در شبکه محلی، می‌توانید به جای `localhost` از IP سیستم خود استفاده کنید
- برای تغییر پورت، می‌توانید عدد 5000 را در فایل `app.py` تغییر دهید
- برای بهبود عملکرد، می‌توانید تنظیمات دوربین را در فایل `app.py` تغییر دهید 
