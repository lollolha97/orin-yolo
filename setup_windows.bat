@echo off
REM Windows RTX 4060 환경 설정 스크립트

echo ========================================
echo Windows RTX 4060 환경 설정
echo ========================================
echo.

REM 1. CUDA 확인
echo [1/5] CUDA 설치 확인...
nvidia-smi
if %errorlevel% neq 0 (
    echo ❌ NVIDIA 드라이버 미설치
    echo https://www.nvidia.com/Download/index.aspx 에서 설치
    pause
    exit /b 1
)
echo ✅ NVIDIA 드라이버 확인 완료
echo.

REM 2. Python 가상환경 생성
echo [2/5] Python 가상환경 생성...
python -m venv venv
call venv\Scripts\activate.bat
echo ✅ 가상환경 생성 완료
echo.

REM 3. PyTorch GPU 버전 설치
echo [3/5] PyTorch (CUDA 지원) 설치...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ✅ PyTorch 설치 완료
echo.

REM 4. Ultralytics 설치
echo [4/5] Ultralytics 설치...
pip install ultralytics
echo ✅ Ultralytics 설치 완료
echo.

REM 5. GPU 확인 테스트
echo [5/5] GPU 인식 테스트...
python -c "import torch; print('CUDA 사용 가능:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo.

echo ========================================
echo 설치 완료!
echo ========================================
echo.
echo 학습 실행:
echo   venv\Scripts\activate
echo   python src\training\train_windows.py
echo.
pause
