@echo off
echo Attempting to build Matrix Multiplication...

REM Try to find cl.exe in PATH
where cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] cl.exe not found in PATH.
    echo Please run this script from the "x64 Native Tools Command Prompt for VS 2019/2022".
    echo Alternatively, the script will attempt to find vcvars64.bat automatically...
    
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
        echo Found VS2019 BuildTools, setting up environment...
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo [ERROR] Could not automatically find vcvars64.bat. 
        echo Please run 'build.bat' from a Visual Studio Developer Command Prompt.
        exit /b 1
    )
)

echo Running NVCC...
nvcc matrix_mul.cu -o matrix_mul.exe
if %errorlevel% neq 0 (
    echo [ERROR] NVCC Compilation failed!
    exit /b %errorlevel%
)

echo [SUCCESS] Compilation successful!
echo Running matrix_mul.exe...
matrix_mul.exe
