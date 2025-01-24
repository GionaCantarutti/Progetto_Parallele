@echo off

:: Parse arguments
set "ACTION=%1"
if "%ACTION%"=="" set "ACTION=all"

if /i "%ACTION%"=="cs" goto compile_cs
if /i "%ACTION%"=="cuda" goto compile_cuda
if /i "%ACTION%"=="all" goto compile_all

echo Invalid argument! Use "cs", "cuda", or "all".
pause
exit /b 1

:compile_cs
echo Compiling C# generator...
dotnet build Input_Generator/Input_Generator.csproj -c Release

if %errorlevel% neq 0 (
    echo Compilation of C# failed!
    pause
    exit /b %errorlevel%
)

echo:
echo Running C# generator...
echo:
dotnet Input_Generator/bin/Release/net7.0/Input_Generator.dll

exit /b 0

:compile_cuda
echo Compiling CUDA...
nvcc -o main.exe main.cu char_matrix.cu serial_cc.cu benchmark.cu

if %errorlevel% neq 0 (
    echo Compilation of CUDA failed!
    pause
    exit /b %errorlevel%
)

echo:
echo Running CUDA program...
echo:
main.exe

exit /b 0

:compile_all
echo Compiling C# generator...
dotnet build Input_Generator/Input_Generator.csproj -c Release

if %errorlevel% neq 0 (
    echo Compilation of C# failed!
    pause
    exit /b %errorlevel%
)

echo Compiling CUDA...
nvcc -o main.exe main.cu

if %errorlevel% neq 0 (
    echo Compilation of CUDA failed!
    pause
    exit /b %errorlevel%
)

echo Done compiling!

echo:
echo Running C# generator...
echo:
dotnet Input_Generator/bin/Release/net7.0/Input_Generator.dll

echo:
echo Running CUDA program...
echo:
main.exe

exit /b 0
