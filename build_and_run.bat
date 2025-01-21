@echo off
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
echo Running generator...
echo:
dotnet Input_Generator/bin/Release/net7.0/Input_Generator.dll

echo:
echo Running program...
echo:
main.exe