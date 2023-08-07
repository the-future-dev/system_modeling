@echo off
cd dataset
if not exist *.csv (
    echo No .csv files found in ./dataset directory.
    pause
    exit
)
del *.csv
echo All .csv files in ./models directory have been deleted.
pause
