@echo off
cd models
if not exist *.hdf5 (
    echo No .hdf5 files found in ./models directory.
    pause
    exit
)
del *.hdf5
echo All .hdf5 files in ./models directory have been deleted.
pause
