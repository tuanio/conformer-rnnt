# Conformer RNN-T

## How to use this respository

1. Clone this project to current directory. Using those commands:
```
!git init
!git remote add origin https://github.com/tuanio/conformer-rnnt
!git pull origin main
```
2. Install requirement packages
```
!pip install -r requirements.txt
```
3. Edit `configs.yaml` file for appropriation.
4. Train model using `python main.py -cp conf -cn configs`

## Note
- `sox` is audio backend for linux, `PySoundFile` is audio backend for windows

## Environment variable
- `HYDRA_FULL_ERROR=1`