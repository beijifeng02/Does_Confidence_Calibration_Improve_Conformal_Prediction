# Does_Confidence_Calibration_Improve_Conformal_Prediction

This repository is the official implementation
of [Does confidence calibration improve conformal prediction?](https://openreview.net/forum?id=6DDaTwTvdE&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DTMLR%2FAuthors%23your-submissions)) at TMLR'2025

## How to Run

Results for ConfTS:
```
python main.py --data_dir root_for_imagenet --preprocess 'confts'
```

Results for ConfPS:
```
python main.py --data_dir root_for_imagenet --preprocess 'confps'
```

Results for ConfVS:
```
python main.py --data_dir root_for_imagenet --preprocess 'confvs'
```