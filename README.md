# Attention-Guided-Low-light-Image-Enhancement-with-Scene-Text-Restoration

## 4 modules for IC_15 :
Can be see in [google colab](https://colab.research.google.com/drive/1srI2wA46PNeqaOsGVoiUYcAox5Z0uXu0?usp=sharing).
- preprocess
- network model
- train
- test


The weight of this IC_15 model trained at 500 epoch for IC15 dataset is in [model1.pth](https://drive.google.com/drive/folders/1Fo8LSy4sOKQvkFLLFMolnTakgev6h8ky?usp=sharing).



## Requirements
- Python 3.6
- Pytorch 1.3
- RawPy 0.13.1
- SciPy 1.0.0
```
pip install -r requirements.txt
```
## Dataset
Please follow the steps from the original [code](https://github.com/cchen156/Learning-to-See-in-the-Dark).

Please download RGB images from  [here](https://drive.google.com/drive/folders/1NDlZtsyvfSHuxqEn9l-mCr9BHKztpAy4?usp=sharing).

## Testing


1. To test Sony data, run
```
python test_Sony.py
```
By default, the result will be saved in "result_Sony/final" folder.

2. To test Fuji data, run
```
python test_Fuji.py
```
By default, the result will be saved in "result_Fuji/final" folder.

## Training
1. To train the Sony model, run
```
python train_Sony.py
```
The result and model will be saved in "result_Sony" folder by default.

2. To train the Fuji model, run
```
python train_Fuji.py
```
The result and model will be saved in "result_Fuji" folder by default.

## Citation
Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.

## License
MIT License

## Questions
Please contact me you have any questions. sheephow@gapp.nthu.edu.tw
