# GANomaly

This repository contains PyTorch implementation of the following paper: GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training [[1]](#reference)

##  1. Table of Contents
 ```
 Installation
 Custom Dataset
 Train
 Test
 ```
    

## 2. Installation
1. First clone this repository
   ```
   git clone https://github.com/samet-akcay/ganomaly.git
   ```
2. Create the virtual environment via conda
    ```
    conda env create -f requirements_cuda11.1.yaml -n "gan" 
    ```
3. Activate the virtual environment.
    ```
    conda activate gan
    ```


## 3. Custom Dataset
To train the model on a custom dataset, the dataset should be copied into `./data` directory, and should have the following directory & file structure:

```
./data/casting 

├── test
│   ├── normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

## 4. Train
To list the arguments, run the following command:
```
python train.py
```



## 5. Test
If you use this repository or would like to refer the paper, please use the following BibTeX entry
```
python train.py --phase "test" --save_test_images --load_weights --batchsize 1
```

## 6. Reference
[1]  Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham
