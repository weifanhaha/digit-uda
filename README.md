# Unsupervised Domain Adaptation for Digit Image Classification

Train [DANN](https://arxiv.org/abs/1409.7495) and [ADDA](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf) models to conduct unsupervised domain adaptation for digit image classification. In this repository, I tried the models on three datasets: `USPS`, `MNIST-M`, `SVHN` and consider three senario:

1. Use source images and labels for training
2. Use target images and labels for training
3. Use source images and labels + target images for training
   and test on images in target dataset.

# Usage

## Download Dataset

```
./get_dataset.sh
```

## Install packages

```
pip install -r requirements.txt
```

## Train

To train DANN / ADDA model, you need to enter the corresponding folder and run training script. please take care of the path of training data and output model path. You can change the the source domain, the target domain and whether to use the images in target domain in the code.

Train DANN

```
cd dann
python train.py
```

Train ADDA

```
cd adda
python train.py
```

## Predict Labels

You can predict the labels of the images in target dataset with trained models.

Predict labels with DANN

```
cd dann
python predict.py --test_path $test_path --d_target $d_target --output_predict_path $output_predict_path
```

Predict labels with ADDA

```
cd adda
python predict.py --test_path $test_path --d_target $d_target --output_predict_path $output_predict_path
```

-   `$test_path`: path of the test image directory
-   `$d_target`: name of target dataset, should be one of `mnistm`, `usps`, or `svhn`
-   `$output_predict_path`: path of the output csv

# Results

Here I compared the accuracy on target images in the following scenario:

1. Use source images and labels for training (lower bound)
2. Use target images and labels for training (upper bound)
3. Use source images and labels + target images for training (DANN)
4. Use source images and labels + target images for training (ADDA)

|       Scenario       | USPS -> MNIST-M | MNIST-M -> SVHN | SVHN -> USPS |
| :------------------: | :-------------: | :-------------: | :----------: |
| 1. Trained on source |     0.2855      |     0.3293      |    0.6367    |
| 2. Trained on target |     0.9629      |     0.9163      |    0.9621    |
|       3. DANN        |     0.4152      |     0.4339      |    0.6203    |
|       4. ADDA        |     0.4619      |     0.4544      |    0.6472    |

We can see that both DANN and ADDA improves the accuracy while ADDA performs better than DANN.

# Reference

The implementation refers to:

-   DANN: [dann github](https://github.com/fungtion/DANN)
-   ADDA: [pytorch-adda github](https://github.com/corenel/pytorch-adda)
