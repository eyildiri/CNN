# Brain Hemorrhage Classification Project with CNN

This project aims to classify different types of brain hemorrhages from medical images, using both classic Convolutional Neural Networks (CNN) and pretrained models (such as ResNet18).

**Note:** The dataset is not included in this repository.  
Please download the dataset from [https://www.kaggle.com/datasets/jhoward/rsna-hemorrhage-jpg] and place it in a folder named `data/` at the root of the project, as required by the scripts.


## Script Overview

- **prepare_data.py**  
  Prepares and organizes the data into `train` and `test` folders for each class, based on images and labels.  
  - Filters images to keep only those with a single type of hemorrhage (or none).
  - Balances the number of images per class.
  - Creates the folder structure expected for training and testing.

- **cnn_model.py**  
  Defines a "from scratch" CNN architecture for multi-class image classification.

- **train.py**  
  Trains the CNN model defined in `cnn_model.py` on the prepared data.  
  - Handles the training loop, validation, and model saving.

- **test.py**  
  Evaluates the trained CNN model on the test set.  
  - Computes performance metrics (accuracy, confusion matrix, etc).

- **pretrained_model.py**  
  Defines a model based on a pretrained network (e.g., ResNet18), adapted for hemorrhage classification.

- **pretrained_train.py**  
  Trains the pretrained model defined in `pretrained_model.py` on the prepared data.  
  - Allows fine-tuning the model on your specific dataset.

- **pretrained_test.py**  
  Evaluates the pretrained model on the test set.  
  - Computes performance metrics.

## Data Folder Organization

After running `prepare_data.py`, the following structure is created:
```
multi_class_new_data/
    train/
        epidural/
        subdural/
        subarachnoid/
        intraparenchymal/
        intraventricular/
        none/
    test/
        epidural/
        subdural/
        subarachnoid/
        intraparenchymal/
        intraventricular/
        none/
```

## Usage

1. **Prepare the data**  
   ```
   python prepare_data.py
   ```

2. **Train a classic CNN**  
   ```
   python train.py
   ```

3. **Test the classic CNN**  
   ```
   python test.py
   ```

4. **Train a pretrained model (ResNet18, etc.)**  
   ```
   python pretrained_train.py
   ```

5. **Test the pretrained model**  
   ```
   python pretrained_test.py
   ```