# AMLS_II_assignment23_24-23073916
My project is the Cassava Leaf Disease Classification challenge of Kaggle, which can be found here:https://www.kaggle.com/competitions/cassava-leaf-disease-classification

The file includes a main.py file and a functions.py file which contains the functions used in this project. The functions.py is under the task folder. 

Some ".csv" files are also under the task folder. These files are the losses and accuracies results of different models. As the model training will cost much time, the plots are drawn directly using these ".csv" files. It is worth mention that in the main.py, the plots will be drawn first for the same reason, but in the normal process the model training and validation should be implemented first. In main(), I just commented out the model train and validation code, and let the same code run after the plotting. The plots will be saved under task folder after runing the code.

Because the training dataset is too large and cannot be uploaded to Github, you can download the dataset at:https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data. After downloading, you need to move the train_images folder and the train.csv file into the Dataset folder.

The packages needed are:
pandas
seaborn
matplotlib
opencv-python
numpy
tqdm
torch
scikit-learn
albumentations
timm
