# Multi-Label Classification of Fashion Articles Images
I developed a computer vision model for multi-output classification of Fashion Article images reaching an average accuracy of 94.1%

## Code and Resources
* **Python version:** 3.7
* **Preferable development environment:** Colaboratory
* **Packages:** tensorflow, keras, PIL, numpy, pandas, matplotlib, seaborn, os
* **Dataset Source:**  https://www.kaggle.com/paramaggarwal/fashion-product-images-small 
* **Universal Machine Learning Workflow from:** Deep Learning with Python by F.Chollet

## Project Overview
* Developed a CNN model that predicts the gender, the category and the subcategory of fashion article images
* This project follows the Universal Machine Learning Workflow which includes: 
  1. Define the problem
  2. Set the measure of success
  3. Set an evaluation protocol
  4. Load and prepare the data
  5. Develope a model that does better than a baseline model
  6. Develope a model that overfits by scaling up
  7. Tune the hyperparameters, add regularisation and train and test the final model.
* The data preparation step includes:
  1. Data exploration and data cleansing
  2. Image Processing
  3. Finalising data preparation for NN
* I applied the keras functional API for building models with multiple outputs 
* During the hyperparameters tunning, I explored models with:
  1. Different local receptive field sizes on the conv2d layers
  2. Different number of filters on the conv2d layers
  3. Different number of nodes in the dense layer
  4. Addition of BatchNormalisation in the model at different positions
  5. Addition of Dropout layer with different values
* The final model achieved an averaged accuracy of 96.2%, which includes a 99.2% accuracy for masterCategory, 97.3% for subCategory and 92.1% for gender.
* The confusion matrices show that the model successfully classified all the different classes within masterCategory and subCategory but for the gender task classification, the model struggled to classify correctly the unisex articles.
* Improvements of the model could be done by increasing the amount of images for unrepresented classes.


## 1. Problem definition
For a new  e-commerce platform to be success, it should be very well structure. As there could be multiple sellers on this platform, creating a dataset with all articles sorted according to the same standard may be difficult. The purpose of this work is to facilitate this task by developing a computer vision model that can identify different categories of fashion articles such as gender and article type.
 
The hypothesis was defined as follow:
* The gender and article type (both for a masterCategory and subCategory) could be predicted from the article image.

## 2. Set a measure of success
The measure of success was the accuracy, it was calculated for each classification task, and it was averaged to measure the overall performance of the model.

## 3. Set an evaluation protocol
The chosen evaluation protocol was "one hold-out" where we train with the training set and evaluate with the validation set once for every epoch.

## 4. Load and prepare the data
### Data exploration and data cleansing
An already existing structured dataset of fashion articles data and its corresponding images was utilized in this project.
The dataset containes information and images of 44424 articles. However, processing such an amount of images requires more than the available RAM in Colab, hence, I opted to work with a reduced dataset that contains a third of the original one. 

The final dataframe contains information about 14808 articles, which are categorised in differnt ways, by gender, by masterCategory, by subCategory and more detailed categories such as article type, detail description of the products between others. By looking at the amount of distinct values for each column, I decided to work with gender, masterCategory and subCategory. 

By exploring the distribution of the different classes for each category, I understood that there are many classes that are underrepresented, hence I decided to remove from the dataset, the classes that represented less than 1%.

The final distributions for the different classes are shown bellow:

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/subClassdistribution.png)

![alt text](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/masterClassdistribution.png)
![alt text](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/genderDistribution.png)


### Image Processing
I opened each image, converted into a 3d tensor, where two dimension represent the image size and the 3rd dimension represent the channels (the primary colors). I checked the tensor shape, and discard the images that did not have the correct shape, I scaled the values of the tensor to be between 0 and 1 and put all the images in a list. This list was then converted into a 4d tensor, where the first dimension corresponds to the number of samples.

### Finalising data preparation for NN
In this step:
* I deleted the rows that corresponded to the wrong shape images 
* I converted the labels into one-hot encoding tensors.
* I split the data into train, validation and test sets

## 5. Develope a model that does better than a baseline model
* First, I created a baseline model using only fully-connected layers. This model achieved an average accuracy of 74.9%
* Secondly, I created a  basic convolutional layer model with a single convolutional layer, a maxpooling layer, a flatten layer, a single dense layer in addition to the output layers for classification. This model improved the baseline model achieving an average accuracy of 91.6%.




