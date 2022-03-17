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
* The final model achieved an averaged accuracy of 94.1%, which includes a 98.3% accuracy for masterCategory, 94.7% for subCategory and 89.4% for gender.
* The confusion matrices show that the model successfully classified all the different classes within masterCategory, almost all of them within subCategory but for the gender task classification, the model struggled to classify correctly the unisex articles and the girls articles which were the less represented.
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
* Secondly, I created a  basic convolutional layer model with a single convolutional layer, a maxpooling layer, a flatten layer, a single dense layer in addition to the output layers for classification. This model improved the baseline model achieving an average accuracy of 91.6%. The reason for the success is that the CNN models are able to retain graphical patterns from the images, and do not assume that the object is in the same position for every image as the baseline model did.

## 6. Develope a model that overfits by scaling up
I created a model as bellow

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/model.png)

From the validation graphs, I could see that there is a high overfitting of the data, however, it reached higher validation accuracies than the simple CNN model.

For the next step, I continued using this model architecture but I explored different hyperparamaters.

## 7. Tune the hyperparameters, add regularisation and train and test the final model.

### Hyperparameter tunning
I created an experiment combining the different hyperparameters that I wanted to explore: local receptive field size, number of filters per layer and number of nodes in the dense layer.

As the number of filters for each layer will be different, I defined the hyperparameter "filters_amount" as "few" or "many" which dictated the number of filters for each layer. Similarly, I used the hyperparameter for the local receptive field as "small" or "large" taking in account that for the initial layers, I could use larger patches than the ones in further up layers.

The results of the experimental training and validation looked as follow

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/experimentalResults.png)

For each trial, I stored the minimum validation loss and the average validation accuracy at this point which I called maximum validation accuracy. From this, I could plot the following graph to understand better the influence of each hyperparameter in the learning process.

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/boxplotloss.png)
![alt text](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/boxplotacc%20(1).png)

From these results, I understood that the hyperparameter that improves the training without affecting the overfitting is the patch size, which should be small, while for the other parameters, number of filters in the conv2d and number of nodes in the dense layer, it is best to have a combination of a high value in one and a low value in the other.

The experiment which reached the minimum validation loss is the experiment with small local receptive fields, fewer filters per layer and 100 nodes in the dense layer so I continued working with these hyperparameters for the next step. 

## Add Batch Normalisation
In this step, I tried adding batch normalisation at different stages in the model. These were the results:

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/BNresults.png)

The Batch normalisation didn't improve the model, I could see that the validation loss vs epochs plots for the trials with batch normalisation show much more oscilation without reducing the validation loss. Hence the best trial is the one obtained in the previous step without batch normalisation.

## Add Dropout Regulariser
I add a dropout layer before the dense layer after the flatten layer. For this experiment, I tried using three values of dropout rate and we compared them to the original model without dropout. In addition, I increased the number of epochs, as the dropout generally slows the learning down but it should reach lower validation losses and higher validation accuracies.

The results looked as follows:

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/DOresults.png)

The best dropout trial was the one which applied 0.5 dropout. It achieved the lowest minimum validation loss and the maximum average validation accuracy. Moreover, I could see that the minimum validation loss decreases as the Dropout rate increases.

I used this dropout rate for the final model architecture. From the latest results, I looked for the optimal number of epochs which I used to train the final model.

### Training the final model

I joint the validation and training sets into a final training set, I used the optimal number of epochs to train it.

## Final Results
* **Gender Classification accuracy:** 89.4 %
* **MasterCategory Classification accuracy:** 98.3%
* **SubCategory Classification accuracy:**  94.7%
* **Average Classification accuracy:** 94.1%

#### Confusion Matrices
The confusion matrices for each classification task looked as follows:

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/genderCM.png)

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/masterCatCM.png)

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/subCatCM.png)

#### Test images examples

![](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/correct0.png)
![alt text](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/correct1.png)
![alt text](https://github.com/CarolinaKra/FashionArticlesImageClassification/blob/main/Images/correct2.png)









