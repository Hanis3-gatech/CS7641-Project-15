# Curse of Dimensionality through Random Lasso Regression
### Bilal Mufti, Hira Anis, James Mathhew Hamilton, Noor-us-Saba Khan, Shakir Shakoor Khatti
# 1. Introduction
# 2. Problem Statement
In real world scenarios, especially in the field of bioinformatics and genomics, we frequently encounter datasets with a lot more features than samples such that the ratio of features to data points is in the order of 500. In such situations, the traditional regression techniques fail to produce beneficial results exhibiting decreasing accuracy with increasing features to datapoint ratio. This shortcoming of regression models leads to research constraints in this and other concerned fields. 
# 3. Data Set & Basic Idea
 The Dataset GSE45827 on breast cancer gene expression from CuMiDa is taken from Kaggle. The dataset is available here ([link](https://www.kaggle.com/brunogrisci/breast-cancer-gene-expression-cumida)). Kaggle is an online community of data scientists and machine learning practitioners and offering public datasets for algorithm testing. 
The dataset comprises of 151 samples and gene expression data values for 54676 genes . There are six classes of sample types namely,'HER' , 'basal', 'cell_line', 'luminal_A', 'luminal_B' and 'normal'.
## Data Pre-Processing and formatting
We downloaded a csv file containing 152 rows and 54677 columns, available on the link mentioned earlier.   
  The first column is a sample identification number (samples), and the second is the sample type (type). We loaded the "type" column as a dataframe and used that for color coding and visualization of the PCA plots later. We used the header for gene ids as identification marker to form an array for processing. We used this array to check for 'Nan' or missing values and after confirming that there are none, we proceed to use it as our processing dataset. 
# 4. Unsupervised Machine Learning
## Principal Component Analysis
When it comes to the dimensionality reduction, Principal Component Analysis (PCA), which comes under the unsupervised learning techniques of Machine Learning (ML) is primarily used for the visualization of the high-dimensional data in the new set of coordinates defined using PCA. Here we use PCA in scikit-learn decomposition module.  PCA aids in the dimensionality reduction by orthogonally projecting the data onto lower-dimension linear space by maximizing the variance of projected data and minimizing the mean squared distance between projection and data points.  
  In this application, PCA is applied on the pre-processed data set to visualize the data in newly defined principal components. In this project, the data is visualized in both 2D and 3D using the first two principal components and first three principal components respectively. Cumulative variance for the principal components is plotted against the number of principal components to analyze the number of principal components needed to achieve the required variance.
## Results for Unsupervised Learning Model
### Cumulative Variance
PCA can be used to describe how many components can be used to describe the data completely. We  will determine this by looking at the cumulative explained variance ratio as a function of the number of components. 

|![alt-text-1](cutoffvariance.png)|
|:--:|
|*Fig 1. Cumulative variance plotted against number of principal components*|

This curve shows much of the total variance is contained within the first 'n' PCA components. We can see that our first 30 PCA components contain approximately 60% of the variance. We are interested in the number of components that retain approximately 99% of variance. From the above graph we can see that 99% of variance is contained in the first 143 components. The remaining components contain information that is mostly redundant and is not useful in describing the data labels. 

### 2D Principal Component Data Visualization 
As it can be seen in the presented results for the cumulative variance retained by the number of principal components that first two principal components retain greater amount of variance compared to the rest of principal components, therefore, we have plotted the data using the original labels in the space defined by first two principal components <img src="https://render.githubusercontent.com/render/math?math=PC_{1} \text{ and } PC_{2}">

|![alt-text-2](PCA2D.png)|
|:--:|
|*Fig 2. 2D visualization of data using first two principal components*|

### 3D Principal Component Data Visualization 
Similarly, we have plotted the data using the original labels in the space of first three principal components <img src="https://render.githubusercontent.com/render/math?math=PC_{1}, PC_{2} \text{ and } PC_{3}">

|![alt-text-3](PCA3D.png)|
|:--:|
|*Fig 3. 3D visualization of data using first three principal components*|


### Regression Analysis with and without PCA
In this section, we will see that how PCA can be used to improve the prediction through regression analysis. Linear and ridge regression were applied on dimensionally reduced data, obtained through PCA. The orignal dataset had 54765 features which were reduced down to 151.  

After application of PCA, a new dataset with 151 principal components was achieved. The original dataset had categorical labels having 6 different categories, however it is not possible to use categorical data directly for regression and hence label encoding was used to assign numeric values to the labels. This new dataset and transformed labels were then divided in to training and testing data using 80-20 division rule. 

Initially, the model was trained using linear regression and then the test data was used to find the predicted labels and rmse came out to be 10.246 as shown below from output.  
RMSE Linear Regression: 10.246049817401543  

Next, ridge regression was used for training and testing. Since ridge regression is a type of regression technique that uses regularization constant (lambda) for weights prediction therefore it was required to find a lambda value that could minimize the error. For this purpose a 10-fold cross validation was carried out, and from the figure below it can be seen that the optimal value turned out to be 50000.  

|![alt-text-4](Crossvalidation.png "title-1") | ![alt-text-2](Datatable.PNG "title-2")|
|:--:|:--:|
|*Fig 4. 10-fold cross validation*| Sample |

After doing unsupervised learning analysis using PCA, it was observed from the RMSE values obtained with and without PCA that PCA is not the most accurate technique to handle the dataset with D>>>N. Therefore it was required to come up with a novel algorithm that can potently handle the data with very large number of features compared to samples. An algorithm for random lasso regression was written to cater this problem which will be discussed in detail under supervised learning part of the project.(Moreover it was also learned that linear and ridge regression are not very suitable for categorical dataset)







# 5. Supervised Machine Learning
Supervised learning looks at both an independent (X) and dependent (y) variable, where we attempt to build some model using the given data in X to predict the values in y. Unsupervised learning only looks at and independent variable (X) and seeks to derive some meaning from this dataset alone. We will limit our focus to the subset of supervised learning known as **regression analysis** here.

* The **independent variable** is an NxD matrix, where N is the samples and D is the features.

* The **dependent variable**  is a vector of length N, where N is once again the samples.

### High Dimensional Data | Intuitively Explained

Regression analysis tends to work best when there are more samples than features. **High Dimensional Data** is the term we use then features exceeds samples. The performance of regression techniques will continue to degrade as the ratio of features to samples increases. **This makes some sense intuitively, as N _(samples)_ should really be thought of as our "data", and D _(features)_ is actually something we are trying to predict to ultimately predict y.** The more data we have, the easier it is to make discover how each D predicts y. The larger D is the more features we have to examine, and use to predict y.

Alas, it does not benefit us to look at more features, despite the impression that more-data-is-better. In fact, we would like to examine is little features as possible. So if someone were to tell you:

>"I have a ton of data to look at, we can get some great results. The csv is 13,033,599 by 88!" You should respond, "Is this 13 million features?", to which a "yes" would warrant the conclusion, "We have no data." - A Story of the Curse of Dimensionality

#### Bad:

<p style="text-align: center;">Features >> Samples</p>  
<p style="text-align: center;">D >> N</p>  

But at what point does it the ratio of features to samples become so high that we cannot derive meaning? We analysis this here, but first we must discuss...

### Dimension Reduction

Some of the features in a dataset may not be meaningful in anyway. Our goal is to eliminate these. By reducing the number of features we will essentially have more data; as discussed above. We discussed PCA, which uses a slightly different flavor or a goal. Indeed PCA reduces dimensionally, but **PCA is does not even look at the response variable y**. In general PCA will not help any model perform better in terms of accuracy, and we saw this above. This is because PCA treats the features with low variance as unimportant, and the features with high variance as important. It can be such that these high variance features have nothing to do with the response variable. It can be such that these low variance features have something to do with the response variable.**Alas, we can use regression for dimension reduction, which will take into consideration the response variable!**

The whole point of regression is to predict a coefficient for each feature, which can be used as a model, a formula, to predict y. There are a handful of beautiful regression techniques that researchers have at their disposal:

#### 1. **Ordinary Least Squares → No 0's**
  - Creates the most optimal line of best fit.
  - All coefficients predicted are non-zero.
  - I will show you results later that show the below *'advanced'* regression methods outperforming Ordinary Least Squares. Counterintuitively, this is actually the best regression method. No other regression method can ever be able to hope to outperform this. If you understand this you truly understand regression.
#### 2. **Ridge Regression → No 0's**
  - Similar to Ordinary Least Squares, but introduces a tunable variable λ that improves accuracy in exchange for a small amount for bias.
  - As we increase bias we  decrease variance. There is a tradeoff.
#### 3. **Lasso → Many 0's**
  - Similar to Ridge Regression in that there is a tunable variable λ.
  -The difference is that Lasso shrink coefficients that have little to do with the response variable down to 0. _(Is this what we are looking for?)_
#### 4. **Elastic-Net → Some 0's**
  - This is a hybrid between Ridge Regression and Lasso. So why didn't I put it in the middle of Ridge Regression and Lasso? Chronologically, it was invented later and uses both ridge and lasso.
  - It even has two λ's, which is pretty neat. If either λ_1 or λ_2  are zero, then the Elastic-Net would become either Lasso or Ridge Regression.
#### 5. **Adaptive Lasso & Random Lasso → Slightly More 0's than Lasso**
  - These are what we researched. More on them later.
  
[Insert Image of RMSE on Low Dim Data...]
  
|![alt-text-5](RME_Feature_to_Samples.png "title-1")|
|:--:|
|*Fig 5. 1|

[Analysis Here...]

|![alt-text-6](F1_Feature_to_Samples.png "title-1")|
|:--:|
|*Fig 6. 1|

[Analysis Here...]

|![alt-text-7](Runtime_Feature_to_Samples.png "title-1")|
|:--:|
|*Fig 7. 1|

[Analysis Here...]

|![alt-text-8](RME_Constant.png "title-1")|
|:--:|
|*Fig 8. 1|

[Analysis Here...]


# 6. Model Evaluation
