# Curse of Dimensionality through Random Lasso Regression
### Bilal Mufti, Hira Anis, James Mathhew Hamilton, Noor-us-Sabah Khan, Shakir Shakoor Khatti
# 1. Introduction
# 2. Problem Statement
In real world scenarios, especially in the field of bioinformatics and genomics, we frequently encounter datasets with a lot more features than samples such that the ratio of features to data points is in the order of 500. In such situations, the traditional regression techniques fail to produce beneficial results exhibiting decreasing accuracy with increasing features to datapoint ratio. This shortcoming of regression models leads to research constraints in this and other concerned fields. 
# 3. Data Set & Basic Idea
 The Dataset GSE45827 on breast cancer gene expression from CuMiDa is taken from Kaggle. The dataset is available here ([link](https://www.kaggle.com/brunogrisci/breast-cancer-gene-expression-cumida)). Kaggle is an online community of data scientists and machine learning practitioners and offering public datasets for algorithm testing. 
The dataset comprises of 151 samples and gene expression data values for 54676 genes . There are six classes of sample types namely,'HER' , 'basal', 'cell_line', 'luminal_A', 'luminal_B' and 'normal'.
## The Inspiration
The inspiration for the dataset is to identify the most important genes for classification of each cancer subtype and to deal with class imbalance for classification.
## Data Pre-Processing and formatting
We downloaded a csv file containing 152 rows and 54677 columns, available on the link mentioned earlier.   
  The first column is a sample identification number (samples), and the second is the sample type (type). We loaded the "type" column as a dataframe and used that for color coding and visualization of the PCA plots later. We used the header for gene ids as identification marker to form an array for processing. We used this array to check for 'Nan' or missing values and after confirming that there are none, we proceed to use it as our processing dataset. 
# 4. Unsupervised Machine Learning
## Principal Component Analysis
When it comes to the dimensionality reduction, Principal Component Analysis (PCA), which comes under the unsupervised learning techniques of Machine Learning (ML) is primarily used for the visualization of the high-dimensional data in the new set of coordinates defined using PCA. Here we use PCA in scikit-learn decomposition module.  PCA aids in the dimensionality reduction by orthogonally projecting the data onto lower-dimension linear space by maximizing the variance of projected data and minimizing the mean squared distance between projection and data points.  
  In this application, PCA is applied on the pre-processed data set to visualize the data in newly defined principal components. In this project, the data is visualized in both 2D and 3D using the first two principal components and first three principal components respectively. Cumulative variance for the principal components is plotted against the number of principal components to analyze the number of principal components needed to achieve the required variance.
# 5. Supervised Machine Learning
# 6. Model Evaluation
## 6.1 Results for Unsupervised Learning Model
### Cumulative Variance
PCA can be used to describe how many components can be used to describe the data completely. We  will determine this by looking at the cumulative explained variance ratio as a function of the number of components.  

|![Image of PCA](https://github.com/Hanis3-gatech/CS7641-Project-15/blob/master/cutoffvariance.png)|
|:--:|
|*Space*|

This curve shows much of the total variance is contained within the first 'n' PCA components. We can see that our first 30 PCA components contain approximately 60% of the variance. We are interested in the number of components that retain approximately 99% of variance. From the above graph we can see that 99% of variance is contained in the first 143 components. The remaining components contain information that is mostly redundant and is not useful in describing the data labels. 

### 2D PCA

|![Image of PCA](https://github.com/Hanis3-gatech/CS7641-Project-15/blob/master/PCA2D.png)|
|:--:|
|*Space*|

### 3D PCA

|![Image of PCA](https://github.com/Hanis3-gatech/CS7641-Project-15/blob/master/PCA3D.png)|
|:--:|
|*Space*|

An html link for a hover 3D plot is available here
