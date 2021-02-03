
# %%
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import datetime


# %%
##bringing in the dataset
vehicleinsurance=pd.read_csv("vehicle claim insurance data.csv")
vehicleinsurance.head()


# %%
###checking dimensions of the data
vehicleinsurance.shape


# %%
##lookoing for NaN
vehicleinsurance.isna().sum()

# %% [markdown]
# REPLACING OF NA s WITH THE MODE
# 

# %%
vehicleinsurance["Gender"]=vehicleinsurance["Gender"].fillna(("Others"),inplace=False)
vehicleinsurance["Car_Category"]=vehicleinsurance["Car_Category"].fillna(vehicleinsurance["Car_Category"].mode()[0])
vehicleinsurance["Subject_Car_Colour"]=vehicleinsurance["Subject_Car_Colour"].fillna(vehicleinsurance["Subject_Car_Colour"].mode()[0])
vehicleinsurance["Subject_Car_Make"]=vehicleinsurance["Subject_Car_Make"].fillna(vehicleinsurance["Subject_Car_Make"].mode()[0])
vehicleinsurance["LGA_Name"]=vehicleinsurance["LGA_Name"].fillna(vehicleinsurance["LGA_Name"].mode()[0])
vehicleinsurance["State"]=vehicleinsurance["State"].fillna(vehicleinsurance["State"].mode()[0])


# %%
##reconfirming that replacement of Nas has taken effect
vehicleinsurance.isna().sum()


# %%
#checking the new shape
vehicleinsurance.shape


# %%
#reconfirm for Na
vehicleinsurance.isna().sum()


# %%
##renaming  the target column
vehicleinsurance = vehicleinsurance.rename(columns={'target': 'No_of_claims_3_mon_period'}, index={'13': '13'})
vehicleinsurance.head()


# %%
##checking datatypes
vehicleinsurance.dtypes


# %%
# convert the 'policy startDate','policy end date','first transaction date' columns to datetime format
vehicleinsurance['Policy Start Date']= pd.to_datetime(vehicleinsurance['Policy Start Date'])
vehicleinsurance['Policy End Date']= pd.to_datetime(vehicleinsurance['Policy End Date'])
vehicleinsurance['First Transaction Date']= pd.to_datetime(vehicleinsurance['First Transaction Date'])
# Check the changes
vehicleinsurance.info()


# %%
#checking the entries in the 'gender' dataset
vehicleinsurance['Gender'].unique()


# %%
#convert all other gender types apart from male and female to "others" 
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('Entity'), 'Others', vehicleinsurance['Gender'])
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('NO'), 'Others', vehicleinsurance['Gender'])
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('Joint'), 'Others', vehicleinsurance['Gender'])
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('NOT'), 'Others', vehicleinsurance['Gender'])
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('no'), 'Others', vehicleinsurance['Gender'])
vehicleinsurance['Gender'] = np.where(vehicleinsurance['Gender'].str.contains('SEX'), 'Others', vehicleinsurance['Gender'])
#check if changes has been effected
vehicleinsurance["Gender"].unique()


# %%
##adding a policy duration column
vehicleinsurance['Policy Duration'] = vehicleinsurance['Policy End Date'] - vehicleinsurance['Policy Start Date']


# %%
#extracting the month from policy start date and storing in policy start month
#to carry out summary
vehicleinsurance['policy start month'] = vehicleinsurance['Policy Start Date'].dt.month

vehicleinsurance['policy start month'].mode()##this shows that most policies are started in the first
#month(january) 


# %%
###checking the unique features of age
vehicleinsurance["Age"].unique()

# %% [markdown]
# dealing with outliers¶
# here we are removing values greater than 100 and less than 18 as they are not important to our work because in the real sense individuals below the age of 18 and above the age of 100 can't have a vehicle insurance
# 

# %%
index=vehicleinsurance[(vehicleinsurance["Age"] >=100)|(vehicleinsurance["Age"]<=18)].index
vehicleinsurance.drop(index,inplace=True)
vehicleinsurance["Age"].describe()

# %% [markdown]
# Descriptive statistics

# %%
vehicleinsurance.describe()

# %% [markdown]
# Visualizations

# %%
import matplotlib.pyplot as plt


# %%
sns.countplot(x="Gender",data=vehicleinsurance)


# %%
sns.countplot(x="No_of_claims_3_mon_period",data=vehicleinsurance)


# %%
#bar plot
plt.figure(figsize =(12,8))

plt.bar(vehicleinsurance['Gender'], vehicleinsurance['No_Pol'])

plt.xlabel('Gender')
plt.ylabel('No_Pol')


# %%
sns.boxplot(data=vehicleinsurance,x="Age")


# %%
##histogram
vehicleinsurance.hist(figsize = (15,8))
plt.show()


# %%
vehicleinsurance.boxplot(figsize = (15,8))
plt.show()


# %%
##converting columns to "category" 
vehicleinsurance["Gender"]=vehicleinsurance["Gender"].astype("category")
vehicleinsurance["No_Pol"]=vehicleinsurance["Gender"].astype("category")
vehicleinsurance["Car_Category"]=vehicleinsurance["Car_Category"].astype("category")
vehicleinsurance["Subject_Car_Colour"]=vehicleinsurance["Subject_Car_Colour"].astype("category")
vehicleinsurance["Subject_Car_Make"]=vehicleinsurance["Subject_Car_Make"].astype("category")            
vehicleinsurance["LGA_Name"]=vehicleinsurance["LGA_Name"].astype("category")  
vehicleinsurance["State"]=vehicleinsurance["State"].astype("category") 
vehicleinsurance["ProductName"]=vehicleinsurance["ProductName"].astype("category") 
vehicleinsurance["No_of_claims_3_mon_period"]=vehicleinsurance["No_of_claims_3_mon_period"].astype("category") 


# %%
##to check if conversion has taken place
vehicleinsurance.info()

# %% [markdown]
# ### Label encoding
# using Label encoding to encode certain columns to numerical values because most algorithms work
# better with numerical inputs.this approach requires the category column to be "category" datatype hence the codes above where we changed columns to category types.note we won't be encoding the target variable("No_of_claims_3_mon_period")

# %%
#assign numerical values and storing them in other columns
vehicleinsurance["Gender_cat"]=vehicleinsurance["Gender"].cat.codes
vehicleinsurance["No_Pol_cat"]=vehicleinsurance["No_Pol"].cat.codes
vehicleinsurance["Car_Category_cat"]=vehicleinsurance["Car_Category"].cat.codes
vehicleinsurance["Subject_Car_Colour_cat"]=vehicleinsurance["Subject_Car_Colour"].cat.codes
vehicleinsurance["Subject_Car_Make_cat"]=vehicleinsurance["Subject_Car_Make"].cat.codes           
vehicleinsurance["LGA_Name_cat"]=vehicleinsurance["LGA_Name"].cat.codes  
vehicleinsurance["State_cat"]=vehicleinsurance["State"].cat.codes 
vehicleinsurance["ProductName_cat"]=vehicleinsurance["ProductName"].cat.codes


# %%
vehicleinsurance.head()#to view the changes


# %%
##creating a new dataframe of all the columns i need for my machine learning
vehicle=vehicleinsurance[["Age","Gender_cat","No_Pol_cat","Car_Category_cat","Subject_Car_Colour_cat","Subject_Car_Make_cat","LGA_Name_cat","State_cat","ProductName_cat"
                         ,"No_of_claims_3_mon_period"]]
vehicle.head()


# %%
vehicle.describe()


# %%
##scale some columns in our dataset to normalise,(do not scale target column(No_of_claims_3_mon_period))most 
#machine learning classifiers prefer this
from sklearn import preprocessing
min_max_scaler=preprocessing.MinMaxScaler()
cols_to_norm=["Age","Gender_cat","No_Pol_cat","Car_Category_cat","Subject_Car_Colour_cat","Subject_Car_Make_cat","LGA_Name_cat","State_cat","ProductName_cat"
                         ]
vehicle[cols_to_norm]=min_max_scaler.fit_transform(vehicle[cols_to_norm])


# %%
vehicle.describe()

# %% [markdown]
# ### calculating correlation matrix

# %%
vehicle_corr=vehicle.corr()
vehicle_corr

# %% [markdown]
# #### visualization to show correlation between variables

# %%
plt.figure(figsize=(10,10))
sns.heatmap(vehicle_corr,annot=True)

# %% [markdown]
# splitting our dataset into two

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',probability=True)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# %%
# x are all our features (after droping our target "No_of_claims_3_mon_period")
# y is the target we are trying to predict ("No_of_claims_3_mon_period")
x = vehicle.drop("No_of_claims_3_mon_period", axis=1)
y = vehicle["No_of_claims_3_mon_period"]


# %%

#1. Training data to train all our classifiers on real data and
#2. Test data to evaluate their performance on unseen data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Test size is 0.2 means we would be training each of our 
#classifiers with 80% of the data and validating them with the 
#remaining 20%.

# x_train - 80% of the features for training our classifiers algorithms
# y_train - 80% of the corresponding target(No_of_claims_3_mon_period) for
#training our classifiers algorithms

# x_test - Remaining 20% of our features for testing the performance 
#of our classifiers from the training phase above by producing - "y_pred".

# y_test - Remaining 20% of the corresponding target. Used in comparing
#actuals (y_test) with y_pred (predictions made)

#Random state guarantee that same sequence of random numbers are
#generated for training each of our classifiers


# %%
x_train.shape


# %%
y_train.shape#80% of target variable (No_of_claims_3_mon_period) - 8911 records


# %%
x_test.shape


# %%
#20% of target variable (Outcome) - 2228 records
y_test.shape

# %% [markdown]
# implementing machine learning algorithms on dataset

# %%
#Define the Machine Learning Classifiers
lr = LogisticRegression() 
gs = GaussianNB()
sv = SVC(kernel='rbf', random_state=0)
gb= GradientBoostingClassifier(learning_rate=0.1, n_estimators=100)
kn = KNeighborsClassifier(n_neighbors = 18) 
ad = AdaBoostClassifier(n_estimators=100, random_state=0)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators = 70)


# %%
#Fit function trains all our machine learning classifiers on 80% of our data.
lr.fit(x_train, y_train);
gs.fit(x_train, y_train);
svclassifier.fit(x_train, y_train);
gb.fit(x_train,y_train);
kn.fit(x_train,y_train);
ad.fit(x_train,y_train);
dt.fit(x_train, y_train);
rf.fit(x_train, y_train);


# %%
#Get individual classifiers training scores
print("Training_score_LogisticRegression : " , lr.score(x_train, y_train))
print("Training_score_GaussianNB : " , gs.score(x_train, y_train))
print("Training_score_SVC : " , svclassifier.score(x_train, y_train))
print("Training_score_GradientBoostingClassifier : " , gb.score(x_train, y_train))
print("Training_score_KNeighborsClassifier : " , kn.score(x_train, y_train))
print("Training_score_ AdaBoostClassifier: " , ad.score(x_train, y_train))
print("Training_score_DecisionTree : " , dt.score(x_train, y_train))
print("Training_score_randomForest : " , rf.score(x_train, y_train))

# %% [markdown]
# #### TESTING FOR ALL OUR CLASSIFIERS

# %%
# Predict targets for unseen dataset (x_test) by using all our classifiers
y_pred1 = lr.predict(x_test)
y_pred2 = gs.predict(x_test)
y_pred3 = svclassifier.predict(x_test)
y_pred4 = gb.predict(x_test)
y_pred5 = kn.predict(x_test)
y_pred6 = ad.predict(x_test)
y_pred7 = dt.predict(x_test)
y_pred8 = rf.predict(x_test)


# %%
# Get accuracy score which show the predictive power of each 
#classifier at classifying correct 'outcome'(WHETHER THERE WILL BE AN INSURANCE CLAIM IN 3 MONTHS OR NOT)
print("Testing_score_LogisticRegression : ", accuracy_score(y_test, y_pred1))
print("Testing_score_GaussianNB : ", accuracy_score(y_test, y_pred2))
print("Testing_score_SVC : ", accuracy_score(y_test, y_pred3))
print("Testing_score_GradientBoostingClassifier : ", accuracy_score(y_test, y_pred4))
print("Testing_score_KNeighborsClassifier : ", accuracy_score(y_test, y_pred5))
print("Testing_score_AdaBoostClassifier : ", accuracy_score(y_test, y_pred6))
print("Testing_score_DecisionTree : ", accuracy_score(y_test, y_pred7))
print("Testing_score_randomForest : ", accuracy_score(y_test, y_pred8))

# %% [markdown]
# from the above we can say that most of the classifiers are okay to use since they have consistent test and training scores except random forest and decisiomn tree.for the purpose of this work,i will use gradient boosting classifier

# %%
### Get important features in the training set using our best 
# model - Gradient Boosting Classifier
predictors=list(x_train)
feat_imp = pd.Series(gb.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')

#Re-print accuracy of Gradient Boosting in the training phase.
print('Accuracy of the GBM on training set: {:.2f}'.format(gb.score(x_train, y_train)))

#product name, Age  and car colour are the only important features, 
#but Boosting help us to combine the other weak learners 
#into a single strong learner.


# %%
#Extract the predictions OF OUR  ML CLASSIFIER into a dataframe
gb_results = pd.DataFrame({'y_test': y_test,
                             'y_pred': y_pred4})


# %%
gb_results['y_pred'].value_counts()  


# %%
unique, counts = np.unique(y_pred4, return_counts = True)
print(np.asarray((unique, counts)).T)


# %%
gb_results


# %%
#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred4)
print(cf_matrix)

# %% [markdown]
# k-Fold Cross Validation using the Gradient Boosting Classifier K-fold Cross Validation(CV) divide the data into folds and ensure that each fold is used as a testing set at some point. (This ensures all folds have equal chance of featuring as test set and train set at some point).

# %%
# Evaluate the Gradiant Boosting Classifier With k-Fold Cross Validation
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state=None)
results = cross_val_score(gb, x, y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100,results.std()*100))

# %% [markdown]
# ROC AUC Plots to Evaluate the Performance of all our Classifiers
# %% [markdown]
# #It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a number #of different candidate threshold values between 0.0 and 1.0.

# %%
#calculate values for ROC AUC plot
gb_roc_auc = roc_auc_score(y_test, gb.predict(x_test))
fprgb, tprgb, thresholdsgb = roc_curve(y_test, gb.predict_proba(x_test)[:,1])

lr_roc_auc = roc_auc_score(y_test, lr.predict(x_test))
fprlr, tprlr, thresholdslr = roc_curve(y_test, lr.predict_proba(x_test)[:,1])

gs_roc_auc = roc_auc_score(y_test, gs.predict(x_test))
fprgs, tprgs, thresholdsgs = roc_curve(y_test, gs.predict_proba(x_test)[:,1])

ad_roc_auc = roc_auc_score(y_test, ad.predict(x_test))
fprad, tprad, thresholdsad = roc_curve(y_test, ad.predict_proba(x_test)[:,1])

kn_roc_auc = roc_auc_score(y_test, kn.predict(x_test))
fprkn, tprkn, thresholdskn = roc_curve(y_test, kn.predict_proba(x_test)[:,1])

dt_roc_auc = roc_auc_score(y_test, dt.predict(x_test))
fprdt, tprdt, thresholdsdt = roc_curve(y_test, dt.predict_proba(x_test)[:,1])

rf_roc_auc = roc_auc_score(y_test, rf.predict(x_test))
fprrf, tprrf, threshozldsrt = roc_curve(y_test, rf.predict_proba(x_test)[:,1])

svclassifier_roc_auc = roc_auc_score(y_test, svclassifier.predict(x_test))
fprsv, tprsv, thresholdssv = roc_curve(y_test, svclassifier.predict_proba(x_test)[:,1])


plt.figure(figsize=(10,5))
plt.plot(fprgb, tprgb, label='Gradient Boosting Classifier (area = %0.2f)' % gb_roc_auc)
plt.plot(fprlr, tprlr, label='Logistics Regression Classifier (area = %0.2f)' % lr_roc_auc)
plt.plot(fprgs, tprgs, label='Guassian Naive Bayes Model (area = %0.2f)' % gs_roc_auc)
plt.plot(fprsv, tprsv, label='Support Vector Classifier Model (area = %0.2f)' % svclassifier_roc_auc)
plt.plot(fprad, tprad, label='Adaboost Model (area = %0.2f)' % ad_roc_auc)
plt.plot(fprkn, tprkn, label='K-Nearest Neighbour Model (area = %0.2f)' % kn_roc_auc)
plt.plot(fprdt, tprdt, label='Decision Tree Classifier (area = %0.2f)' % dt_roc_auc)
plt.plot(fprrf, tprrf, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# %% [markdown]
# #ROC AUC: It tells how much model is capable of seperating classes. #Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s. #By analogy, Higher the AUC, better the model is at distinguishing if insurance policy holders will make a claim after a 3 months period.
# %% [markdown]
# Here Again we see that the Gradiant Boosting Classifier has the best ROC AUC (area = 0.9) followed by the logistic regression.

# %%



