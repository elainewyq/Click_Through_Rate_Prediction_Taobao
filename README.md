# Click_Through_Rate_Prediction_Taobao
Click Through Rate Prediction using Taobao Dataset 

## Objectives
Online advertising is dominant in marketing in this digital world. Click through rate is the ratio of clicks on a specific ad to the total number of views, which is a very important metric to evaluate ad performance. The objective of the project is to predict the probability of whether an user will click the displayed ad when the user is visiting a page. 

## Data Source
The data is from the website of Taobao for 8 days of ad impression / click logs. The ad impression/click logs for 1.1 million users is from 2017-05-05 16:00:00 to 2017-05-13 15:59:46, involving 0.8 million ads.
User behavior logs dataset covers all the involved 1.1 million users' behavior history for the 22 days prior to 2017-5-13, and includes 723 million entries in total. Link to data https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

### Data Schema

Impression/click log
* User ID
* Adgroup ID
* Time: time_stamp of the log
* PID: the position where the ad is presented in the webpage
* Clk: the user clicked the ad or not

Ad features:
* Cate_ID: the ad category id
* Campaign_ID: the ad campaign id
* Customer: the advertiser's id (advertised item supplier)
* Brand: the brand id of the customer
* Price: the price of the advertised item

User features:
* Cms_segid: user micro group id
* Cms_group_id: user cms group id
* Final_gender_code: gender - 1 for male , 2 for female
* Age_level: age_level
* Pvalue_level: Consumption grade - 1: low,  2: mid,  3: high 
* Shopping_level: Shopping depth - 1: shallow user, 2: moderate user, 3: depth user
* Occupation: is the user a college student 1: yes, 0: no
* New_user_class_level: City level based on the population size of the city

Below is the data struction:
<p align="center">
  <image src=Visualization/CTRDataStructure.png />
</p>

## Data Analysis

### Page View Dataset 
The analysis below is based on the entire impression/click logs dataset.

#### Page views per user
The distribution of page views per user is right skewed. On average, a user has 23 page views, while the median of page views is 7. The data indicates 87% of the users have less than 8 page views. In addition, we group the dataset by 'user id' and 'adgroup id', and find 22% of a user-adgroup combination is repeated, which means some users have visited a page for more than one time.
<p align="center">
  <image src=Visualization/plots_with_complete_data/DistributionOfViewsPerUser_all_data.png />
</p>

#### Views per ad
The distribution of page views per ad is also right skewed, with a average reviews of 31 and a median of 4. It shows that 82% of the ads get less than 20 views. 
<p align="center">
  <image src=Visualization/plots_with_complete_data/DistributionOfViewsPerAd_all_data.png />
</p>

#### Click through rate
Click through rate is the ratio of the number of clicks to the number of views an ad got. The overall click through rate is 5.1%. Below is a snappet of the CTR calculated and the plot of the distribution of the CTR. In ad level, it shows 72% ads have extremely low CTR. We also calculated the CTR measured in campaign, advertiser and brand levels. They are slightly higher than that in ad level, but still fairly low. That may be because there are so many products in Taobao and user's attension is farily "sparse". 
<p align="center">
  <image src=Visualization/plots_with_complete_data/CTRCalc_all_data.png />
</p>
<p align="center">
  <image src=Visualization/plots_with_complete_data/CTRDensityDistribution_all_data.png />
</p>
<p align="center">
  <image src=Visualization/plots_with_complete_data/CTR_Diff_Level_all_data.png />
</p>

#### Relationship between features and click
In order to predict the possibility of whether a user will click an ad, we need to appropriately select features with predictive power from available dataset. We have 16 features in total. They are 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand', 'price', 'cms_segid', 'cms_group_id','final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level ', 'hour'. Except for 'price', all features are categorical features.

From the plot below, we see the distributions of the features 'shopping_level' and 'occupation' for the two groups (click and no-click) are highly overlapped and the distribution of other features, to some extent, are showing difference between the two groups.
<p align="center">
  <image src=Visualization/plots_with_complete_data/Distribution_of_Features_in_clk_noclk_groups.png />
</p>

#### Chi-squared test
In addition to the analysis above, we performed chi-square test to asist the feature selection. Chi-squared test is used to determine if there is a difference between the two variables. The null and alternative hypothesis is:
* H_0: A given features is not associated with whether an ad will be clicked.
* H_1: A given feature is associated with whether an ad will be clicked.

The significance level is set to be 0.05. Since we have 15 categorical features, the p_value should be less than 0.003 (0.05/15) to reject the null hypothesis. Based on the test result presented below, we select the following 10 features - 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'brand' 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'shopping_level' and 'hour'.
<p align="center">
  <image src=Visualization/plots_with_complete_data/Chi_squared_all_data.png />
</p>

### User Behavior Dataset
The analysis below is based on 7 million user behavior logs (1% of the entire dataset).

#### Behavior in user level

From the distribution, we can see 30% of the users only visited one category and one brand in the dataset.
<p align="center">
  <image src=Visualization/Number_of_categories_an_user_interacts_with.png />
</p>
<p align="center">
  <image src=Visualization/Number_of_brands_an_user_visit.png />
</p>

#### Behavior in brand/category level
Below is the distribution of the conversion rate of each brand and each category from 'browse' to 'buy'. We can see the conversion rate is fairly low, this may be due to the short period of time (we only extracted 1% of the 22 days logs). There is 35% of brands with zero conversion and 90% of brands with a conversion rate less than 20%. In category level, there is 42% of categories with zero conversion and 95% of categories with a conversion rate less than 20%.
<p align="center">
  <image src=Visualization/Conversion_Browse_Buy.png />
</p>

## Models
* Logistic Regression
* Decision Tree
* Random Forest

### Evaluation metric
To evaluate the performance of each model, we used the ROC AUC score, also take into account the confusion matrix since the data is highly imbalanced.
### Challenges
1. feature engineering - when plotting the distribution, most of the features barely show any difference between click and nonclick groups. (add most of the features, observe the baseline model's feature importance to take out unimportant features (e.g. hour, occupation, pvalue_level, etc) ) 
2. Sparsed matrix - Except for the price, all the features are categorical. (create dummy variable for selected feature and read each line as dictionary so that only the col with 1 is saved to reduce the memory requirement.
3. Imbalanced data - the ratio of positive and negative labels is 5% to 95%, meaning 5% click in total. It requires us to manually generate fairly balanced data for model training and use the original imbalanced data for model test.

## Reference
* Chi-squared test for nominal (categorical) data http://learntech.uwe.ac.uk/da/Default.aspx?pageid=1440
* https://github.com/moebg/click-through-rate-prediction
* https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d
* https://github.com/moebg/click-through-rate-prediction/blob/master/src/ctr_prediction.py
* Methods for dealing with imbalanced data https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18





