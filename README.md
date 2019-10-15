# Click_Through_Rate_Prediction_Taobao
Click Through Rate Prediction using Taobao Dataset

## Objectives
The objective of the project is to create a classification model to predict the click through rate based on the data from the website of Taobao for 8 days of ad impression / click logs.

# Data Source(s)
Ad impression/click logs for 1.1 million users from May 5th 2017 to May 13th 2017, involving 0.8 million ads
User behavior logs dataset for 22 days includes 723 million entries for the behavior of the users covered in the ad impression/click logs https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

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

