# What's causing the error in our Zestimates

--- 

## Project Goal 

- The goal of this project is to create a reproducible  machine learning regression model and clusters to find the key drivers of error in Zestimates 
- In order to identify key drivers of errors in our Zestimate, this project will explore the data and indentify clusters which can help lower the estimate. This notebook serves as a way to understand why and how I made the prediction model.

---

## Project Description

- The residential real estate market is constantly evolving and fluctating. Zestimates offer a way for our users to to understand and negotiate good deals by having a good idea what the house should be worth. An accurate algorithim is essential in user statisfaction and a way for them to properly plan their home purchase. Lowering the error will inspire more confidence with users trusting the zestimate and help Zillow grow. 

---

### Key Questions

1. Does logerror vary by county?

2. Does log erorr vary if we look distance?

3. Does log error vary by when the house was sold?

4. Does log error vary by cencsus block / square ft?

5. to what extent does the age of the house affect house value? 

---

### How to Replicate the Results

- You will need an env.py file that contains the hostname, username and password of the mySQL database that contains the Zillow table. Store that env file locally in the repository.
- Clone my repo (including the zillow_wrangle.py, and exploration.py) (confirm .gitignore is hiding your env.py file)
- Libraries used are pandas, matplotlib, seaborn, numpy, sklearn.
- You should be able to run final_zillow_prediction

---

### The plan

1. Wrangle the data using the wrangle.py folder and performing a mySQL query to pull multiple tables to find the key drivers of log error into the notebook.
    - The tables I pulled from mySQL are: properties_2017, predicitions_2017, predictions_2016, transactions and propertylandusetype
2. Additionally inside of the wrangle.py folder, I prepped, cleaned, and removed the outliers within the dataset to still include 99.5% of data entries. I also       featured engineered columns to help resolve multicollinearity. 
3. During the exploration phase in explore.ipynb, I visualize multiple features to asses which features to include in the model and to find if the features are statistically significant. Inside my explore notebook, I make multiple models and compare the results against each other to determine the final models to include within the final report.
4. Move and organize all important and useful features to a final notebook to deliver a streamlined product for stakeholders and management.  
5. Deliver a final report notebook with all required .py files

---

### Data Dictionary

Variable | Definiton | 
--- | --- | 
Polar_combo | polar coordinates multiplied together |
--- | --- | 
transaction_quarter | The quarter in which the property sold |
--- | --- | 
transaction_quarter | The month in which the property sold |
--- | --- | 
county_ratio | The county divided by transaction month |
--- | --- | 
polar_combo | multiplication of the polar coordinates |
--- | --- | 
sqft_x_bbr | The total sqaure footage times the bed bath ratio | 
--- | --- | 
Lot_Size | The size of the lot the house sits on |
--- | --- | 
Pool_Encoded | 1 - The property has a pool, 2 - The property does not have a pool |
--- | --- | 
House_Age | How old the house is in years |
--- | --- | 
Garagecarcnt | The count of garage size in 'cars' |
--- | --- | 
rawcensustractandblock | Raw census data |


---

### Exploring the Questions and Hypothesis


1. Does log error vary polar coordinate location?
- Log error varies by county In order to break up the location further, I changed the latitude longitude into polar coordinates to put all the house in two dimensions. This allows for all the homes to be on the same plane to accurate distancing. Binning the data into 6 equal bins shows that there is variance between the log errors and location. 
- Ho Log error does not vary by location | Ha Log error is not equal between locations | We reject the null

2. Does log error vary depending on which quarter the house was sold?
- We can confidently say the transaction quarters have different log error. The model is better at predicting home value in second and fourth quarter.
- Null Hypothesis: log error is independent of transaction quarter. Alt Hypothesis: Log error is dependent on transaction quarter. Result: We reject the null. 

3. Does log error vary by the month it was sold and by census location?
- Three out of the 6 census tracts and blocks measured home value better when taking the transaction month into consideration 
- Null Hypothesis: The log error is equal between transaction month and location. Alt Hypothesis: Log error is not equal between location and transaction month. Result: We reject the null

4. How does log error vary by county and time sold?
- The logerror is different when taking into consideration the county and time sold 
- Null Hypothesis: The logerror is equal for all counties and time sold. Alt: The log error is different between counties and time. Result: We reject the null.


5. Does the product of square feet and the bed bath ratio affect log error?
- As the ratio increases so does the logerror. There is a negative correlation between the house age and home value
- Null Hypothesis: The logerror is the same for bins of square feet * bed bath ratio. Alt hypothesis: The log error is different across all bins. Result: We reject the null. 







---
## Overall Exploration and Testing Takeaways
---
- The log error varies by location and setting it to polar coordinates makes the location easily measurable 
- The quarter in which the home was sold affects the home price
- The time sold and census location affects logerror
- The month when sold and county location affects logerror
- The product of sq ft and bed bath ratio has strong variance in logerror.
- Created the age bin in explore but did not include in questions
* ### There must be some underlying variables affecting the log error during transaction quarter becuase after looking at multiple features, the log error remained constant.
--- 
# Summary
* The tweedie regression model used on the data beat the baseline prediction by 0.05% on the test data
* There is some underlying features that I do not know which is causing an infulence on the logerror
* The top features to help predict the log error are the custom bins I made above.
* My clusters did not add to the model, so I did not select them 

---
# Recommendations
* Try to gain additional features that affect the logerror to determine more key errors
* Try to get better information to lower the amount of nulls for features that have strong correlation, such as construction quality
*  Compare these models to other states to see how logerror is affect and if the features here are unique to the LA area

---
# Next Steps
* Continue to explore the variables and bins of them to help lower log error
* Try new cluster combinations to see if any would help model preform any better
* Proof of concecpt - can continue to build and work on the model. 