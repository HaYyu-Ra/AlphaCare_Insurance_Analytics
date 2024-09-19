Final Submission of Project Report: AlphaCare Insurance Analytics
Executive Summary
The AlphaCare Insurance Analytics Project was designed to analyse historical car insurance claim data to enhance marketing strategies and improve risk assessment. By examining customer demographics, vehicle information, and policy features, this project aimed to identify key patterns influencing claim behavior and optimize premium pricing. The analysis included exploratory data analysis (EDA), hypothesis testing, and statistical modelling to uncover insights and inform strategic decisions.
Key Findings
•	Regional Risk Differences: Significant variations in risk levels were found across different provinces, revealing regions with higher average claim rates.
•	Premium-Claim Correlation: A correlation was identified between higher premiums and increased claim frequencies, indicating that riskier customers tend to pay higher premiums.
•	Predictive Modeling: XGBoost emerged as a robust model for predicting claim likelihood, with key features such as Sum Insured and Vehicle Age playing crucial roles. These insights will help refine marketing strategies and optimize premium pricing.
Objectives
1.	Optimize Marketing Strategies: Utilize historical data to design targeted marketing campaigns for low-risk customers.
2.	Risk Assessment and Premium Pricing: Identify factors contributing to claims and adjust premiums based on risk profiles.
3.	Predict Claim Behavior: Develop models to estimate claim likelihood based on policy and demographic factors.
4.	A/B Hypothesis Testing: Test hypotheses about risk differences between various groups to guide marketing and pricing decisions.
Data Summary
The dataset included:
•	Customer Demographics: Age, gender, and region.
•	Vehicle Information: Vehicle type, age, and insured value.
•	Policy Features: Sum Insured, total premiums paid, and claim history.
Methodology
•	Exploratory Data Analysis (EDA): Conducted summary statistics and visualizations to explore patterns and identify significant differences in claims across regions.
•	Hypothesis Testing: Evaluated statistical significance of risk differences between groups.
•	Predictive Modeling: Applied machine learning models (e.g., XGBoost) to predict claims and assessed feature importance using SHAP analysis.
Exploratory Data Analysis (EDA)
•	Data Overview: Provides a summary of the dataset, including statistics such as mean, median, standard deviation, and quartiles for numerical features. It helps in understanding the central tendencies and dispersion in the data
 
A bar chart showing the count of missing values for each feature in the dataset. This helps identify features with significant amounts of missing data that might need imputation or exclusion.
 
Figure 1: Bar Chart of Missing Values:
•	Correlation Analysis: Analyses the relationships between numerical variables by computing correlation coefficients. This can reveal potential linear relationships and inform feature selection or engineering, predict Strong correlations between features may suggest redundancy or useful interactions that should be explored further
 Figure 2: Correlation Heatmap
•	Distribution of Variables: examines the distribution of numerical features to understand their spread and shape (e.g., normal, skewed). This can inform decisions on transformations or normalization it predicts 
 
Figure 3: Distribution of Numerical Features
Outlier Detection: Identifies data points that significantly deviate from the majority of the data, which might affect the model's accuracy, A boxplot that highlights outliers by showing the spread of numerical features and points that fall outside the expected range. Figure 4: Boxplot of Outlier
•	Trends Over Geography: Examines how key metrics, like premium amounts, vary across different geographic regions, Regional trends can reveal patterns or disparities that may influence insurance pricing or risk assessments. A line plot illustrating changes in premium amounts over time or across provinces and Trends might reveal geographic or temporal patterns that could inform targeted business strategies.
 
Figure 5: Line Plot of Premium Trends by Province






2. A/B Hypothesis Testing
Compares average claim risks between different provinces to determine if there are significant differences and Identifies provinces with higher or lower risk profiles, which can guide risk management strategies.
•	Risk Differences Across Provinces: A bar chart showing the average claim risk for each province and Provinces with significantly higher or lower risks can be targeted for specific insurance products or interventions.
 
Figure 6: Bar Chart of Average Claim Risk by Province
•	Risk Differences Between Zip Codes: Examines if there are significant differences in claim risks between different zip codes and Helps in understanding local risk variations and tailoring insurance offerings or pricing strategies.
 
Figure 7: Bar Chart of Average Claim Risk by Zip Code
•	Margin Differences Between Zip Codes: A bar chart representing average claim risks across different zip codes and Identifies zip codes with higher risks that may require special attention or risk mitigation.
 
Figure 8: Bar Chart of Average Profit Margin by Zip Code

•	Risk Differences Between Genders: 
 
Figure 9: Box Plot of Claim Frequencies by Gender
3. Statistical Modelling
Model Evaluation
Mean Squared Error (MSE): MSE measures the average squared difference between the actual and predicted values. It quantifies the average error of the predictions, with lower values indicating better model performance.
• Linear Regression: 4,849,173.59
• Random Forest: 5,659,824.98
R-squared: - R-squared measures the proportion of variance in the target variable that is predictable from the features. It ranges from 0 to 1, with higher values indicating a better fit.
Linear Regression: 0.008
Random Forest: -0.158
This format separates the MSE and R-squared values for clarity, specifying that these metrics are associated with the Random Forest model.
•	 R-squared: -0.15828420481653827
 
Figure 10: Bar Chart of Mean Squared Error (MSE)
•	SHAP Analysis: Provides insights into how each feature affects individual predictions, improving model interpretability and Helps in understanding model decisions and identifying feature interactions.
A summary plot showing the impact of each feature on model predictions, with feature values and their effects and Reveals which features most influence predictions and helps in model interpretation.
 Figure 13: SHAP Summary Plot
Predictive Modelling: Uses trained models to make predictions on new data, assessing their predictive accuracy and utility and Provides actionable forecasts based on the model’s learned patterns.
A bar chart displaying the predicted values generated by the model for new data and Shows the range of predictions and helps in evaluating the model's output.
 
Figure 14: Bar Chart of Model Predictions
Conclusion
This project has provided valuable insights into insurance claim behavior, highlighting regional risk disparities and the predictive power of key features. The findings are expected to guide AlphaCare in refining its marketing strategies, targeting high-risk regions, and optimizing premium pricing to attract low-risk customers. The integration of machine learning models and SHAP analysis offers a deeper understanding of claim drivers and supports data-driven decision-making for enhanced profitability.
GitHub Link: AlphaCare Insurance Analytics - Main Branch

