# Loan_status_prediction

visit : https://loanstatusprediction1.streamlit.app/

1. **How was the project done?**

   The project involved collecting data on individuals' physical attributes and exercise details, such as age, gender, height, weight, exercise duration, heart rate, and body temperature. This data was then used to train a machine learning model to predict the number of calories burned during exercise.

2. **What algorithm was used and why?**

   The Random Forest Regressor algorithm was used. It's effective for regression tasks because it combines multiple decision trees to improve prediction accuracy and control over-fitting. 

3. **Were there alternative algorithms, and why was this one preferred?**

   Yes, alternatives like Linear Regression and XGBoost Regressor were considered. However, Random Forest was preferred due to its robustness in handling various data types and its ability to capture complex patterns in the data. 

4. **Who benefits from completing this project?**

   This project is beneficial for individuals tracking their fitness progress, health professionals monitoring patients' exercise routines, and developers creating fitness applications.

5. **What are the steps required to do the project from start to end?**

   - Collect relevant data (e.g., age, gender, height, weight, exercise duration, heart rate, body temperature).

   - Preprocess the data (handle missing values, encode categorical variables).

   - Split the data into training and testing sets.

   - Train the machine learning model using the training set.

   - Evaluate the model's performance using the testing set.

   - Deploy the model in a user-friendly interface, such as a web application. 

6. **What are the input parameters, and how are they used to predict calories?**

   The input parameters include gender, age, height, weight, exercise duration, heart rate, and body temperature. These factors are used by the model to estimate the number of calories burned during exercise. 

7. **What is the accuracy of this model compared to other algorithms?**

   The Random Forest Regressor model demonstrated a mean absolute error (MAE) of 2.71, indicating high accuracy. In comparison, a Linear Regression model had a higher MAE of 8.38, showing that Random Forest provided more precise predictions in this context. citeturn0search5

