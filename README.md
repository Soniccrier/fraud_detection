# fraud_detection
📌 Credit Card Fraud Detection – Machine Learning Project
This project builds a fraud detection model using the popular Credit Card Fraud Dataset.
It demonstrates data preprocessing, handling imbalanced data, model training, and evaluating performance using industry-standard metrics such as ROC-AUC.
The goal is to detect fraudulent transactions from anonymized credit card data.

📂 Project Structure
The notebook performs the following steps:
    1. Load and inspect the dataset
    2. Explore class imbalance
    3. Visualize fraud vs non-fraud counts
    4. Scale important numeric features
    5. Train a Logistic Regression model
    6. Generate predictions
    7. Evaluate the model using:
        ◦ Classification Report
        ◦ Confusion Matrix
        ◦ ROC-AUC Score

📊 Dataset Description
The dataset contains 284,807 credit card transactions with the following columns:
    • Time – Number of seconds elapsed between each transaction
    • V1–V28 – PCA-transformed features (sensitive data anonymized)
    • Amount – Transaction amount
    • Class – Target variable
        ◦ 0 → Normal transaction
        ◦ 1 → Fraudulent transaction
The dataset is highly imbalanced (fraud ≈ 0.17%), which makes evaluation metrics more important than accuracy alone.

🔍 Exploratory Data Analysis (EDA)
    • Checked dataset size, info, and missing values
    • Analyzed distribution of Class (fraud vs non-fraud)
    • Used a Seaborn barplot to visualize the imbalance
Example visualization used:
class_count = df.groupby('Class', as_index=False).size()
class_count.rename(columns={'size': 'count'}, inplace=True)
class_count = class_count.sort_values('count', ascending=False)

plt.figure(figsize=(12,6))
ax = sns.barplot(data=class_count, x='Class', y='count')
for container in ax.containers:
    ax.bar_label(container, fmt='%1.0f', label_type='edge', color='black')

plt.title("Fraud vs Non-Fraud Count")
plt.xlabel("Class (0 = Non-Fraud, 1 = Fraud)")
plt.ylabel("Transaction Count")
plt.show()

⚙️ Data Preprocessing
The columns Amount and Time were scaled using StandardScaler to bring all values into a comparable range:
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df = df.drop(['Amount', 'Time'], axis=1)
This improves model performance and stability.

🤖 Model Training
A Logistic Regression model was trained:
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Why Logistic Regression?
    • Fast to train
    • Works well on linearly separable problems
    • Excellent baseline for fraud detection

📈 Model Evaluation
The model was evaluated using:
    • Classification Report
    • Confusion Matrix
    • ROC-AUC Score
The project achieved:
⭐ ROC-AUC = 0.957
This means the model correctly distinguishes fraud vs. non-fraud 95.7% of the time, which is very strong for a baseline model on an imbalanced dataset.

🚀 Key Takeaways
    • Fraud detection datasets are extremely imbalanced
    • ROC-AUC is a better metric than accuracy
    • Logistic Regression performs surprisingly well
    • Feature scaling improves performance
    • Visualizations help explain fraud patterns

📁 Technologies Used
    • Python
    • Pandas
    • NumPy
    • Seaborn
    • Matplotlib
    • Scikit-learn
