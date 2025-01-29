import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\francesco_Manco\data\df_MV-ASD_clr_combined.csv')

# Assuming the last column is the target variable
X = data.drop(['target'], axis=1)
y = data['target']

# Initialize the Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,oob_score=True)

# Setup Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize results dictionary
mean_results = {
    'macro avg precision': [], 
    'macro avg recall': [], 
    'macro avg f1-score': [],
    'weighted avg precision': [], 
    'weighted avg recall': [], 
    'weighted avg f1-score': []
}


# Perform cross-validation with detailed metrics
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    #applica la PCA e addestra il random forest
    pca = PCA(n_components=10, svd_solver='auto',random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    # Train and predict
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True,zero_division=0)
    
    # Store metrics
    mean_results['macro avg precision'].append(report['macro avg']['precision'])
    mean_results['macro avg recall'].append(report['macro avg']['recall'])
    mean_results['macro avg f1-score'].append(report['macro avg']['f1-score'])
    mean_results['weighted avg precision'].append(report['weighted avg']['precision'])
    mean_results['weighted avg recall'].append(report['weighted avg']['recall'])
    mean_results['weighted avg f1-score'].append(report['weighted avg']['f1-score'])

# Calculate and save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric': list(mean_results.keys()),
    'Mean': [np.mean(values) for values in mean_results.values()],
    'Std': [np.std(values) * 2 for values in mean_results.values()]
})

# Save to CSV
metrics_df.to_csv(r'C:\Users\Utente\Desktop\SAE-microbiome\results\random_forest_combined_PCA.csv', index=False)

# Print metrics
print("\nCross-validation metrics:")
for metric, values in mean_results.items():
    print(f"{metric}: {np.mean(values):.3f} (+/- {np.std(values) * 2:.3f})")


