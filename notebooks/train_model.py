"""
Rheumatoid Arthritis Prediction - Machine Learning Model
Random Forest Classifier with comprehensive evaluation and feature importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import joblib
import os
from datetime import datetime

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class RAPredictor:
    """Rheumatoid Arthritis Prediction Model using Random Forest"""
    
    def __init__(self, data_path='../data/ra_dataset.csv'):
        """Initialize the predictor with dataset path"""
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
    def load_data(self):
        """Load and display basic information about the dataset"""
        print("=" * 70)
        print("LOADING DATASET")
        print("=" * 70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\nDataset loaded successfully from: {self.data_path}")
        print(f"Shape: {self.df.shape} (rows, columns)")
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nTarget Distribution:")
        print(self.df['RA_Diagnosis'].value_counts())
        print(f"\nRA Positive: {(self.df['RA_Diagnosis']==1).sum()} ({(self.df['RA_Diagnosis']==1).mean()*100:.1f}%)")
        print(f"RA Negative: {(self.df['RA_Diagnosis']==0).sum()} ({(self.df['RA_Diagnosis']==0).mean()*100:.1f}%)")
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess data: handle missing values, encode categorical features"""
        print("\n" + "=" * 70)
        print("PREPROCESSING DATA")
        print("=" * 70)
        
        # Drop PatientID as it's not a feature
        if 'PatientID' in self.df.columns:
            self.df = self.df.drop('PatientID', axis=1)
        
        # Check for missing values
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Family_History', 'Smoking_Status']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_Encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"\nEncoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Separate features and target
        self.X = self.df.drop(['RA_Diagnosis', 'Gender', 'Family_History', 'Smoking_Status'], 
                              axis=1, errors='ignore')
        self.y = self.df['RA_Diagnosis']
        
        self.feature_names = self.X.columns.tolist()
        print(f"\nFeatures ({len(self.feature_names)}): {self.feature_names}")
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print("\nFeatures scaled using StandardScaler")
        print("Preprocessing complete!")
        
        return self.X_scaled, self.y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "=" * 70)
        print("SPLITTING DATA")
        print("=" * 70)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"\nTraining set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        print(f"Test size: {test_size*100}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators=30, max_depth=4, random_state=42):
        """Train Random Forest Classifier"""
        print("\n" + "=" * 70)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 70)
        
        # Initialize Random Forest
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt'
        )
        
        print(f"\nModel parameters:")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - max_depth: {max_depth}")
        print(f"  - random_state: {random_state}")
        
        # Train the model
        print("\nTraining in progress...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete!")
        
        # Cross-validation
        print("\nPerforming 5-Fold Cross-Validation...")
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        print(f"Cross-Validation Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self.model
    
    def hyperparameter_tuning(self):
        """Perform basic hyperparameter tuning using GridSearchCV"""
        print("\n" + "=" * 70)
        print("HYPERPARAMETER TUNING")
        print("=" * 70)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        print("\nSearching for best parameters...")
        print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance with comprehensive metrics"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        # Predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print("\n" + "-" * 70)
        print("PERFORMANCE METRICS")
        print("-" * 70)
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        
        print("\n" + "-" * 70)
        print("CLASSIFICATION REPORT")
        print("-" * 70)
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['No RA', 'RA Positive']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\n" + "-" * 70)
        print("CONFUSION MATRIX")
        print("-" * 70)
        print(cm)
        print(f"\nTrue Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
        
        # Save metrics
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist()
        }
        
        return self.metrics
    
    def plot_results(self):
        """Generate and save visualizations"""
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Create output directory
        os.makedirs('model/plots', exist_ok=True)
        
        # 1. Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No RA', 'RA'], yticklabels=['No RA', 'RA'])
        plt.title('Confusion Matrix - RA Prediction', fontsize=14, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('model/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: confusion_matrix.png")
        plt.close()
        
        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - RA Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('model/plots/roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: roc_curve.png")
        plt.close()
        
        # 3. Feature Importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=self.feature_importance, x='importance', y='feature', palette='viridis')
        plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('model/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_importance.png")
        plt.close()
        
        # 4. Prediction Distribution
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                label='No RA', color='blue', edgecolor='black')
        plt.hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                label='RA Positive', color='red', edgecolor='black')
        plt.xlabel('Predicted Probability', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.subplot(1, 2, 2)
        labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        values = [cm[0,0], cm[0,1], cm[1,0], cm[1,1]]
        colors = ['#66b3ff', '#ff9999', '#ffcc99', '#99ff99']
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Prediction Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model/plots/prediction_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: prediction_distribution.png")
        plt.close()
        
        print("\nAll visualizations saved to: model/plots/")
    
    def save_model(self):
        """Save trained model and preprocessors"""
        print("\n" + "=" * 70)
        print("SAVING MODEL")
        print("=" * 70)
        
        os.makedirs('model', exist_ok=True)
        
        # Save model
        joblib.dump(self.model, 'model/ra_random_forest_model.joblib')
        print("✓ Saved: ra_random_forest_model.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, 'model/scaler.joblib')
        print("✓ Saved: scaler.joblib")
        
        # Save label encoders
        joblib.dump(self.label_encoders, 'model/label_encoders.joblib')
        print("✓ Saved: label_encoders.joblib")
        
        # Save feature names
        joblib.dump(self.feature_names, 'model/feature_names.joblib')
        print("✓ Saved: feature_names.joblib")
        
        # Save feature importance
        self.feature_importance.to_csv('model/feature_importance.csv', index=False)
        print("✓ Saved: feature_importance.csv")
        
        # Save metrics
        import json
        with open('model/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print("✓ Saved: metrics.json")
        
        print("\nAll model files saved to: model/")
    
    def predict_single(self, patient_data):
        """Make prediction for a single patient"""
        # Convert to DataFrame
        df_input = pd.DataFrame([patient_data])
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in df_input.columns:
                df_input[col + '_Encoded'] = le.transform(df_input[col])
                df_input = df_input.drop(col, axis=1)
        
        # Ensure correct feature order
        df_input = df_input[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df_input)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability': {
                'No_RA': float(probability[0]),
                'RA_Positive': float(probability[1])
            }
        }

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("RHEUMATOID ARTHRITIS PREDICTION - RANDOM FOREST MODEL")
    print("=" * 70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize predictor
    predictor = RAPredictor()
    
    # Load data
    predictor.load_data()
    
    # Preprocess
    predictor.preprocess_data()
    
    # Split data
    predictor.split_data(test_size=0.2)
    
    # Train model
    predictor.train_model(n_estimators=30, max_depth=4)
    
    # Optional: Hyperparameter tuning (uncomment to use)
    # predictor.hyperparameter_tuning()
    
    # Evaluate model
    predictor.evaluate_model()
    
    # Plot results
    predictor.plot_results()
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNext steps:")
    print("1. Check model/plots/ for visualizations")
    print("2. Review model/metrics.json for detailed performance")
    print("3. Use the model in backend/app.py for predictions")
    print("=" * 70)

if __name__ == "__main__":
    main()
