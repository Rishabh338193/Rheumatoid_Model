"""
Rheumatoid Arthritis Synthetic Dataset Generator
Creates a realistic dataset for RA prediction with clinical and lab features
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_ra_dataset(n_samples=1000):
    """
    Generate synthetic RA dataset with realistic clinical features
    
    Features:
    - Age: 18-85 years
    - Gender: Male/Female
    - Morning Stiffness Duration: 0-180 minutes
    - Joint Pain Score: 0-10
    - Swollen Joint Count: 0-28
    - Rheumatoid Factor (RF): 0-200 IU/ml
    - Anti-CCP: 0-300 U/ml
    - ESR (Erythrocyte Sedimentation Rate): 0-100 mm/hr
    - CRP (C-Reactive Protein): 0-50 mg/L
    - Fatigue Score: 0-10
    - Family History: Yes/No
    - Smoking Status: Yes/No
    
    Target: RA_Diagnosis (0=No, 1=Yes)
    """
    
    print("Generating Rheumatoid Arthritis Dataset...")
    print("=" * 60)
    
    data = []
    
    for i in range(n_samples):
        # Determine if patient has RA (40% prevalence)
        has_ra = np.random.choice([0, 1], p=[0.6, 0.4])
        
        if has_ra == 1:
            # RA positive patients - higher values for clinical markers
            age = np.random.randint(35, 75)  # Peak age for RA
            gender = np.random.choice(['Female', 'Male'], p=[0.70, 0.30])  # More common in females
            morning_stiffness = np.random.randint(20, 180)  # More variability
            joint_pain = np.random.randint(4, 11)  # Higher pain scores
            swollen_joints = np.random.randint(2, 28)  # Multiple joints affected
            rf = np.random.uniform(15, 200)  # Elevated RF with more variance
            anti_ccp = np.random.uniform(15, 300)  # Elevated Anti-CCP
            esr = np.random.uniform(15, 100)  # Elevated ESR
            crp = np.random.uniform(3, 50)  # Elevated CRP
            fatigue = np.random.randint(5, 11)  # High fatigue
            family_history = np.random.choice(['Yes', 'No'], p=[0.55, 0.45])
            smoking = np.random.choice(['Yes', 'No'], p=[0.5, 0.5])
            
        else:
            # RA negative patients - normal/lower values with more overlap
            age = np.random.randint(18, 85)
            gender = np.random.choice(['Female', 'Male'], p=[0.55, 0.45])
            morning_stiffness = np.random.randint(0, 60)  # More overlap
            joint_pain = np.random.randint(0, 7)  # More overlap
            swollen_joints = np.random.randint(0, 6)  # More overlap
            rf = np.random.uniform(0, 35)  # Increased upper limit
            anti_ccp = np.random.uniform(0, 35)  # Increased upper limit
            esr = np.random.uniform(0, 35)  # Increased upper limit
            crp = np.random.uniform(0, 12)  # Increased upper limit
            fatigue = np.random.randint(0, 7)  # More overlap
            family_history = np.random.choice(['Yes', 'No'], p=[0.25, 0.75])
            smoking = np.random.choice(['Yes', 'No'], p=[0.3, 0.7])
        
        # Add substantial noise to make the problem more challenging (more realistic)
        if np.random.random() < 0.35:  # 35% of cases have swapped or noisy labels
            if has_ra == 0:
                # Some healthy people have elevated markers
                rf = np.random.uniform(15, 80)
                anti_ccp = np.random.uniform(15, 100)
                esr = np.random.uniform(10, 50)
                crp = np.random.uniform(3, 20)
                # Occasionally flip the label for more challenge
                if np.random.random() < 0.15:
                    has_ra = 1
            else:
                # Some RA patients have lower markers
                rf = np.random.uniform(5, 40)
                anti_ccp = np.random.uniform(5, 50)
                esr = np.random.uniform(5, 25)
                crp = np.random.uniform(1, 8)
                # Occasionally flip the label for more challenge
                if np.random.random() < 0.15:
                    has_ra = 0
        
        data.append({
            'PatientID': f'P{i+1:04d}',
            'Age': age,
            'Gender': gender,
            'Morning_Stiffness_Duration': int(morning_stiffness),
            'Joint_Pain_Score': joint_pain,
            'Swollen_Joint_Count': swollen_joints,
            'Rheumatoid_Factor': round(rf, 2),
            'Anti_CCP': round(anti_ccp, 2),
            'ESR': round(esr, 2),
            'CRP': round(crp, 2),
            'Fatigue_Score': fatigue,
            'Family_History': family_history,
            'Smoking_Status': smoking,
            'RA_Diagnosis': has_ra
        })
    
    df = pd.DataFrame(data)
    
    # Save dataset
    output_path = 'ra_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {n_samples}")
    print(f"RA Positive: {df['RA_Diagnosis'].sum()} ({df['RA_Diagnosis'].mean()*100:.1f}%)")
    print(f"RA Negative: {(df['RA_Diagnosis']==0).sum()} ({(df['RA_Diagnosis']==0).mean()*100:.1f}%)")
    print(f"\nDataset saved to: {output_path}")
    print("=" * 60)
    
    # Display sample data
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    print("\nDataset statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    df = generate_ra_dataset(n_samples=1000)
    print("\nDataset generation complete!")
