import pandas as pd
from sklearn.model_selection import train_test_split

# Define the mapping for body part classes
class_mapping = {
    'XR_ELBOW': 0,
    'XR_FINGER': 1,
    'XR_FOREARM': 2,
    'XR_HAND': 3,
    'XR_HUMERUS': 4,
    'XR_SHOULDER': 5,
    'XR_WRIST': 6
}

# Load the labeled study data
df_train_studies = pd.read_csv('./MURA-v1.1/train_labeled_studies.csv', header=None, names=['path', 'label'])
df_val_studies = pd.read_csv('./MURA-v1.1/valid_labeled_studies.csv', header=None, names=['path', 'label'])

# Extract the body part from the path
df_train_studies['body_part'] = df_train_studies['path'].apply(lambda x: x.split('/')[2])
df_val_studies['body_part'] = df_val_studies['path'].apply(lambda x: x.split('/')[2])

# Add the class column based on the body part
df_train_studies['class'] = df_train_studies['body_part'].map(class_mapping)
df_val_studies['class'] = df_val_studies['body_part'].map(class_mapping)

# Initialize empty DataFrames for the final splits
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()

# Process each body part separately
for body_part in class_mapping.keys():
    print(f"\nProcessing {body_part}...")
    
    # Filter data for the current body part
    train_body_part = df_train_studies[df_train_studies['body_part'] == body_part]
    test_body_part = df_val_studies[df_val_studies['body_part'] == body_part]
    
    # Split the training data into train and validation sets with stratification
    train_subset, val_subset = train_test_split(
        train_body_part,
        test_size=0.2,
        stratify=train_body_part['label'],
        random_state=42
    )
    
    # Print statistics for the current body part
    print(f"  Training set size: {len(train_subset)}")
    print(f"  Validation set size: {len(val_subset)}")
    print(f"  Testing set size: {len(test_body_part)}")
    
    # Print label distribution
    print("  Training label distribution:")
    print(train_subset['label'].value_counts(normalize=True))
    print("  Validation label distribution:")
    print(val_subset['label'].value_counts(normalize=True))
    print("  Testing label distribution:")
    print(test_body_part['label'].value_counts(normalize=True))
    
    # Append to the unified DataFrames
    train_df = pd.concat([train_df, train_subset])
    val_df = pd.concat([val_df, val_subset])
    test_df = pd.concat([test_df, test_body_part])
    
    # Save individual body part splits
    train_subset.to_csv(f'processed_{body_part.lower().replace("xr_", "")}_train.csv', index=False)
    val_subset.to_csv(f'processed_{body_part.lower().replace("xr_", "")}_valid.csv', index=False)
    test_body_part.to_csv(f'processed_{body_part.lower().replace("xr_", "")}_test.csv', index=False)

# Save the unified DataFrames
train_df.to_csv('processed_all_train.csv', index=False)
val_df.to_csv('processed_all_valid.csv', index=False)
test_df.to_csv('processed_all_test.csv', index=False)

# Print statistics for the unified datasets
print("\n=== UNIFIED DATASETS STATISTICS ===")
print(f"Total training set size: {len(train_df)}")
print(f"Total validation set size: {len(val_df)}")
print(f"Total testing set size: {len(test_df)}")

# Print label distribution for each body part in the unified datasets
print("\nLabel distribution by body part in the unified training set:")
for body_part in class_mapping.keys():
    part_df = train_df[train_df['body_part'] == body_part]
    print(f"\n{body_part}:")
    print(f"  Count: {len(part_df)}")
    print(f"  Label distribution: {part_df['label'].value_counts(normalize=True).to_dict()}")
