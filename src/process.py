from sklearn.model_selection import train_test_split
from google.cloud import bigquery
import pandas as pd

def create_dataset():
    # Create a BigQuery client instance
    client = bigquery.Client()

    query = """
        SELECT
        transactions.user_id,
        products.brand,
        products.category,  
        products.department,
        products.retail_price,
        users.gender,
        users.age,
        users.created_at,
        users.country,
        users.city,
        transactions.created_at  
        FROM `bigquery-public-data.thelook_ecommerce.order_items` as transactions
        LEFT JOIN `bigquery-public-data.thelook_ecommerce.users` as users
        ON transactions.user_id = users.id
        LEFT JOIN `bigquery-public-data.thelook_ecommerce.products` as products
        ON transactions.product_id = products.id
        WHERE transactions.status <> 'Cancelled';  -- Make sure to use the correct alias 'transactions'
    """

    # Setting up the bqquery
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
    query_job = client.query(query, job_config=safe_config)

    # API request - run the query, and return a pandas DataFrame
    df = query_job.to_dataframe() 
    print(f"{len(df)} rows retrieved")

    # Predicting customers based on which brand will they buy in the next purchase
    recurrent_customers = df.groupby('user_id')['created_at'].count().to_frame("n_purchases")

    # Merging with dataset and filter those with more than 1 purchase
    df = df.merge(recurrent_customers, left_on='user_id', right_index=True, how='inner')
    df = df.query('n_purchases > 1')

    # Fill missing values
    df.fillna('NA', inplace=True)

    target_brands = [
        'Allegra K', 
        'Calvin Klein', 
        'Carhartt', 
        'Hanes', 
        'Volcom', 
        'Nautica', 
        'Quiksilver', 
        'Diesel',
        'Dockers'
    ]

    aggregation_columns = ['brand', 'department', 'category']

    # Group purchases by user chronologically
    df_agg = (df.sort_values('created_at')
            .groupby(['user_id', 'gender', 'country', 'city', 'age'], as_index=False)[['brand', 'department', 'category']]
            .agg({k: ";".join for k in ['brand', 'department', 'category']})
            )

    # Creating the target
    df_agg['last_purchase_brand'] = df_agg['brand'].apply(lambda x: x.split(";")[-1])
    df_agg['target'] = df_agg['last_purchase_brand'].isin(target_brands)*1

    df_agg['age'] = df_agg['age'].astype(float)

    # Removing last item of sequence features to avoid target leakage:
    for col in aggregation_columns:
        df_agg[col] = df_agg[col].apply(lambda x: ";".join(x.split(";")[:-1]))
    
    # Dropping unnecessary features
    df_agg.drop('last_purchase_brand', axis=1, inplace=True)
    df_agg.drop('user_id', axis = 1, inplace = True)

    return df_agg


def split_data(df_agg, output_dir):
    # Splitting data into train, validation, and test sets
    df_train, df_val = train_test_split(df_agg, stratify=df_agg['target'], test_size=0.2)
    print(f"{len(df_train)} samples in train")

    df_val, df_test = train_test_split(df_val, stratify=df_val['target'], test_size=0.5)
    print(f"{len(df_val)} samples in val")
    print(f"{len(df_test)} samples in test")

    # Saving the datasets to CSV
    train_path = f"{output_dir}/train.csv"
    val_path = f"{output_dir}/validation.csv"
    test_path = f"{output_dir}/test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"Train, validation, and test datasets saved to {output_dir}")

    return train_path, val_path, test_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the CSV files")
    args = parser.parse_args()
    output_dir = "/home/forhad/Study/Self_Work/GCP/GCP-Pipeline/GCP-Pipeline/dataset"

    # Create dataset from BigQuery
    df_agg = create_dataset()

    # Split data and save as CSV files
    split_data(df_agg, args.output_dir)
