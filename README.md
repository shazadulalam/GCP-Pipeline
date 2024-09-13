# Customer Purchase Prediction

This repository contains the project for predicting customer purchase behavior, specifically focused on a selection of brands. The main objective is to assign a score to each customer, representing the probability that they will purchase a product from one of the targeted brands during their next transaction.

### Target Brands
The following brands are the focus of the special offer, and the prediction will be based on identifying which customers are most likely to purchase their next product from these brands:
- Allegra K
- Calvin Klein
- Carhartt
- Hanes
- Volcom
- Nautica
- Quiksilver
- Diesel
- Dockers
- Hurley

### Objective
The objective is to predict which brand a customer will purchase in their next transaction. To achieve this, a score will be assigned to each customer that reflects the probability of buying from the brands listed above.

### Querying the BigQuery

The following SQL query is used to fetch data from the public BigQuery dataset:

```sql
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
FROM `bigquery-public-data.thelook_ecommerce.order_items` AS transactions
LEFT JOIN `bigquery-public-data.thelook_ecommerce.users` AS users
ON transactions.user_id = users.id
LEFT JOIN `bigquery-public-data.thelook_ecommerce.products` AS products
ON transactions.product_id = products.id
WHERE transactions.status <> 'Cancelled';
```

### Approach

1. **Grouping Purchases Chronologically**
   - Group each customer's purchases in chronological order.

2. **Feature and Target Selection**
   - For customers with more than one purchase, use the **N-1 purchases** as the features and the **Nth purchase** as the target.

3. **Exclusion of Single-Purchase Customers**
   - Customers who have only made a single purchase are excluded from the dataset because at least two purchases are needed: one for features and another for the target.

---

### Files and Folders

- `data/`: Contains the raw and processed datasets. The dataset was created using BigQuery’s public dataset `thelook_ecommerce` and saved as `.csv` files, which were used for the analysis.
- `src/`: Source code for preprocessing, feature extraction, model training, and evaluation.
- `output/`: Trained machine learning models saved for further use.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and modeling experiments.
- `README.md`: This file, explaining the objectives, approach, and organization of the project.

---

### Dependencies

- Google-Cloud-BigQuery
- Python 3.7 or higher
- Pandas
- Scikit-learn
- CatBoost
- TensorFlow (optional)

You can install the required packages using:

```bash
pip install -r requirements.txt
```

### Usage

#### Clone the repository:

```bash
git clone https://github.com/shazadulalam/GCP-Pipeline.git
```

### Preprocess the data:

```bash
python src/preprocess.py
```
### Train the model:

```bash
python src/train.py
```

### Evaluate the model:

```bash
python src/evaluate.py
```

### Docker:

Prepare your [Dockerfile](https://github.com/yourusername/yourrepository/blob/main/Dockerfile) file accordingly. To set the Docker environment, execute the following command lines:

```bash
export PROJECT_ID=YOUR_GCP_PROJECT_ID
export IMAGE_NAME=NAME_OF_YOUR_IMAGE
export IMAGE_TAG=latest #IMAGE TAG
export IMAGE_URI=LOCATION_OF_YOUR_ARTIFACT_REGISTRY;example{eu.gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}}
```

### Build and push the Docker Image: Once the environment variables are set, push the Docker image to Google Artifact Registry.

```bash
gcloud builds submit --tag $IMAGE_URI . 
```
