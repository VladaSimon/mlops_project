import argparse
import os
import pickle
from pathlib import Path

import boto3
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def read_s3_csv():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["KEY_ID"],
        aws_secret_access_key=os.environ["ACCESS_KEY"],
        region_name=os.environ["S3_REGION"],
    )

    response = s3.get_object(
        Bucket=os.environ["S3_BUCKET"],
        Key=os.environ["S3_KEY"]
    )

    return pd.read_csv(response["Body"])

def main():
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument("--output-dir", default="/artifacts")
    args = parser.parse_args()

    df_s3 = read_s3_csv()
    df = df_s3.drop_duplicates()

    X = df[["latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights", "neighbourhood_cleansed", "property_type", "room_type"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_columns = ["latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights"]
    category_columns = ["neighbourhood_cleansed", "property_type", "room_type"]

    numeric_pipeline = Pipeline([('fill', SimpleImputer(strategy='median'))])

    categorical_pipeline = Pipeline([('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('numbers', numeric_pipeline, numeric_columns), ('category', categorical_pipeline, category_columns)], remainder='drop')

    final_pipeline = Pipeline([('prepare', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))])

    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2:", round(r2, 3))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "model.pkl").open("wb") as file:
        pickle.dump(final_pipeline, file)

if __name__ == "__main__":
    main()