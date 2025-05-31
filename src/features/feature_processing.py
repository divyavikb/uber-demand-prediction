import logging
from pathlib import Path
import pandas as pd


# create a logger
logger = logging.getLogger("feature_processing")
logger.setLevel(logging.INFO)

# attach a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

# make a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

if __name__ == "__main__":
    # current path
    current_path = Path(__file__)
    # set the root path
    root_path = current_path.parent.parent.parent
    # data_path
    data_path = root_path / "data/processed/resampled_data.csv"
    
    # read the data
    df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"])
    logger.info("Data read successfully")
    
    # extract the day of week information
    df["day_of_week"] = df["tpep_pickup_datetime"].dt.day_of_week
    # extract the month information
    df["month"] = df["tpep_pickup_datetime"].dt.month
    logger.info("Datetime Features extracted successfully")
    
    # set the datetime column as index
    df.set_index("tpep_pickup_datetime", inplace=True)
    logger.info("Datetime column set as index successfully")
    
    # create the region grouper
    region_grp = df.groupby("region")
    # shifting periods
    periods = list(range(1,5))
    # generate the lag features
    for p in periods:
        df[f"lag_{p}"] = region_grp["total_pickups"].shift(p)

   
    logger.info("Lag features generated successfully")
       
    # drop the missing values
    df.dropna(inplace=True)
    
    logger.info("Column names renamed successfully")
    # split the data into train and test
    trainset = df.loc[df["month"].isin([1,2]),['region', 'total_pickups', 'avg_pickups', 'day_of_week',
       'lag_1', 'lag_2', 'lag_3', 'lag_4']]

    testset = df.loc[df["month"].isin([3]),['region', 'total_pickups', 'avg_pickups', 'day_of_week',
       'lag_1', 'lag_2', 'lag_3', 'lag_4']]
    
    # save the train and test data
    train_data_save_path = root_path / "data/processed/train.csv"
    test_data_save_path = root_path / "data/processed/test.csv"

    trainset.to_csv(train_data_save_path, index=True)
    logger.info("Train data saved successfully")
    
    testset.to_csv(test_data_save_path, index=True)
    logger.info("Test data saved successfully")