import warnings

import pandas as pd

from utils import print_df_to_table

warnings.filterwarnings("ignore", category=Warning)


def show_count(df, class_col):
    print("[INFO] Class Distribution")
    cvc = df[class_col].value_counts(sort=False)
    sdf = cvc.to_frame()
    sdf.insert(0, "Class", cvc.index)
    sdf.columns = ["Class", "Count"]
    print_df_to_table(sdf)
    return df


def load_data():
    df = pd.read_csv("Data\data.csv")
    print("[INFO] Data Shape :: {0}".format(df.shape))
    return df


if __name__ == "__main__":
    load_data()
