import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()

def check_df(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

from datetime import datetime
columns_to_convert = df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]]

for col in columns_to_convert:
    df[col] = pd.to_datetime(df[col])

df.groupby("order_channel").agg({"master_id": "count", "total_order_num": "sum", "total_customer_value": "sum"})

df[["master_id", "total_customer_value"]].sort_values(by= "total_customer_value", ascending=False)[:10]

df[["master_id", "total_order_num"]].sort_values(by= "total_order_num", ascending=False)[:10]

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1) #Gerçek hayatta analiz tarihini veriyi temin eden firma belirler.

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date.max()).days,
                             "total_order_num": lambda total_order_num: total_order_num,
                             "total_customer_value": lambda total_customer_value: total_customer_value})

rfm.columns = ["recency", "frequency", "monetary"]

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm["recency_score"]).astype(str) + (rfm["frequency_score"]).astype(str)

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

rfm.groupby("segment").agg({"recency": "mean",
                            "frequency": "mean",
                            "monetary": "mean"})

df = df.reset_index(drop=True)
rfm = rfm.reset_index(drop=True)

rfm["interested_in_categories_12"] = df["interested_in_categories_12"]
rfm["master_id"] = df["master_id"]

yeni_marka_hedef_müşteri_id = rfm.loc[(rfm["segment"].isin(["champions", "loyal_customers"])) & (rfm["interested_in_categories_12"] == "[KADIN]"), ["master_id"]]

yeni_marka_hedef_müşteri_id.to_csv("yeni_marka_hedef_müşteri_id.csv")

indirim_hedef_müşteri_ids = rfm.loc[(rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])), ["master_id"]]

indirim_hedef_müşteri_ids.to_csv("indirim_hedef_müşteri_ids.csv")