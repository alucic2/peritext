import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils import resample


def main():

    dt_47 = pd.read_csv(r"result7.csv")
    dt_47 = dt_47[["text", "label"]]
    dt_47.head()
   
    
    dt_paratext = dt_47[dt_47["label"] == 0]
    dt_main  = dt_47[dt_47["label"] == 1]
    print(dt_paratext.shape)
    print(dt_main.shape)
    
    
    dt_main = resample(dt_main,
             replace=True,
             n_samples=len(dt_paratext),
             random_state=42)

    print(dt_main.shape)
    
    data_downsampled = pd.concat([dt_main, dt_paratext])
    data_downsampled.to_csv("result_downsampled.csv")

    print(data_downsampled["label"].value_counts())

    
    fig = plt.figure(figsize=(8, 6))
    data_downsampled.groupby('label').size().plot(kind='pie',
                                       y = "v1",
                                       label = "Type",
                                       autopct='%1.1f%%')
    fig.savefig("resampled_dt.png")
    
                                       
main()