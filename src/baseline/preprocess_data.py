from sklearn.utils import resample
import pandas as pd
import numpy as np

def down_sampling(df, random_state=8964):    
    positive = df[df['diagnosed_diabetes'] == 1]
    negative = df[df['diagnosed_diabetes'] == 0]

    down_sampling = resample(positive, replace=False, n_samples=len(negative), random_state=random_state)
    df = pd.concat([down_sampling, negative])
    print(df['diagnosed_diabetes'].value_counts())
    return df

def remove_outliers(df, high_q = 0.999936657516334, verbose = True):
    low_q  = 1 - high_q

    ncols = df.select_dtypes(include=[np.number]).columns.drop(["diagnosed_diabetes"])

    outliner_data = pd.DataFrame()
    standard_data = df.copy()

    for i, col in enumerate(ncols): 
        q_low  = df[col].quantile(low_q)
        q_high = df[col].quantile(high_q)
        outliers = list(df[df[col] < q_low].index) + list(df[df[col] > q_high].index)
        if verbose: print(f"feature: {col}, outliers: {len(outliers)}")

        if len(outliers) > 0:
            outliner_data = pd.concat([outliner_data, df.loc[outliers]]) 
            standard_data = standard_data.drop(outliers, errors="ignore")
            
    if verbose: print(f"total outliers: {len(outliner_data)}")
    return standard_data, outliner_data