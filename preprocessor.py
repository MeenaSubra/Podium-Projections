import pandas as pd


def preprocess(df, region_df):
    # Filter for Summer Olympics
    df = df[df['Season'] == 'Summer']

    # Merge with 'region_df' on the 'NOC' column
    df = df.merge(region_df, on='NOC', how='left')

    # One-hot encode 'Medal' and concatenate it with the original DataFrame
    medal_dummies = pd.get_dummies(df['Medal'], prefix='Medal')
    df = pd.concat([df, medal_dummies], axis=1)

    # You can choose to drop duplicates or handle them differently
    # df.drop_duplicates(inplace=True)

    return df
