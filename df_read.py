import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('ophthalmoscope_v3.csv')
    print(df['quality'].value_counts())