import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    df.columns = df.columns.str.strip()
    selected_columns = [
        'Age at enrollment',
        'Curricular units 1st sem (enrolled)',
        'Curricular units 1st sem (evaluations)',
        'Curricular units 1st sem (approved)',
        'Scholarship holder',
        'Target'
    ]

    df = df[selected_columns]
    df = df[df['Target'] != 'Enrolled']
    df['Target'] = df['Target'].map({
        'Graduate': 1,
        'Dropout': 0
    })
    numeric_cols = df.columns.drop('Target')

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    df.dropna(inplace=True)
    print("\n[Preprocessing]")
    print("Shape:", df.shape)
    print("Target distribution:\n", df['Target'].value_counts())

    return df
