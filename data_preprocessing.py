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
    df.dropna(inplace=True)
    return df
