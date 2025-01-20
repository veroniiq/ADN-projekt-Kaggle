## Import potrzebnych bibliotek
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
import morfeusz2

# Import danych z plików csv
train_data = pd.read_csv("pzn-rent-train.csv", sep= ',')
test_data = pd.read_csv("pzn-rent-test.csv", sep= ',')

print(train_data.shape)
print(test_data.shape)

## Funkcja do usunięcia outlierów w zbiorze treningowym
def remove_top_n_values(df, columns, n):
    for col in columns:
        df = df[~df[col].isin(df[col].nlargest(n))]
    return df

# Kolumny, dla których chcemy usunąć outliery
columns_to_filter = ['flat_area', 'flat_rooms', 'flat_rent', 'flat_deposit', 'building_floor_num']

# Usunięcie rekordów z największymi wartościami dla wybranych kolumn
train_data = remove_top_n_values(train_data, columns_to_filter, n=3)

## Połączenie danych przed imputacją
len_train = len(train_data)
len_test = len(test_data)

data = pd.concat([train_data, test_data], sort=False)

print(data.shape)
print(data.head())

# Kopia danych
data_copy = data.copy()

## Zidentyfikowanie braków w danych
data.info()
data.isnull().sum()
print(data.columns[data.isnull().any()])

## Tworzenie histogramów dla danych numerycznych
numerical_columns = data.select_dtypes(include=['number']).columns
numerical_columns = numerical_columns.drop(['id'])

plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    plt.hist(data[col], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Histogram dla {col}')
    plt.xlabel(col)
    plt.ylabel('Częstotliwość')

plt.tight_layout()
plt.show()

## Sprawdzenie wartości występujących w kolumnach liczbowych
def count_column_values(df, column_names):
    results = {}

    for column_name in column_names:
        if column_name not in df.columns:
            raise ValueError(f"Kolumna '{column_name}' nie istnieje w podanej tabeli.")

        value_counts = df[column_name].value_counts()
        result = value_counts.reset_index()
        result.columns = [column_name, 'Count']
        result = result.sort_values(by=column_name, ascending=False)
        results[column_name] = result

    return results

results = count_column_values(data, ['flat_area', 'flat_rooms', 'flat_rent', 'flat_deposit', 'building_floor_num'])
for column, result in results.items():
    print(f"Kolumna: {column}")
    print(result)

## Zamiana wartości ujemnych w flat_area oraz flat_rooms na NaN
def replace_negative_with_nan(df, column_name):
    df[column_name] = df[column_name].where(df[column_name] >= 0, np.nan)
    return df

data = replace_negative_with_nan(data, 'flat_area')
data = replace_negative_with_nan(data, 'flat_rooms')

# Sprawdzenie wartości po zmianach
wyniki = count_column_values(data, ['flat_area', 'flat_rooms', 'flat_rent', 'flat_deposit', 'building_floor_num'])
for kolumna, wynik in wyniki.items():
    print(f"Kolumna: {kolumna}")
    print(wynik)

data.isnull().sum()

## Imputacja kolumn liczbowych
numeric_columns = [col for col in data.select_dtypes(include=['float64', 'int64']).columns if col != 'price']
for col in numeric_columns:
    data[col].fillna(data[col].mean(), inplace=True)

data.isnull().sum()

## Imputacja kolumn binarnych
binary_columns = [col for col in data.columns if data[col].dropna().isin([True, False]).all()]
for col in binary_columns:
    true_prob = data[col].mean()  # Prawdopodobieństwo True
    data[col].fillna(true_prob, inplace=True)

data.isnull().sum()

## Przetwarzanie ad_title do modelu

# Inicjalizacja Morfeusza
morfeusz = morfeusz2.Morfeusz()

# Funkcja do tokenizacji i lematyzacji pojedynczego tekstu
def tokenize_and_lemmatize(text):
    if not isinstance(text, str):
        return []
    # Tokenizacja i analiza morfologiczna
    analysis = morfeusz.analyse(text)
    # Pobranie lematów
    lemmas = [result[2][1] for result in analysis if result[2][2] != 'ign']
    return lemmas

# Tokenizacja i lematyzacja kolumny 'ad_title'
data['tokens'] = data['ad_title'].apply(tokenize_and_lemmatize)

# Pobranie wszystkich unikalnych tokenów
unique_tokens = sorted(set(token for tokens in data['tokens'] for token in tokens))

# Tworzenie kolumn dla każdego unikalnego tokena
for token in unique_tokens:
    data[token] = data['tokens'].apply(lambda tokens: 1 if token in tokens else 0)

# Usunięcie kolumn z tokenami, które występują tylko raz
columns_to_keep = [col for col in unique_tokens if data[col].sum() > 1]
data = data[list(data.columns[:len(data.columns) - len(unique_tokens)]) + columns_to_keep]  # Zachowanie pierwotnych kolumn i wybranych tokenów

# Usunięcie zbędnych kolumn
data = data.drop(columns=['ad_title', 'tokens'])

print(data)

## Przekształcenie dat do przetworzenia przez model
data['date_activ'] = pd.to_datetime(data['date_activ'], dayfirst=True, errors='coerce')
data['date_modif'] = pd.to_datetime(data['date_modif'], dayfirst=True, errors='coerce')
data['date_expire'] = pd.to_datetime(data['date_expire'], dayfirst=True, errors='coerce')

# Ekstrakcja cech
# Długość aktywności ogłoszenia (w dniach)
data["ad_duration_days"] = (data["date_expire"] - data["date_activ"]).dt.days

# Czas od aktywacji do modyfikacji (w dniach)
data["time_to_modif_days"] = (data["date_modif"] - data["date_activ"]).dt.days

# Usunięcie oryginalnych kolumn datowych
data = data.drop(columns=['date_activ', 'date_modif', 'date_expire'])

## Dodanie nowej cechy wykorzystującej kolumnę quarter
data['median_price_per_sqm'] = data.groupby('quarter')['price'].transform('median')/data['flat_area']

## One-hot encoding kolumny quarter
data = pd.get_dummies(data, columns=['quarter'], prefix='quarter')
print(data)

## Dodanie kolejnej nowej cechy
data['area_per_room'] = data['flat_area'] / data['flat_rooms']

## Podzielenie danych po imputacji
data_train = data.iloc[:len_train, :].reset_index(drop=True)
data_test = data.iloc[len_train:, :].reset_index(drop=True)

print(data_train.shape)
print(data_test.shape)

## Podział danych na treningowe i testowe
X_train = data_train.drop(columns=['price', 'id'])
y_train = data_train['price']

X_test = data_test.drop(columns=['price', 'id'])
test_ids = data_test['id']

## Trening modelu
model = CatBoostRegressor(iterations=7000, learning_rate=0.05, depth=10, random_state=42)
model.fit(X_train, y_train, early_stopping_rounds=50)

feature_importance = model.get_feature_importance(prettified=True)
print(feature_importance)

## Predykcja
y_test_pred = model.predict(X_test)

## Przygotowanie wyników do wrzucenia na Kaggle
results = pd.DataFrame({
    'ID': range(1, len(y_test_pred) + 1),
    'TARGET': y_test_pred
})

# Zapis do pliku CSV
results.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'")