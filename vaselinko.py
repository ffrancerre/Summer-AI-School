import os
import json
import re
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import GPT2Tokenizer, GPT2Model
import torch
from torch.utils.data import Dataset, DataLoader

# Завантаження необхідних ресурсів NLTK
nltk.download('punkt')

# Функція для очищення тексту
def clean_text(text):
    # Видаляємо HTML теги
    text = re.sub(r'<.*?>', '', text)
    # Видаляємо спеціальні символи (залишаємо тільки букви, цифри і пробіли, включаючи кирилицю)
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z0-9\s]', '', text)
    # Видаляємо зайві пробіли
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Функція для токенізації тексту на слова і фрази
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# Завантаження даних з JSONL файлу (передбачається, що файл вже завантажений в середу Kaggle)
def load_data_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data

# Завантаження даних з CSV файлу
def load_data_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._prepare_examples()

    def _prepare_examples(self):
        examples = []
        for item in self.data:
            text = clean_text(item['text'])
            encoded_text = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Перевіряємо, що тензори не порожні
            if encoded_text['input_ids'].size(1) > 0 and encoded_text['attention_mask'].size(1) > 0:
                examples.append({
                    'input_ids': encoded_text['input_ids'].squeeze(0),
                    'attention_mask': encoded_text['attention_mask'].squeeze(0)
                })
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Завантаження даних з CSV файлів
train_csv_file_path = '/kaggle/input/dataset/train.csv'  # Шлях до вашого файлу CSV на Kaggle
test_csv_file_path = '/kaggle/input/dataset/test.csv'  # Шлях до вашого файлу CSV на Kaggle
jsonl_file_path = '/kaggle/input/dataset/messages.jsonl'  # Шлях до вашого файлу JSONL на Kaggle

# Завантаження даних з CSV файлів
train_df = load_data_from_csv(train_csv_file_path)
test_df = load_data_from_csv(test_csv_file_path)

# Завантаження даних з JSONL файлу
jsonl_data = load_data_from_jsonl(jsonl_file_path)

# Підготовка даних для навчання
cleaned_data = []
labels = []

# Відповідність source_id категорії з train.csv
source_id_to_category = dict(zip(train_df['source_id'], train_df['source_category']))

for item in jsonl_data:
    if 'text' in item and 'source_id' in item:
        cleaned_text = clean_text(item['text'])
        if item['source_id'] in source_id_to_category:
            cleaned_data.append({'text': cleaned_text})
            labels.append(source_id_to_category[item['source_id']])

# Створення Dataset і DataLoader для навчального набору
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Додавання спеціального токена для паддингу

train_dataset = TextDataset(cleaned_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Приклад використання моделі GPT для отримання ембеддингів текстів
model = GPT2Model.from_pretrained('gpt2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.eval()

for batch in train_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Обробка ембеддингів тут (наприклад, усереднення)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Розділення даних на навчальну і тестову вибірки
    X_train, X_test, y_train, y_test = train_test_split(embeddings.cpu().numpy(), labels, test_size=0.2, random_state=42)

    # Навчання моделі класифікації (наприклад, логістична регресія)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Прогнозування на тестовій вибірці та оцінка точності
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy}")

    # Підготовка даних для тестування
    test_data = []
    test_ids = []

    for _, row in test_df.iterrows():
        test_ids.append(row['source_id'])
        # Очищаємо і токенізуємо текст (для прикладу використовуємо тільки поле 'source_url')
        cleaned_text = clean_text(row['source_url'])
        tokens = tokenize_text(cleaned_text)
        if tokens:  # Додаємо тільки непорожні токени
            test_data.append({'text': ' '.join(tokens)})  # Зберігаємо текст як рядок для токенайзера GPT

    # Перетворення текстів у токени для тестування
    encoded_test_texts = tokenizer.batch_encode_plus(
        [item['text'] for item in test_data],
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    # Отримання ембеддингів текстів для тестового набору
    with torch.no_grad():
        test_outputs = model(input_ids=encoded_test_texts['input_ids'].to(device), 
                             attention_mask=encoded_test_texts['attention_mask'].to(device))

    # Отримання ембеддингів текстів
    test_embeddings = test_outputs.last_hidden_state.mean(dim=1)  # Приклад: усереднення ембеддингів

    # Прогнозування на тестовому наборі
    test_predictions = classifier.predict(test_embeddings.cpu().numpy())

    # Вивід результатів для тестового набору
    results_df = pd.DataFrame({'source_id': test_ids, 'predicted_category': test_predictions})
    print(results_df.head())
    break  # Перериваємо цикл для прикладу, щоб не робити зайвих ітерацій
