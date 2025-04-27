import pickle
import pandas as pd
from model.model_football import model
# Load model


# # Gọi hàm classify_sentiment từ model
#
#
# comment = "tôi cảm penaldo raucon"
# prediction, words = model.classify_sentiment(comment)
#
# print(f"Prediction: {prediction} (0: vui vẻ, 1: không toxic, 2: toxic)")
# print(f"Từ góp phần vào dự đoán: {words}")
file = "D://Downloads//A_data_Football.xlsx"
df = pd.read_excel(file, engine='openpyxl')
df = df.dropna(subset=['clean_comment', 'Label'])
df = df.reset_index(drop=True)
df['clean_comment'] = df['clean_comment'].fillna('').astype(str)
df['Label'] = df['Label'].fillna('').astype(int)
df['Labels'] = df['clean_comment'].apply(lambda x: model.classify_sentiment(x)[0])

# Tạo một cột mới để đánh dấu sự khác nhau giữa hai cột
df['Label_diff'] = df['Label'] != df['Labels']

# Hiển thị các hàng mà hai cột khác nhau
different_rows = df[df['Label_diff'] == True]

print(f"Số lượng hàng khác nhau: {len(different_rows)}")
print(different_rows[['Label', 'Labels']])
df.to_excel('ten_file.xlsx', index=False, engine='openpyxl')