import pickle
import pandas as pd
from model.model_football import model_football
from model.model_information import model_info
# Load model
def predict_sentiment(row):
    comment = str(row)  # Đảm bảo kiểu chuỗi
    label, matched = model_info.classify_sentiment(comment)
    return pd.Series([label])

# # Gọi hàm classify_sentiment từ model
#
#
comment = "tôi cảm penaldo raucon vnch"
prediction, words = model_info.classify_sentiment(comment)

print(f"Prediction: {prediction} (0: vui vẻ, 1: không toxic, 2: toxic)")
print(f"Từ góp phần vào dự đoán: {words}")
file = "D://Downloads//voaaa.xlsx"
df = pd.read_excel(file, sheet_name="Bình luận")
df.drop(columns=['Label'], inplace=True)
df[['sentiment']] = df['clean_comment'].apply(predict_sentiment)

count = df['sentiment'].sum()
print(f"Số dòng có label = 1 là: {count}")
