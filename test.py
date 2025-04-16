import pickle
from model.model_football import model
# Load model


# Gọi hàm classify_sentiment từ model


comment = "tôi cảm penaldo raucon"
prediction, words = model.classify_sentiment(comment)

print(f"Prediction: {prediction} (0: vui vẻ, 1: không toxic, 2: toxic)")
print(f"Từ góp phần vào dự đoán: {words}")