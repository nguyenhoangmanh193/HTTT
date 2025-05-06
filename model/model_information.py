# sentiment_model.py

class SentimentModel:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def classify_sentiment(self, comment):
        words = comment.split()
        matched_words = []
        labels_found = set()

        for word in words:
            if word in self.lexicon:
                matched_words.append(word)
                labels_found.update(self.lexicon[word])

        if 1 in labels_found:
            return 'toxic', matched_words
        elif 0 in labels_found:
            return 'bình thường', matched_words
        else:
            return 'bình thường', []

# Từ điển cảm xúc
lexicon = {
    "tướng_cướp": [1], 'trơ_trẽn':[1], 'vc':[1],'nhục':[1],'tham_ô':[1],'tham_nhũng':[1],
    'vô_nhân_đạo':[1], 'lạm_quyền':[1], 'tử_hình':[1],'sâu_bo':[1],'sâu_bọ':[1], 'bẩn':[1],
    'xử_tử ':[1], 'chửi':[1],'chết':[1], 'bẩn_thỉu':[1],'tù':[1], 'vnch':[1], 'móc_ruột':[1],
    'sủa':[1], 'giết':[1],'cướp':[1], 'khủng_bố':[1],'mõm':[1],'ác_quỷ':[1],
    'tâm_thần':[1], 'đâm_chém':[1],'dối_trá':[1],'độc_tài':[1],'ỉa':[1],'địa_ngục':[1],
      'thối nát':[1], 'cc':[1],'hấp_diêm':[1],'dối trá':[1],'giết_hại':[1],'ngu':[1]
}

# Tạo đối tượng model sẵn để import
model_info = SentimentModel(lexicon)