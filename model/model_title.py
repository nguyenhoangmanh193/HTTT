# sentiment_model.py

class SentimentModel:
    def __init__(self, lexicon):
        self.lexicon = lexicon

    def classify_sentiment(self, comment):
        words = comment.split()
        matched_words = []
        label_counts = {}  # Đếm số từ cho mỗi nhãn

        for word in words:
            if word in self.lexicon:
                matched_words.append(word)
                for label in self.lexicon[word]:
                    label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            # Chọn nhãn có số lượng từ khớp nhiều nhất
            best_label = max(label_counts, key=label_counts.get)
            return best_label, matched_words
        else:
            return 2, []  # Mặc định nếu không khớp gì thì chọn nhãn 2 (thời sự)

# Từ điển cảm xúc
lexicon = {
   'bóng_đá':[0],'thể_thao':[0],'câu_lạc_bộ':[0],'football':[0],'sports':[0],'hội_cđv':[0],
    'ghi_bàn':[0],'ca_sĩ':[1],'âm_nhạc':[1],'âm_nhạc':[1],'pops':[1],'music':[1],'hát':[1],
    'ca nhạc':[1], 'remix':[1],'nghệ_sĩ':[1],'nghệ sỹ':[1],'hòa_nhạc':[1],'nhạc':[1],
    'thời_sự':[2],'xã_hội':[2],'kinh_tế':[2],'chính_trị':[2],'thông_tấn':[2],'thông_tin':[2],
    'tin_tức':[2], 'đảng':[2],'truyền_thông':[2],'báo_chí':[2],'thương_mại':[2],'báo':[2],
    'giáo_dục':[2],'quân_sự':[2]

}

# Tạo đối tượng model sẵn để import
model_title = SentimentModel(lexicon)