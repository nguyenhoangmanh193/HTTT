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

        if 2 in labels_found:
            return 2, matched_words
        elif 1 in labels_found:
            return 1, matched_words
        elif 0 in labels_found:
            return 0, matched_words
        else:
            return 0, []

# Từ điển cảm xúc
lexicon = {
    "lạc_đà": [1], "ricon": [1], "bẩn":[1], "bẩn_tính":[1], "dơ_bẩn":[2], "bố":[2],"mày":[1], "đần":[2], "phú_ngao":[2],'phấn_đào':[2], "đi_tiểu":[1], "ăn_bẩn":[1], "phấn":[1],'kền_đần':[2],'varrid':[1],
    'hôi_pen':[1], 'kendan':[2],'ia':[1], 'city_đần':[2], 'citi_đần':[2], 'đỵt':[2], 'mõm':[1],'điên':[1], 'dơ_bẩn':[2],'dái':[2], 'lolll':[2],'loll':[2],'lol':[2], 'sủa':[2], 'ẳng':[2],
    'nhục':[1], 'chó':[2], 'thecity_đần':[2], "man_đần":[2],'mân':[2],'mân_đàn':[2], "tự_nhục":[1], 'chồn':[2],'chồn_xanh':[2], 'ăn_vạ':[1], 'quân_đần':[2], 'ngáo':[1], 'ung_thư':[1],
    'sa_tị':[1],'râu_con':[1],'sôn_lì':[2],'râu':[1],'mesex':[2],'mesexx':[2],'messex':[2],'lùn':[1],'ăn_lollll':[2],'ăn_lol':[2], 'cak':[2],'tâm_thần':[2],'bệnh':[1], 'sôn':[2],'chị':[1],
    'riconcak':[2],'7_tạ':[1],'chọ':[1],'tạ':[1],'cắn':[2],'rách':[2],'rẻ_rách':[2],'richa':[1],'raucha':[1],'râu_cha':[1],'mất_dạy':[2],'râucon':[1],'chửi':[2],'súc_vật':[2],
    'cay':[1], 'râuconcak':[2],'cỏ':[1], 'khỉ':[2], 'lz':[2], 'thẩm_du_tinh_thần':[1],'thẩm_du':[1], 'vardrid':[1], 'varid':[1], 'mân_đần':[2], 'vl':[1], 'mờ_ngu':[1],'penaldo':[1],'pessi':[1],
    'câm':[1], 'địt':[2],'sati':[1],'bẩn_tính':[1],'mu_đần':[2],'bẩn_thỉu':[2],'tổn_luồi':[2],'côn_đồ':[2],'mù':[2],'bóp_cổ':[1], 'cayyy':[1],'cayy':[1],'ngu':[2],'nguu':[2],'nguuu':[2],
    'si_tạ':[1], 'khóc':[1], 'nhục_nhã':[1],'bướm':[2],'culi':[2],'đàn_bà':[1], 'si_lùn':[1],'riconcac ':[2],'rauconcac':[2], 'cc':[2],'rô_đĩ':[2],'đĩ':[2],'đỹ':[2], 'ngáo':[1],
    'vạ':[1], 'rauconcak':[2],'raucon':[1]
}

# Tạo đối tượng model sẵn để import
model_football = SentimentModel(lexicon)