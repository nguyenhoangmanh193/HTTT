import unicodedata
from underthesea import text_normalize
from underthesea import word_tokenize
import re
import string
import pandas as pd

stop_words = ['bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến_nỗi', 'đều', 'điều', 'do', 'đó', 'được', 'dưới', 'gì', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một_cách', 'này', 'nên', 'nếu', 'ngay', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'phải', 'qua', 'ra', 'rằng', 'rằng', 'rất', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'vì', 'việc', 'với', 'vừa']


def standardize_unicode(sentence):
    sentence = unicodedata.normalize('NFC', sentence)
    return sentence

def normalize_bar(sentence):
    sentence = text_normalize(sentence)
    return sentence

def split_words(sentence):
    sentence = word_tokenize(sentence, format='text')
    return sentence

def to_lower(sentence):
    sentence = sentence.lower()
    return sentence

def remove_html(sentence):
    return re.sub(r'<[^>]*>', '', str(sentence))

def remove_tags(sentence):
    return re.sub(r'@\w*', '', sentence).strip()

def remove_emoji(sentence):
    # Biểu thức chính quy để loại bỏ emoji
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002700-\U000027BF"  # Dingbats (bao gồm nhiều ký tự như ❤, ☝)
        u"\U000024C2-\U0001F251"  # Enclosed characters
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-B
        "]+", flags=re.UNICODE
    )

    # Loại bỏ emoji
    sentence = emoji_pattern.sub(r'', sentence)

    # Loại bỏ các ký tự khoảng trắng không bình thường
    sentence = re.sub(r'[\u200B\uFE0F]', '', sentence)

    return sentence

def remove_punctuation(sentence):
    result = sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return ' '.join(result.split())

def remove_stopwords(sentence):
    filtered_word = [w for w in sentence.split() if not w in stop_words]
    return " ".join(filtered_word)

def clean_up_pipeline(sentence):
    cleaning_utils = [
        remove_html,
        remove_tags,
        remove_emoji,
        remove_punctuation,
        standardize_unicode,
        normalize_bar,
        to_lower,
        split_words,
        remove_stopwords
    ]
    for o in cleaning_utils:
        sentence = o(sentence)
    return sentence


# file = "D://Downloads//all_video.csv"
# file2 = "D://Downloads//vide.csv"
# df = pd.read_csv(file,encoding='utf-8')
# df2 = pd.read_csv(file2,encoding='utf-8')
# df['clean_comment'] = df['comment'].apply(clean_up_pipeline)
#
# df = df.merge(
#     df2[['id', 'views', 'comments']],
#     left_on='video_id',
#     right_on='id',
#     how='left'
# )
# df.drop(columns=['id'], inplace=True)
# print(df.columns)
# print(df)