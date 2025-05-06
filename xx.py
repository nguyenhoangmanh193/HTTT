from model.model_title import model_title

comment = "bọn chúng thật sự tham_nhũng và ngu"
label, matched = model_title.classify_sentiment(comment)
print("Label:", label)
print("Matched words:", matched)