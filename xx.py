import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Giả sử đây là dữ liệu của bạn
df = pd.DataFrame({
    'video_title': ['Video 1', 'Video 2', 'Video 3', 'Video 4', 'Video 5'],
    'label': [0, 1, 1, 0, 0]  # 0 là không tiêu cực, 1 là tiêu cực
})

# Tính tổng số bình luận và số bình luận có label != 0 theo từng video
total_by_video = df.groupby('video_title').size()
toxic_by_video = df[df['label'] != 0].groupby('video_title').size()

# Tính tỷ lệ % và xác định video nào có tỷ lệ toxic > 11%
toxic_percent = (toxic_by_video / total_by_video * 100).fillna(0)

# Đánh dấu video tiêu cực (tỷ lệ toxic > 11%)
toxic_videos = toxic_percent[toxic_percent > 11].index
non_toxic_videos = toxic_percent[toxic_percent <= 11].index

# Tạo một Series với các video tiêu cực và không tiêu cực
video_labels = ['Tiêu cực' if video in toxic_videos else 'Không tiêu cực' for video in df['video_title']]

# Tính số lượng video tiêu cực và không tiêu cực
labels_count = pd.Series(video_labels).value_counts()

# Hàm vẽ biểu đồ tròn
def draw_pie_chart(labels_count):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#66b3ff'])
    ax.set_title('Tỷ lệ video tiêu cực vs không tiêu cực')
    return fig  # Trả về biểu đồ để có thể gọi lại sau

# Hàm vẽ biểu đồ khác
def draw_another_pie_chart():
    toxic_percentage = (labels_count['Tiêu cực'] / labels_count.sum()) * 100
    non_toxic_percentage = 100 - toxic_percentage
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie([toxic_percentage, non_toxic_percentage], labels=['Tiêu cực', 'Không tiêu cực'], autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#66b3ff'])
    ax.set_title('Tỷ lệ tiêu cực và không tiêu cực')
    return fig

# Streamlit: Tiêu đề
st.title('Ứng dụng Streamlit: Biểu đồ video tiêu cực vs không tiêu cực')

# Streamlit: Radio button để lựa chọn biểu đồ
chart_selection = st.radio(
    "Chọn loại biểu đồ để hiển thị:",
    ("Tỷ lệ video tiêu cực vs không tiêu cực", "Tỷ lệ tiêu cực và không tiêu cực")
)

# Hiển thị biểu đồ dựa trên lựa chọn
if chart_selection == "Tỷ lệ video tiêu cực vs không tiêu cực":
    pie_chart = draw_pie_chart(labels_count)
    st.pyplot(pie_chart)

elif chart_selection == "Tỷ lệ tiêu cực và không tiêu cực":
    another_pie_chart = draw_another_pie_chart()
    st.pyplot(another_pie_chart)
