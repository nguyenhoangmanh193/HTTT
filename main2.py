import streamlit as st
import pandas as pd
import requests
import re
import numpy as np
from io import BytesIO
from PIL import Image
from process_data import clean_up_pipeline
import xlsxwriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.dates as mdates
from model.model_title import model_title
from model.model_information import  model_info
from  model.model_football import model_football

#API_KEY = "AIzaSyANUWlnh43MDqZ3SS0DqCRiR8ns_5aP5DY"
API_KEY = "AIzaSyByJ01WdcsbNk53ifHvLiSrUCxAcjwuaZ4"

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/channels"
YOUTUBE_VIDEO_API_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_COMMENTS_API_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def get_channel_id(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    match = re.search(r'"externalId":"(UC[\w-]+)"', response.text)
    return match.group(1) if match else None


def get_recent_videos(channel_id):
    """Lấy 20 video gần nhất của kênh."""
    params = {
        "part": "snippet",
        "channelId": channel_id,
        "maxResults": 50,
        "order": "date",
        "type": "video",
        "key": API_KEY
    }
    response = requests.get(YOUTUBE_VIDEO_API_URL, params=params)
    data = response.json()

    if "items" not in data:
        return []

    videos = []
    video_ids = []


    for item in data["items"]:
        video_id = item["id"]["videoId"]
        video_ids.append(video_id)
        videos.append({
            "channel_name": item["snippet"]["channelTitle"],
            "title": item["snippet"]["title"],
            "published_date": item["snippet"]["publishedAt"],
            "id": video_id,
            "link": f"https://www.youtube.com/watch?v={video_id}"
        })

    # Gọi API để lấy số lượt xem và số bình luận
    stats_url = "https://www.googleapis.com/youtube/v3/videos"
    stats_params = {
        "part": "statistics",
        "id": ",".join(video_ids),
        "key": API_KEY
    }
    stats_response = requests.get(stats_url, params=stats_params)
    stats_data = stats_response.json()

    stats_dict = {item["id"]: item["statistics"] for item in stats_data.get("items", [])}

    for v in videos:
        vid = v["id"]
        v["views"] = stats_dict.get(vid, {}).get("viewCount", "N/A")
        v["comments"] = stats_dict.get(vid, {}).get("commentCount", "N/A")

    return videos


def get_all_comments(video_id, channel_id, video_title):
    """Lấy toàn bộ bình luận từ video, bao gồm cả phản hồi (replies)."""
    comments_list = []
    next_page_token = None

    while True:
        params = {
            "part": "snippet,replies",
            "videoId": video_id,
            "maxResults": 100,
            "textFormat": "plainText",
            "key": API_KEY
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(YOUTUBE_COMMENTS_API_URL, params=params)
        comments_data = response.json()

        if "items" in comments_data:
            for item in comments_data["items"]:
                # Top-level comment
                comment_snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments_list.append({
                    "channel_id": channel_id,
                    "video_id": video_id,
                    "video_title": video_title,
                    "author": comment_snippet["authorDisplayName"],
                    "comment": comment_snippet["textDisplay"],
                    "publishedAt": comment_snippet["publishedAt"],
                    "is_reply": False,
                    "reply_to": None
                })

                # Nếu có phản hồi (replies) thì duyệt thêm
                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_snippet = reply["snippet"]
                        comments_list.append({
                            "channel_id": channel_id,
                            "video_id": video_id,
                            "video_title": video_title,
                            "author": reply_snippet["authorDisplayName"],
                            "comment": reply_snippet["textDisplay"],
                            "publishedAt": reply_snippet["publishedAt"],
                            "is_reply": True,
                            "reply_to": comment_snippet["authorDisplayName"]  # ai được reply tới
                        })

        next_page_token = comments_data.get("nextPageToken")
        if not next_page_token:
            break

    return comments_list


# Cập nhật phần crawl để hiển thị thông tin bình luận
def crawl(url_channel):
    """Crawl thông tin kênh và danh sách video."""
    channel_id = get_channel_id(url_channel)
    if not channel_id:
        return None

    params = {"part": "snippet,statistics", "id": channel_id, "key": API_KEY}
    response = requests.get(YOUTUBE_API_URL, params=params)
    data = response.json()

    if "items" not in data or not data["items"]:
        return None

    channel_info = data["items"][0]
    snippet = channel_info["snippet"]
    stats = channel_info["statistics"]

    recent_videos = get_recent_videos(channel_id)

    return {
        "Created": snippet.get("publishedAt", "N/A")[:10],
        "Country": snippet.get("country", "N/A"),
        "Subscribers": int(stats.get("subscriberCount", 0)),
        "Total_videos": int(stats.get("videoCount", 0)),
        "Avatar": snippet["thumbnails"]["high"]["url"] if "thumbnails" in snippet else None,
        "Description": snippet.get("description", "Không có mô tả"),
        "List_id": channel_id,
        "Recent_videos": recent_videos
    }


def save(file_csv):
    df = pd.DataFrame([file_csv])  # Chuyển dictionary thành DataFrame
    csv_bytes = BytesIO()
    df.to_csv(csv_bytes, index=False, encoding="utf-8-sig")  # Lưu file với UTF-8 (hỗ trợ tiếng Việt)
    csv_bytes.seek(0)
    return csv_bytes


def profile_overview(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Column names:", df.columns.tolist())  # Print column names for debugging
    return {
        "Created": df['Created'].iloc[0] if 'Created' in df.columns else "N/A",
        "Add_to_ViralStat": df['Add_to_ViralStat'].iloc[0] if 'Add_to_ViralStat' in df.columns else "N/A",
        "Country": df['Country'].iloc[0] if 'Country' in df.columns else "N/A",
        "Subscribers": df['Subscribers'].iloc[0] if 'Subscribers' in df.columns else 0,
        "Total_videos": df['Total_videos'].iloc[0] if 'Total_videos' in df.columns else 0
    }


def profile_stats(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Column names:", df.columns.tolist())  # Print column names for debugging
    return {
        "Subscribers": df['Subscribers'].sum() if 'Subscribers' in df.columns else 0,
        "Total_view": df['Total_view'].sum() if 'Total_view' in df.columns else 0,
        "Avg": df['Avg'].mean() if 'Avg' in df.columns else 0
    }


def analyze_comments(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Column names:", df.columns.tolist())  # Print column names for debugging
    # Placeholder for comment analysis logic
    return {
        "Positive_comments": df['Positive_comments'].sum() if 'Positive_comments' in df.columns else 0,
        "Negative_comments": df['Negative_comments'].sum() if 'Negative_comments' in df.columns else 0,
        "Neutral_comments": df['Neutral_comments'].sum() if 'Neutral_comments' in df.columns else 0
    }


def recommend_videos(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.write("Column names:", df.columns.tolist())  # Print column names for debugging
    # Placeholder for recommendation logic
    return {
        "Recommended_videos": ["Video1", "Video2", "Video3"]  # Example recommendations
    }


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Chức năng")
    page = st.sidebar.radio("Chọn trang", ["Crawl", "Statistical", "Đề xuất"])

    if page == "Crawl":
        st.title("Crawl dữ liệu")
        url = st.text_input("Nhập URL kênh")

        if st.button("Tìm kiếm"):
            st.session_state["channel_data"] = crawl(url)
            if st.session_state["channel_data"] is None:
                st.error("Không tìm thấy dữ liệu!")
                st.stop()

        # Nếu đã có dữ liệu kênh
        if "channel_data" in st.session_state:
            data = st.session_state["channel_data"]

            col1, col2 = st.columns([1, 3])
            with col1:
                if data["Avatar"]:
                    response = requests.get(data["Avatar"])
                    image = Image.open(BytesIO(response.content))
                    st.image(image, width=100)
            with col2:
                st.write(f"**Created:** {data['Created']}")
                st.write(f"**Country:** {data['Country']}")
                st.write(f"**Subscribers:** {data['Subscribers']}")
                st.write(f"**Total Videos:** {data['Total_videos']}")
                st.write(f"**Description:** {data['Description']}")
            st.write("**Danh sách 20 video gần nhất**")
            # 🟢 Lưu danh sách video vào session_state để tránh reload mất dữ liệu
            if "Recent_videos" not in st.session_state:
                st.session_state["Recent_videos"] = data["Recent_videos"]

            df_videos = pd.DataFrame(st.session_state["Recent_videos"])
            # 🟢 Thêm phần tải về CSV

            # 🟢 Dropdown chọn video
            video_ids = [video["id"] for video in st.session_state["Recent_videos"]]
            selected_video_id = st.selectbox("Chọn video:", video_ids, format_func=lambda vid: next(
                v["title"] for v in st.session_state["Recent_videos"] if v["id"] == vid))

            # 🟢 Hiển thị thông tin video đã chọn
            #selected_video = next(v for v in st.session_state["Recent_videos"] if v["id"] == selected_video_id)
            def safe_video_generator():
                for v in st.session_state["Recent_videos"]:
                    try:
                        if v["id"] == selected_video_id:
                            yield v
                    except Exception:
                        continue  # Bỏ qua phần tử lỗi

            selected_video = next(safe_video_generator(), None)
            st.write(selected_video)
            try:
                st.write(f"**Tiêu đề video:** {selected_video['title']}")
                st.write(f"**Lượt xem:** {selected_video['views']}")
                st.write(f"**Số bình luận:** {selected_video['comments']}")
            except Exception as e:
                st.write("Không thể hiển thị thông tin video.")

            # 🟢 Lấy bình luận chỉ khi chưa có
            if "video_comments" not in st.session_state or st.session_state["video_comments"][
                "video_id"] != selected_video_id:
                st.session_state["video_comments"] = {"video_id": selected_video_id,
                                                      "comments": get_all_comments(selected_video_id, data['List_id'],
                                                                                   selected_video['title'])}

            # 🟢 Hiển thị bình luận
            df_comments = pd.DataFrame(st.session_state["video_comments"]["comments"])
            df_comments['clean_comment'] = df_comments['comment'].apply(clean_up_pipeline)
            st.dataframe(df_comments)

            # 🟢 Thêm phần tải về CSV

            # 🟢 Nút lấy toàn bộ comment của tất cả video
            if st.button("Lấy toàn bộ bình luận của 20 video"):
                all_comments = []
                for video in st.session_state["Recent_videos"]:
                    comments = get_all_comments(video["id"], data["List_id"], video["title"])
                    all_comments.extend(comments)

                # Lưu toàn bộ comment vào session
                st.session_state["all_video_comments"] = all_comments
                st.success("Đã lấy xong toàn bộ bình luận!")

            # 🟢 Hiển thị bảng toàn bộ bình luận nếu có
            if "all_video_comments" in st.session_state:
                df_all_comments = pd.DataFrame(st.session_state["all_video_comments"])

                if not df_all_comments.empty:
                    st.write("### Toàn bộ bình luận của tất cả các video")
                    st.dataframe(df_all_comments)

                    # Tải về CSV
            if st.button("📥 Tải file Excel tổng hợp (.xlsx)"):
                excel_bytes = BytesIO()
                with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
                    # Sheet 1 - Thông tin kênh
                    channel_info_df = pd.DataFrame([{
                        "Ngày tạo": data["Created"],
                        "Quốc gia": data["Country"],
                        "Lượt đăng ký": data["Subscribers"],
                        "Tổng số video": data["Total_videos"],
                        "Mô tả kênh": data["Description"]
                    }])
                    channel_info_df.to_excel(writer, sheet_name="Thông tin kênh", index=False)

                    # Sheet 2 - Danh sách video gần đây
                    df_videos.to_excel(writer, sheet_name="Video gần đây", index=False)

                    # Sheet 3 - Danh sách bình luận (ưu tiên lấy toàn bộ nếu có)
                    df_all_comments = pd.DataFrame(
                        st.session_state["all_video_comments"]
                    ) if "all_video_comments" in st.session_state else df_comments
                    df_all_comments['clean_comment'] = df_all_comments['comment'].apply(clean_up_pipeline)
                    df_all_comments.to_excel(writer, sheet_name="Bình luận", index=False)

                excel_bytes.seek(0)
                st.download_button(
                    label="📄 Tải xuống file Excel",
                    data=excel_bytes,
                    file_name="youtube_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    elif page == "Statistical":
        st.title("Thống kê Phân tích YouTube")
        uploaded_file = st.file_uploader("Tải lên file Excel (.xlsx)", type=["xlsx"])

        if uploaded_file:
            video_data = pd.read_excel(uploaded_file, sheet_name="Video gần đây")
            channel_info = pd.read_excel(uploaded_file, sheet_name="Thông tin kênh")
            video_comment = pd.read_excel(uploaded_file, sheet_name="Bình luận")

            video_comment['clean_comment'] = video_comment['clean_comment'].astype(str)

            #################
            cleaned = clean_up_pipeline(channel_info['Mô tả kênh'][0])
            label, words = model_title.classify_sentiment(cleaned)
            st.write(label)
            # st.write(words)
            if label == '0':
                video_comment['label'] = video_comment['clean_comment'].apply(
                    lambda x: model_football.classify_sentiment(x)[0])
            else:
                video_comment['label'] = video_comment['clean_comment'].apply(
                    lambda x: model_info.classify_sentiment(x)[0])





            ##################
            video_data['published_date'] = pd.to_datetime(video_data['published_date'])
            video_data['title_length'] = video_data['title'].apply(lambda x: len(str(x)))
            video_data['comment_view_ratio'] = video_data['comments'] / video_data['views']
            video_data['month'] = video_data['published_date'].dt.to_period('M')
            video_data['week'] = video_data['published_date'].dt.to_period('W')
            video_data['day'] = video_data['published_date'].dt.to_period('D')
            video_data['short_title'] = video_data['title'].apply(lambda x: (x[:40] + '...') if len(str(x)) > 40 else x)

            # Dòng tiêu đề với chữ trái - phải
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("### 📊 Tổng quan kênh")

            with c2:
                st.markdown("### 🔍 Theo dõi")

            tab1, tab2, tab3, tab4, tab5, tab6 , tab7, tab8, tab9= st.tabs(
                ["Tổng quan", "Phổ biến", "Tỷ lệ tương tác", "Hiệu suất", "Phân nhóm", "Tỉ lệ tổng quan",'Tỉ lệ theo thời gian', "Top video vi phạm",
                 "Đánh giá tổng thể"])

            with tab1:
                st.subheader("Thông tin kênh")
                st.write(f"**Ngày tạo:** {channel_info.iloc[0]['Ngày tạo']}")
                st.write(f"**Quốc gia:** {channel_info.iloc[0]['Quốc gia']}")
                st.write(f"**Lượt đăng ký:** {channel_info.iloc[0]['Lượt đăng ký']:,}")
                st.write(f"**Tổng số video:** {channel_info.iloc[0]['Tổng số video']:,}")
                st.write(f"**Mô tả:** {channel_info.iloc[0]['Mô tả kênh']}")

            with tab2:
                st.markdown("## 📊 So sánh Top 10 Videos (Mở rộng ngang, chữ nhỏ)")

                col1, col2 = st.columns([1, 1])  # giữ nguyên chia 2 cột bằng nhau

                with col1:
                    st.markdown("### 👁️ Views")
                    top10_views = video_data.sort_values(by='views', ascending=False).head(10)
                    fig1 = plt.figure(figsize=(7, 3))  # rộng hơn, thấp hơn
                    sns.barplot(data=top10_views, y='short_title', x='views', palette='Blues_r')
                    plt.title('Top 10 by Views', fontsize=10)
                    plt.xlabel('Views', fontsize=9)
                    plt.ylabel('')
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig1)

                with col2:
                    st.markdown("### 💬 Comments")
                    top10_comments = video_data.sort_values(by='comments', ascending=False).head(10)
                    fig2 = plt.figure(figsize=(7, 3))  # rộng hơn, thấp hơn
                    sns.barplot(data=top10_comments, y='short_title', x='comments', palette='Oranges_r')
                    plt.title('Top 10 by Comments', fontsize=10)
                    plt.xlabel('Comments', fontsize=9)
                    plt.ylabel('')
                    plt.xticks(fontsize=8)
                    plt.yticks(fontsize=7)
                    plt.tight_layout()
                    st.pyplot(fig2)

            with tab3:
                with st.expander("Theo bình luận và lượt xem"):
                    st.markdown("### Bubble chart: Views vs Comments (Size = Comments/Views)")

                    # Tạo layout với 2 cột
                    col1, col2 = st.columns([7, 3])  # Cột bên trái chiếm 50% và cột bên phải chiếm 50%

                    # --- Cột 1: Biểu đồ Bubble Chart ---
                    with col1:
                        plt.figure(figsize=(10, 6))  # Kích thước biểu đồ
                        sizes = ((video_data['comments'] / video_data['views']) * 300000).clip(10, 5000)
                        plt.scatter(video_data['views'], video_data['comments'], s=sizes, alpha=0.5, edgecolors='w')

                        for i in range(len(video_data)):
                            short_title = video_data['short_title'].iloc[i]
                            plt.annotate(short_title,
                                         (video_data['views'].iloc[i], video_data['comments'].iloc[i]),
                                         fontsize=8, alpha=0.6)

                        plt.title('Bubble Chart: Views vs Comments (Size = Comments/Views)', fontsize=12)
                        plt.xlabel('Views', fontsize=10)
                        plt.ylabel('Comments', fontsize=10)
                        plt.grid(True)
                        plt.tight_layout()
                        st.pyplot(plt)
                        # Tạo đường kẻ phân cách giữa hai cột
                    st.markdown(
                        """
                        <style>
                        .divider {
                            border-left: 2px solid #D3D3D3;
                            height: 100%;
                            margin-left: 10px;
                            margin-right: 10px;
                        }
                        </style>
                        """, unsafe_allow_html=True
                    )
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                    # --- Cột 2: Hiển thị chữ "Sl" ---
                    with col2:
                        st.markdown("### Sl")
                        st.write("Đây là phần dành cho chữ 'Sl'. Bạn có thể thay đổi nội dung theo yêu cầu.")

                with st.expander("Tỉ lệ tương tác theo độ dài tiêu đề"):
                    st.markdown("### Scatter: Title length vs Comments/Views")

                    # Tạo layout với 2 cột, biểu đồ chiếm 70%, chữ "Sl" chiếm 30%
                    col1, col2 = st.columns([7, 3])  # Cột bên trái chiếm 70% và cột bên phải chiếm 30%

                    # --- Cột 1: Biểu đồ Scatter ---
                    with col1:
                        video_data['comment_view_ratio'] = video_data['comments'] / video_data['views']

                        plt.figure(figsize=(10, 6))  # Kích thước biểu đồ
                        sns.scatterplot(data=video_data, x='title_length', y='comment_view_ratio', color='green',
                                        marker='o')

                        plt.title('Title Length vs Comment/View Ratio', fontsize=12)
                        plt.xlabel('Title Length', fontsize=10)
                        plt.ylabel('Comment/View Ratio', fontsize=10)
                        plt.grid(True)
                        plt.tight_layout()
                        st.pyplot(plt)

                    # --- Cột 2: Hiển thị chữ "Sl" ---
                    with col2:
                        # Thêm đường phân cách giữa cột 1 và cột 2
                        st.markdown(
                            """
                            <style>
                            .divider {
                                border-left: 2px solid #D3D3D3;
                                height: 100%;
                                margin-left: 10px;
                                margin-right: 10px;
                            }
                            </style>
                            """, unsafe_allow_html=True
                        )
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)  # Hiển thị đường phân cách

                        # Nội dung cột 2
                        st.markdown("### Sl")
                        st.write("Đây là phần dành cho chữ 'Sl'. Bạn có thể thay đổi nội dung theo yêu cầu.")

            with tab4:
                st.markdown("### 📊 Phân tích số video và lượt tương tác theo thời gian")
                # --- 6. Số video theo tháng, tuần, ngày ---

                # Monthly stats (có đầy đủ tháng)
                st.markdown("")
                full_month_range = pd.period_range(start=video_data['month'].min(), end=video_data['month'].max(),
                                                   freq='M')
                monthly_stats = video_data.groupby('month').agg({
                    'title': 'count',
                    'views': 'sum',
                    'comments': 'sum'
                }).rename(columns={'title': 'video_count'})
                monthly_stats = monthly_stats.reindex(full_month_range, fill_value=0)

                # Weekly stats (có đầy đủ tuần)
                full_week_range = pd.period_range(start=video_data['week'].min(), end=video_data['week'].max(),
                                                  freq='W')
                weekly_stats = video_data.groupby('week').agg({
                    'title': 'count',
                    'views': 'sum',
                    'comments': 'sum'
                }).rename(columns={'title': 'video_count'})
                weekly_stats = weekly_stats.reindex(full_week_range, fill_value=0)

                # Daily stats (có đầy đủ ngày)
                full_day_range = pd.period_range(start=video_data['day'].min(), end=video_data['day'].max(), freq='D')
                daily_stats = video_data.groupby('day').agg({
                    'title': 'count',
                    'views': 'sum',
                    'comments': 'sum'
                }).rename(columns={'title': 'video_count'})

                daily_stats = daily_stats.reindex(full_day_range, fill_value=0)
                # 🎯 Tính trung bình views/comments CHỈ CHO NHỮNG NGÀY CÓ VIDEO
                # (không dùng reindex)

                monthly_avg = monthly_stats[monthly_stats['video_count'] > 0]
                weekly_avg = weekly_stats[weekly_stats['video_count'] > 0]
                daily_avg = daily_stats[daily_stats['video_count'] > 0]

                monthly_avg['avg_views_per_video'] = monthly_avg['views'] / monthly_avg['video_count']
                weekly_avg['avg_views_per_video'] = weekly_avg['views'] / weekly_avg['video_count']
                daily_avg['avg_views_per_video'] = daily_avg['views'] / daily_avg['video_count']

                monthly_avg['avg_comments_per_video'] = monthly_avg['comments'] / monthly_avg['video_count']
                weekly_avg['avg_comments_per_video'] = weekly_avg['comments'] / weekly_avg['video_count']
                daily_avg['avg_comments_per_video'] = daily_avg['comments'] / daily_avg['video_count']

                # Số video theo ngày

                # --- Expander: Thống kê theo ngày ---
                with st.expander("📅 Thống kê theo ngày", expanded=False):
                    # 6.3 Daily - Số video theo ngày
                    st.markdown("Số video trên ngày")
                    plt.figure(figsize=(20, 6))
                    ax = daily_stats['video_count'].plot(kind='bar', color='lightcoral')
                    plt.title('Number of Videos per Day')
                    plt.tight_layout()
                    st.pyplot(plt)

                    # --- Daily ---
                    st.markdown("Số views trên ngày")
                    daily_stats_nonzero = daily_stats[daily_stats['video_count'] > 0].copy()
                    daily_stats_nonzero['avg_views_per_video'] = daily_stats_nonzero['views'] / daily_stats_nonzero[
                        'video_count']
                    fig, ax = plt.subplots(figsize=(20, 6))
                    ax.plot(daily_stats_nonzero.index.to_timestamp(), daily_stats_nonzero['avg_views_per_video'],
                            marker='o', color='green', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    plt.xticks(rotation=45)
                    plt.title('Average Views per Video per Day')
                    plt.xlabel('Day')
                    plt.ylabel('Avg Views per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Daily ---
                    st.markdown("Số comments trên ngày")
                    daily_stats_nonzero['avg_comments_per_video'] = daily_stats_nonzero['comments'] / \
                                                                    daily_stats_nonzero[
                                                                        'video_count']
                    fig, ax = plt.subplots(figsize=(20, 6))
                    ax.plot(daily_stats_nonzero.index.to_timestamp(), daily_stats_nonzero['avg_comments_per_video'],
                            marker='o', color='red', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    plt.xticks(rotation=45)
                    plt.title('Average Comments per Video per Day')
                    plt.xlabel('Day')
                    plt.ylabel('Avg Comments per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Expander: Thống kê theo tuần ---
                with st.expander("📈 Thống kê theo tuần", expanded=False):
                    # 6.2 Weekly - Số video theo tuần
                    st.markdown("Số video trên tuần")
                    plt.figure(figsize=(14, 6))
                    ax = weekly_stats['video_count'].plot(kind='bar', color='lightgreen')
                    plt.title('Number of Videos per Week')
                    plt.tight_layout()
                    st.pyplot(plt)

                    # --- Weekly ---
                    st.markdown("Số views trên tuần")
                    weekly_stats_nonzero = weekly_stats[weekly_stats['video_count'] > 0].copy()
                    weekly_stats_nonzero['avg_views_per_video'] = weekly_stats_nonzero['views'] / weekly_stats_nonzero[
                        'video_count']
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(weekly_stats_nonzero.index.to_timestamp(), weekly_stats_nonzero['avg_views_per_video'],
                            marker='o', color='blue', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    plt.xticks(rotation=45)
                    plt.title('Average Views per Video per Week')
                    plt.xlabel('Week')
                    plt.ylabel('Avg Views per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Weekly ---
                    st.markdown("Số comments trên tuần")
                    weekly_stats_nonzero['avg_comments_per_video'] = weekly_stats_nonzero['comments'] / \
                                                                     weekly_stats_nonzero['video_count']
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(weekly_stats_nonzero.index.to_timestamp(), weekly_stats_nonzero['avg_comments_per_video'],
                            marker='o', color='purple', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
                    plt.xticks(rotation=45)
                    plt.title('Average Comments per Video per Week')
                    plt.xlabel('Week')
                    plt.ylabel('Avg Comments per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    # --- Expander: Thống kê theo tháng ---
                with st.expander("🗓️ Thống kê theo tháng", expanded=False):
                    # 6.1 Monthly - Số video theo tháng
                    st.markdown("Số video trên tháng")
                    plt.figure(figsize=(10, 5))
                    ax = monthly_stats['video_count'].plot(kind='bar', color='skyblue')
                    plt.title('Number of Videos per Month')
                    plt.tight_layout()
                    st.pyplot(plt)

                    # --- Monthly ---
                    st.markdown("Số Views trên tháng")
                    monthly_stats_nonzero = monthly_stats[monthly_stats['video_count'] > 0].copy()
                    monthly_stats_nonzero['avg_views_per_video'] = monthly_stats_nonzero['views'] / \
                                                                   monthly_stats_nonzero[
                                                                       'video_count']
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(monthly_stats_nonzero.index.to_timestamp(), monthly_stats_nonzero['avg_views_per_video'],
                            marker='o', color='orange', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
                    plt.xticks(rotation=45)
                    plt.title('Average Views per Video per Month')
                    plt.xlabel('Month')
                    plt.ylabel('Avg Views per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # --- Monthly ---
                    st.markdown("Số comments trên tháng")
                    monthly_stats_nonzero['avg_comments_per_video'] = monthly_stats_nonzero['comments'] / \
                                                                      monthly_stats_nonzero['video_count']
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(monthly_stats_nonzero.index.to_timestamp(), monthly_stats_nonzero['avg_comments_per_video'],
                            marker='o', color='green', linestyle='-')
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
                    plt.xticks(rotation=45)
                    plt.title('Average Comments per Video per Month')
                    plt.xlabel('Month')
                    plt.ylabel('Avg Comments per Video')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)

            with tab5:
                st.markdown("KMeans Clustering: Views vs Comments")
                features = video_data[['views', 'comments']]
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)
                kmeans = KMeans(n_clusters=4, random_state=42)
                video_data['cluster'] = kmeans.fit_predict(scaled_features)

                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=video_data, x='views', y='comments', hue='cluster', palette='Set2')
                plt.title('KMeans Clustering')
                st.pyplot(plt)

                st.markdown("""
                **Nhóm phân loại:**
                - Nhóm 0: View cao, comment cao
                - Nhóm 1: View cao, comment thấp
                - Nhóm 2: View thấp, comment cao
                - Nhóm 3: View thấp, comment thấp
                """)

            with tab6:
                st.markdown("Bảng ")
                st.subheader("Dữ liệu bảng Bình luận")
                st.write(label)

                # Tính toán số lượng các giá trị trong cột 'label'
                label_counts = video_comment['label'].value_counts().sort_index()
                labels = ['Tích cực' if i == 'bình thường' else 'Tiêu cực' for i in label_counts.index]

                # Tạo layout 2 cột
                col1, col2 = st.columns(2)

                with col1:
                    # Biểu đồ tròn trong cột bên trái
                    fig, ax = plt.subplots(figsize=(2.5, 2.5))  # Điều chỉnh kích thước tùy ý
                    ax.pie(label_counts, labels=labels, autopct='%.1f%%',
                           colors=['#66b3ff', '#ff6666'], startangle=140, textprops={'fontsize': 5})
                    ax.set_title('Tỷ lệ cảm xúc', fontsize=7)
                    st.pyplot(fig)

                with col2:
                    # Phần chữ hoặc nội dung bên phải
                    st.markdown("### Phân tích cảm xúc")
                    st.write(
                        "Biểu đồ bên trái thể hiện tỷ lệ phần trăm các bình luận tích cực và tiêu cực. "
                        "Dựa vào dữ liệu, bạn có thể nhận biết sự phân bố cảm xúc trong tập bình luận."
                    )

            with tab7:

                # Đảm bảo cột 'publishedAt' là kiểu datetime
                video_comment['publishedAt'] = pd.to_datetime(video_comment['publishedAt'])

                # Tạo cột "3 ngày"
                video_comment['3_days'] = video_comment['publishedAt'].dt.to_period('D').apply(lambda r: r.start_time).dt.floor(
                    'D') + pd.to_timedelta(video_comment['publishedAt'].dt.day // 3 * 3, unit='D')

                # Đếm số lượng bình luận theo khoảng 3 ngày và nhãn
                count_by_3_days = video_comment.groupby(['3_days', 'label']).size().unstack(fill_value=0)

                col1, col2 = st.columns([3, 2])  # 3 phần cho biểu đồ, 2 phần cho nội dung khác (tổng = 5 -> 60%)

                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    count_by_3_days.plot(kind='bar', ax=ax, color=['#66b3ff', '#ff6666'])

                    ax.set_xlabel('Ngày (Mỗi 3 ngày)', fontsize=8)
                    ax.set_ylabel('Số lượng bình luận', fontsize=8)
                    ax.set_title('Số lượng bình luận theo mỗi 3 ngày và nhãn (0: Bình thường, 1: Toxic)', fontsize=9)
                    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
                    ax.tick_params(axis='y', labelsize=7)
                    ax.legend(title='Label', labels=count_by_3_days.columns.astype(str), fontsize=7, title_fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.markdown("### Thống kê")
                    st.markdown("- Biểu đồ thể hiện số lượng bình luận theo từng khoảng 3 ngày.")
                    st.markdown("- Màu xanh: Bình thường, Màu đỏ: Toxic.")

            with tab8:
                # Tính tổng số bình luận và số bình luận có label != 0 theo từng video
                total_by_video = video_comment.groupby('video_title').size()
                toxic_by_video = video_comment[video_comment['label'] != 'bình thường'].groupby('video_title').size()

                # Tính tỷ lệ % và chọn top 10 video có tỷ lệ toxic cao nhất
                toxic_percent = (toxic_by_video / total_by_video * 100).fillna(0)
                top10_videos = toxic_percent.sort_values(ascending=False).head(10)

                # Layout chia 2 cột: 60% - 40%
                col1, col2 = st.columns([3, 2])

                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(range(1, 11), top10_videos.values, color='#ff6666')
                    ax.set_xticks(range(1, 11))
                    ax.set_xticklabels(range(1, 11))
                    ax.set_ylabel('Tỷ lệ % bình luận toxic', fontsize=9)
                    ax.set_title('Top 10 video có tỷ lệ bình luận toxic cao nhất', fontsize=10)
                    ax.tick_params(labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.markdown("**Danh sách video tương ứng:**")
                    video_labels = top10_videos.index.tolist()
                    for i, title in enumerate(video_labels, start=1):
                        st.markdown(f"{i}. {title}")

            with tab9:

                # Tính tổng số bình luận và số bình luận có label != 0 theo từng video
                total_by_video = video_comment.groupby('video_title').size()
                toxic_by_video = video_comment[video_comment['label'] != 'bình thường'].groupby('video_title').size()

                # Tính tỷ lệ % và xác định video nào có tỷ lệ toxic > 11%
                toxic_percent = (toxic_by_video / total_by_video * 100).fillna(0)

                # Đánh dấu video tiêu cực (tỷ lệ toxic > 11%)
                toxic_videos = toxic_percent[toxic_percent > 11].index
                non_toxic_videos = toxic_percent[toxic_percent <= 11].index

                # Tạo một Series với các video tiêu cực và không tiêu cực
                video_labels = ['Tiêu cực' if video in toxic_videos else 'Không tiêu cực' for video in
                                video_comment['video_title']]
                labels_count = pd.Series(video_labels).value_counts()

                # Tạo layout 2 cột với tỷ lệ 3:2 (60% và 40%)
                col1, col2 = st.columns([1, 1])

                with col1:
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(labels_count, labels=labels_count.index, autopct='%1.1f%%', startangle=90,
                           colors=['#ff6666', '#66b3ff'])
                    ax.set_title('Tỷ lệ video tiêu cực vs không tiêu cực', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig)

                with col2:
                    st.markdown("**Phân loại video:**")
                    for label, count in labels_count.items():
                        st.markdown(f"- **{label}**: {count} video")



            plt.show()
            # --- Kết thúc ---
            print("\n✅ Đã hiển thị toàn bộ biểu đồ!")


    elif page == "Đề xuất":
        st.title("Đề xuất video")
        uploaded_file = st.file_uploader("Tải lên file CSV", type=["csv"])
        if uploaded_file:
            recommendation_data = recommend_videos(uploaded_file)

            st.markdown("**Video Recommendations**")
            for video in recommendation_data["Recommended_videos"]:
                st.text(video)


if __name__ == "__main__":
    main()