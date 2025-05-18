import pandas as pd
# Khai báo 3 mảng sentiment đã được phân loại
positive = [
    'Positive','Excitement', 'Happiness', 'Joy', 'Love', 'Amusement', 'Enjoyment',
    'Admiration', 'Affection', 'Awe', 'Acceptance', 'Adoration',
    'Anticipation', 'Kind', 'Pride', 'Elation', 'Euphoria',
    'Contentment', 'Serenity', 'Gratitude', 'Hope', 'Empowerment',
    'Compassion', 'Tenderness', 'Arousal', 'Enthusiasm', 'Fulfillment',
    'Reverence', 'Determination', 'Zest', 'Hopeful', 'Proud', 'Grateful',
    'Empathetic', 'Compassionate', 'Playful', 'Free-spirited', 'Inspired',
    'Confident', 'Thrill', 'Overjoyed', 'Inspiration', 'Motivation',
    'Satisfaction', 'Blessed', 'Accomplishment', 'Wonderment', 'Optimism',
    'Enchantment', 'Intrigue', 'PlayfulJoy', 'Mindfulness', 'DreamChaser',
    'Elegance', 'Whimsy', 'Harmony', 'Creativity', 'Radiance', 'Wonder',
    'Rejuvenation', 'Coziness', 'Adventure', 'Melodic', 'FestiveJoy',
    'InnerJourney', 'Freedom', 'Dazzle', 'Adrenaline', 'ArtisticBurst',
    'CulinaryOdyssey', 'Resilience', 'Immersion', 'Spark', 'Marvel',
    'Success', 'Friendship', 'Romance', 'Tranquility', 'Grandeur',
    'Energy', 'Celebration', 'Charm', 'Ecstasy', 'Colorful', 'Hypnotic',
    'Connection', 'Iconic', 'Journey', 'Engagement', 'Touched',
    'Triumph', 'Heartwarming', 'Breakthrough', 'Joy in Baking',
    'Envisioning History', 'Imagination', 'Vibrancy', 'Mesmerizing',
    'Culinary Adventure', 'Winter Magic', 'Thrilling Journey',
    "Nature's Beauty", 'Celestial Wonder', 'Creative Inspiration',
    'Runway Creativity', "Ocean's Freedom", 'Happy', 'Confidence',
    'Kindness', 'Positivity', 'Amazement', 'Captivation', 'Emotion'
]

negative = [
    'Negative','Sad','Frustrated', 'Anger', 'Fear', 'Sadness', 'Disgust', 'Disappointed',
    'Bitter', 'Shame', 'Despair', 'Grief', 'Loneliness', 'Jealousy',
    'Resentment', 'Frustration', 'Boredom', 'Anxiety', 'Intimidation',
    'Helplessness', 'Envy', 'Regret', 'Melancholy', 'Bitterness',
    'Yearning', 'Fearful', 'Apprehensive', 'Overwhelmed', 'Jealous',
    'Devastated', 'Envious', 'Dismissive', 'Heartbreak', 'Betrayal',
    'Suffering', 'EmotionalStorm', 'Isolation', 'Disappointment',
    'LostLove', 'Exhaustion', 'Sorrow', 'Darkness', 'Desperation',
    'Ruins', 'Desolation', 'Loss', 'Heartache', 'Hate', 'Bad',
    'Embarrassed', 'Pressure', 'Miscalculation', 'Obstacle', 'Challenge'
]

neutral = [
    'Neutral','Bittersweet', 'Surprise', 'Calmness', 'Confusion', 'Curiosity',
    'Numbness', 'Nostalgia', 'Ambivalence', 'Pensive', 'Reflection',
    'Indifference', 'Contemplation', 'JoyfulReunion', 'Appreciation',
    'Sympathy', 'Renewed Effort', 'Solace', 'Relief', 'Mischievous',
    'Whispers of the Past', 'Solitude','Exploration','Suspense'
]

# Hàm để ánh xạ một sentiment cụ thể sang nhóm của nó
def map_sentiment_to_group(sentiment):
    sentiment = sentiment.strip()  # Loại bỏ khoảng trắng thừa
    if sentiment in positive:
        return 'Positive'
    elif sentiment in negative:
        return 'Negative'
    elif sentiment in neutral:
        return 'Neutral'
    else:
        return 'unknown'  # Trường hợp không tìm thấy


# Đọc file CSV đầu vào
def convert_sentiment_groups(input_file, output_file):
    try:
        # Đọc file CSV vào DataFrame
        df = pd.read_csv(input_file)

        # Kiểm tra xem cột 'Sentiment' có tồn tại không
        if 'Sentiment' not in df.columns:
            print("Lỗi: Không tìm thấy cột 'Sentiment' trong file CSV")
            return False

        # Tạo một bản sao của DataFrame để tránh thay đổi dữ liệu gốc
        new_df = df.copy()

        # Áp dụng hàm map_sentiment_to_group cho mỗi giá trị trong cột 'Sentiment'
        new_df['Sentiment'] = new_df['Sentiment'].apply(map_sentiment_to_group)

        # Lưu DataFrame mới vào file CSV đầu ra
        new_df.to_csv(output_file, index=False)
        print(f"Đã chuyển đổi thành công và lưu kết quả vào {output_file}")
        return True

    except Exception as e:
        print(f"Đã xảy ra lỗi: {str(e)}")
        return False

input_file = '../data/external/sentimentdataset.csv'
output_file = '../data/processed/sentimentgroups.csv'

# Gọi hàm để chuyển đổi
convert_sentiment_groups(input_file, output_file)