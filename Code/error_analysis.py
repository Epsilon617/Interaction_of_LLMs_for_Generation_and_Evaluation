import pandas as pd
import math
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv", encoding='ISO-8859-1')

df['score'] = pd.to_numeric(df['score'], errors='coerce')
df['score'] = df['score'].fillna(6)
df['score'] = df['score'].apply(lambda x: math.ceil(x))
df['score'] = df['score'].astype(int)
results = {score: {'expert-v1': 0, 'layperson-v1': 0, 'nice': 0, 'rude': 0, 'neutral': 0, 'expert-v2': 0, 'layperson-v2': 0} for score in range(6)}

df['original'] = pd.to_numeric(df['original'], errors='coerce')
feedback_columns = ['expert-v1', 'layperson-v1', 'nice', 'rude', 'neutral', 'expert-v2', 'layperson-v2']
for col in feedback_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

for _, row in df.iterrows():
    original_score = row['original']
    score_value = row['score']
    if pd.notna(original_score) and score_value < 6:
        if score_value in [0,1,2,3,4,5]:
            if original_score > -0.696:
                for feedback_type in feedback_columns:
                    feedback_score = row[feedback_type]
                    if pd.notna(feedback_score) and feedback_score < original_score:
                        results[score_value][feedback_type] += 1

score_values = list(results.keys())
categories = list(feedback_columns)
counts = {category: [results[score][category] for score in score_values] for category in categories}
bar_width = 0.1
space_between_groups = 0.1

fig, ax = plt.subplots(figsize=(11, 8))

for idx, category in enumerate(categories):
    bar_positions = [x + (idx * bar_width) + (x * space_between_groups) for x in range(len(score_values))]
    ax.bar(bar_positions, counts[category], bar_width, label=category)

ax.set_xlabel('Score')
ax.set_ylabel('Number of Decreases')
ax.set_title('Number of Decreases in Scores by Feedback Type')

ax.set_xticks([x + (3.5 * bar_width) + (x * space_between_groups) for x in range(len(score_values))])
ax.set_xticklabels(score_values)
ax.legend()

plt.show()