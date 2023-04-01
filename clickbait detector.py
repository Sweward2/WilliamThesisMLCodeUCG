import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix
# Load the data from the csv file
df = pd.read_csv(r"D:\clickbait101\smallCBdataset.csv")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["title"], df["clickbait"], test_size=0.2, random_state=42)

# Transform the text into numerical data using a CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print("")
print("\033[36mConfusion Matrix:")
print(conf_mat)
print("\033[0m")

def get_sentiment_analysis(headline):
    # Initialize the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Get the sentiment scores for the headline
    scores = sid.polarity_scores(headline)

    # Return the compound score
    return scores['compound']

def color_text(sentiment):
    if sentiment > 0:
        return "\033[32m{}\033[0m".format(sentiment)
    elif sentiment < 0:
        return "\033[31m{}\033[0m".format(sentiment)
    else:
        return "\033[33m{}\033[0m".format(sentiment)

def Url_Input(url):
    # Scrape the headlines and advertisements from the page
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    headlines = [(h.text, i+1) for i, h in enumerate(soup.find_all('h1'))]
    headlines3 = [(headlines3.text, i+1+len(headlines)) for i, headlines3 in enumerate(soup.find_all('h3'))]

    # Predict whether the headlines and advertisements on the page are clickbait or not
    items = headlines + headlines3
    items = [item for item in items if len(item[0].strip()) > 1 and len(item[0].strip().split()) > 5]
    predictions = model.predict(vectorizer.transform([item[0] for item in items]))

    # Add sentiment analysis to each headline
    for i, item in enumerate(items):
        items[i] = (item[0], item[1], get_sentiment_analysis(item[0]))

    # Return the list of tuples containing the headline text, its position on the page, and its sentiment analysis
    return list(zip(items, predictions))

def create_clickbait_graph(results):
    # Separate the clickbait and non-clickbait headlines and their positions on the page
    clickbait = [item[0][1] for item in results if item[1] == 1]
    not_clickbait = [item[0][1] for item in results if item[1] == 0]

    # Calculate the median position of the headlines
    median_pos = len(results) // 2
    median_clickbait = np.median(clickbait)
    median_not_clickbait = np.median(not_clickbait)

    # Calculate the percentage difference between the median values
    percentage_diff = abs((median_clickbait - median_not_clickbait) / ((median_clickbait + median_not_clickbait) / 2)) * 100

    # Calculate the mean compound score for clickbait and non-clickbait headlines separately
    mean_compound_clickbait = np.mean([item[0][2] for item in results if item[1] == 1])
    mean_compound_not_clickbait = np.mean([item[0][2] for item in results if item[1] == 0])

    # Create a scatter plot of the clickbait and non-clickbait headlines by position
    plt.scatter(['Yes']*len(clickbait), clickbait, color='red', alpha=0.5)
    plt.scatter(['No']*len(not_clickbait), not_clickbait, color='blue', alpha=0.5)

    # Add labels to the graph
    plt.xlabel('Clickbait')
    plt.ylabel('Page Position')
    plt.title('Where do clickbait and non clickbait headlines place on a given webpage?')

    # Add median values and percentage difference to the graph
    plt.plot(['Yes'], [median_clickbait], marker='_', color='red', markersize=5000, linewidth=1000)
    plt.plot(['No'], [median_not_clickbait], marker='_', color='blue', markersize=5000, linewidth=1000)
    plt.text('Yes', median_clickbait+2, f'      Median: {median_clickbait:.2f}', ha='left', va='top')
    plt.text('No', median_not_clickbait-2, f' Median: {median_not_clickbait:.2f}      ', ha='right', va='bottom')
    plt.text(0.5, 0.95, f'Median Percentage Difference: {percentage_diff:.2f}%', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.90, str(url), ha='center', va='center', transform=plt.gca().transAxes)


    # Reverse the y-axis
    plt.gca().invert_yaxis()

    # Show the graph
    plt.show()

# Input URL
url = 'https://www.thesun.co.uk'
results = Url_Input(url)

# Print the clickbait probability and sentiment analysis of each headline and advertisement on the page
for item, prediction in results:
    print("Title: {}".format(item[0]))
    print("Likely Clickbait: {}".format("Yes" if prediction else "No"))
    print("Sentiment Analysis: {}\n".format(color_text(item[2])))

# Count the number of clickbait and non-clickbait articles
num_clickbait = len([item for item, prediction in results if prediction == 1])
num_not_clickbait = len([item for item, prediction in results if prediction == 0])

# Calculate the percentage difference between the number of clickbait and non-clickbait articles
clickbait_ratio = num_clickbait / num_not_clickbait

# Print the number of clickbait and non-clickbait articles and the percentage difference
print("Number of Clickbait Articles: {}".format(num_clickbait))
print("Number of Non-Clickbait Articles: {}".format(num_not_clickbait))
print("Clickbait to Non-Clickbait Ratio: {:.2f}".format(clickbait_ratio))
print("Mean Compound Sentiment Score for Clickbait Headlines: {:.2f}".format(np.mean([item[0][2] for item in results if item[1] == 1])))
print("Mean Compound Sentiment Score for Non-Clickbait Headlines: {:.2f}".format(np.mean([item[0][2] for item in results if item[1] == 0])))

# Create a graph that shows the relationship between the position of a headline on a page and its likelihood of being clickbait
create_clickbait_graph(results)