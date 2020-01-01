# real_or_not

This project builds an ML model to classify whether or not a tweet is about a real disaster.

I haven't done any NLP projects for a while, so I'll work on the Kaggle competition https://www.kaggle.com/c/nlp-getting-started/overview to refresh my knowledge.

# Data

Data is split into train, test and sample_submission, all .csv files.
The train and test contain columns:
	id: a unique identifier for each tweet
	text: the text of the tweet
	location: the location the tweet was sent from (may be blank)
	keyword: a particular keyword from the tweet (may be blank)
	target (only in train): Boolean, describing if tweet is about a real disaster or not

## Requirements

The code is run in Python 3.8, please find the package requirements in requirements.txt

## License
[MIT](https://choosealicense.com/licenses/mit/)
