# datingSelectionClassifier

Dating is hard, especially if one is bad at first impressions. What's even harder, is that according to [Fishman et al. 2006](http://www.stat.columbia.edu/~gelman/stuff_for_blog/sheena.pdf), men and women look for different things. Do these differences also change across other factors?

This project uses Fishman et al. 2006's data set which can be found on [Kaggle](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) to create a plotly dash app for users to explore how different features affect the probability of match, the event when both people choose each other. 

However, what model should be used? Do we optimize on accuracy, recall, precision? This app uses a customizable Ensemble Vote Classifier. Users can pick from a Logrithmic Regressor model, 2 different K-Nearest Neighbor models, 2 Gradient Boosting models, 2 Decision Tree models, and 2 Random Forest models.

## data (not included. Using .gitignore)

Due to data size and standard practice, my dataset is not included in this repo. The data folder is sectioned as follows:

The top level (data) is reserved for data that should not be deleted including, the original data set and a copy of the original data set with coordinates. Running eda.ipynb will delete and remake both data/processedData and data/plotlyDashData.

data/processedData is resevered for data needed for util.py and machineLearning.ipynb to do its processes.

data/plotlyDashData is reserved for data that app.py needs. app.py is only looking for data in this folder

## notebooks

eda.ipynb is my initial eda of the data and a ground for testing python funcationality. It deposits data into data folder. This needs to be ran before running machineLearning.ipynb. Running eda.ipynb will delete and remake both data/processedData and data/plotlyDashData

machineLearning.ipynb is my pipeline used to add dummies to the original dataset, train-test-split the dataset, train and test the classification models, deposit data into data, data/plotlyDashData, and data/processedData, and triggers collectionPipeLine.py to prepare data for app.py. If datingTrain.csv, datingTest.csv, or datingFull.csv is missing, it will delete and remake both data/processedData and data/plotlyDashData. 

## src

util.py is a custom python module that notebooks use. Some methods require certain files in data/processedData and data. If certain files in data/processedData is missing, please run machineLearning.ipynb

collectionPipeLine.py is the pipeline that prepares data from data/plotlyDashData prior to running app.py. 

## app.py

This is the Plotly Dash App that displays the data analysis to the user.

## descriptionDictionary.json and dummyValueDictionary.json

These are .jsons that are manually made. These are outside of the typical gitignored data folder so users who fork the repo do not have to manually make them.

## License

This work uses a MIT License, granting people to use or reuse this project for their own purposes.