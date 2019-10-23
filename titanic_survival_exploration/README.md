# Project 0: Introduction and Fundamentals
## Titanic Survival Exploration
titanic_survival_exploration is my solution to Project 0 of the Udacity Machine Learning Engineer Nanodegree. The goal of the project was to build a decision tree algorithm that could predict whether a passenger survived the titanic. I explored three features ('sex', 'income', 'age') and created a decision tree based on criteria from the three features.

## Project Overview
In this project, you will create decision functions that attempt to predict survival outcomes from the 1912 Titanic disaster 
based on each passenger’s features, such as sex and age. You will start with a simple algorithm and increase its complexity 
until you are able to accurately predict the outcomes for at least 80% of the passengers in the provided data. This project 
will introduce you to some of the concepts of machine learning as you start the Nanodegree program.

In addition, you'll make sure Python is installed with the necessary packages to complete this project. 
There are two Python libraries, `numpy` and `pandas`, that we'll use a bit here in this project. 
Don't worry about how they work for now — we'll get to them in Project 1.
This project will also familiarize you with the submission process for the projects that you will 
be completing as part of the Nanodegree program.

## Starting the Project
You can download the .zip archive containing the necessary project files from Udacity's Machine Learning 
Github project repository [here](https://github.com/udacity/machine-learning). The project can be found 
under **projects** -> **titanic_survival_exploration**. This archive contains three files:

- `Titanic_Survival_Exploration.ipynb`: This is the main file where you will be performing your work on the project.
- `titanic_data.csv`: The project dataset. You’ll load this data in the notebook.
- `titanic_visualizations.py`: This Python script contains helper functions that will visualize the data and survival outcomes.

Once you’ve navigated to the folder containing the project files, you can use the command `ipython notebook Titanic_Survival_Exploration.ipynb` to open up a browser window or tab to work with your notebook. There are five questions in the notebook that you need to complete for the project.

### Install

This project requires **Python 2.7** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Udacity recommends our students install [Anaconda](https://www.continuum.io/downloads), i pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 

### Code

Template code is provided in the notebook `titanic_survival_exploration.ipynb` notebook file. Additional supporting code can be found in `titanic_visualizations.py`. While some code has already been implemented to get you started, you will need to implement additional functionality when requested to successfully complete the project.

### Run

In a terminal or command window, navigate to the top-level project directory `titanic_survival_exploration/` (that contains this README) and run one of the following commands:

```ipython notebook titanic_survival_exploration.ipynb```
```jupyter notebook titanic_survival_exploration.ipynb```

This will open the iPython Notebook software and project file in your browser.

## Data

The dataset used in this project is included as `titanic_data.csv`. This dataset is provided by Udacity and contains the following attributes:

- `survival` - Survival (0 = No; 1 = Yes)
- `pclass` - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
- `name` - Name
- `sex` - Sex
- `age` - Age
- `sibsp` - Number of Siblings/Spouses Aboard
- `parch` - Number of Parents/Children Aboard
- `ticket` - Ticket Number
- `fare` - Passenger Fare
- `cabin` - Cabin
- `embarked` - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
