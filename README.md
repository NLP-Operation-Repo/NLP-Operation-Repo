<a name="top"></a>

# Project: Predicting MLB Team Wins + Analyzing Stats that Lead to Wins

by Martin Reyes

Note: project done in [this](https://github.com/prestige-worldwide-1/baseball-project#date-dictionary) repo

**Description:**

In this project, README's from GitHub repositories are analyzed and modelled using natural language processing techniques. The repositories contain primarily "Python", "Javascript", or "Other" files. Repo names are scraped from the most popular repositories on GitHub. The README contents are then acquired using GitHub's API. After the data is acquired, the README contents are cleaned and prepared so they can be analyzed. Analysis and stats test will be done to compare things such as length and unique words. NLP techniques and classification models will be used to predict which language is primarily used in the GitHub repos.

**Goals:**

- analyze GitHub repository README's and how they compare whether the repository primarily consists of Python, Javascript, or other programming languages.
- predict whether a GitHub Repository primarily consists of Python files, Javascript files, or other files.


## Acquire Data

1. GitHub repository names are scraped from GitHub's most [forked](https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories) and [starred](https://github.com/search?q=stars%3A%3E0&s=stars&type=Repositories) repositories. 
2. After the list of repo names are scraped, the names are run through GitHub's API to get the README contents along with the primary language of the repo.
3. The data is then saved into a [json file]('data2.json').

## Prepare Data
1. README's are cleaned:
    1. characters are lowered
    1. removed punctuation, numbers, extra whitespaces, long words, accented/special characters, and stopwords
    1. text is tokenized (broken down into smaller units)
    1. stemmed and lemmatized versions of the README's are saved
2. Languages are categorized into `Python` (1), `Javascript` (2), and `Other` (0)


## Data Exploration (EDA)

Data is split into training and test data. Analysis is performed on training data to avoid bias and data leakage during modeling. 

Test data is separated to test ML regression models later in the project.

**What are the most common words throughout all README's?**

<img src="viz/common_words.png" alt="common_words.png" width="600">


**Does the length of the README vary by programming language?**

<img src="viz/readme_lengths.png" alt="readme_lengths.png" width="600">


**Do different programming languages use a different number of unique words?**

<img src="viz/num_unique_words.png" alt="num_unique_words.png" width="600">


**Are there any words that uniquely identify a programming language?**

<img src="viz/unique_words_python.png" alt="unique_words_python.png" width="600">

<img src="viz/unique_words_javascript.png" alt="unique_words_javascript.png" width="600">



## Machine Learning Models: NLP and Classification

**Baseline Model**

- `sklearn`'s `DummyClassifier`, which makes a constant prediction of the most frequent target class, `Other`.
- Model accuracy is 74% on the validation set.

**Best Regression Model: Logistic Regression**

- Model predictions are off by about 6 wins, on average, and explain 77% of the variance.

| Model              | Accuracy   |
| :----------------- | ---------- |
| Baseline           |  74%       |
| Logistic Regression  |  77%      |


## Conclusion

### Summary

Analysis:
- Statistical tests (ANOVA) showed no significant difference in both the README lengths and number of unique words among the 3 groups (Python, Javascript, and Other Repos).
- "data" was the most common word for Python repos while "spring" was the most common word for Javascript repos

Modeling:
- Baseline accuracy was 71% on the training set and 74% on validation set.
- The only model to beat baseline was the Logistic Regression model, which had slightly better accuracy scores or 73% on the training set and 77% on the validation set.
- The models not being able to beat baseline reflect the exploratory insights showing little differences in the README lengths and number of unique words among repo groups.

### Next Steps
- Analyze less popular python repos as the differences in python and javascript repos may be greater among less-popular, more-specific repos.


[Back to top](#top)

---


<a name="data-dictionary"></a>

## Data Dictionary

| Column         | Description                                 |
|-----------------|---------------------------------------------|
| repo            | GitHub repository name                     |
| language        | Programming language of the repository     |
| readme_contents | Contents of the README file                |
| clean_text      | Cleaned version of the README contents     |
| stem            | Stemmed version of the cleaned text        |
| lemmatize       | Lemmatized version of the cleaned text     |
| target          | Target variable (e.g., classification label)|


[Back to top](#top)