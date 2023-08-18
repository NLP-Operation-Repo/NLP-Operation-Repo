from collections import Counter
from nltk.tokenize import word_tokenize

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.stats import f_oneway


def find_most_common_words(df,
                           language = 'all',
                           stem_or_lem = 'lemmatize',
                           num_words = 20,
                           plot = True):
    
    df = df[df['target']==language] if type(language) == int else df
    all_readme_contents = ' '.join(df[stem_or_lem])
    tokens = word_tokenize(all_readme_contents)
    word_counts = Counter(tokens)
    most_common_words = word_counts.most_common(num_words)
    
    if plot:
        colors = sns.color_palette("Blues", len(most_common_words))[::-1]
        sns.barplot(y = [word_freqs[0] for word_freqs in most_common_words],
                    x = [word_freqs[1] for word_freqs in most_common_words],
                    palette=colors)
        language_titles = {1: 'Python', 2: 'Javascript', 0: 'Other'}

        lang_title = language_titles.get(language, 'All')
        plt.title(f'{str(num_words)} Most Common Words for {lang_title}')
        sns.despine()
        plt.show()
    
    return most_common_words


def plot_readme_lengths():
    
    df['readme_length'] = df['readme_contents'].apply(len)
    readme_lens = df.groupby('target')['readme_length'].median()
    index_mapping = {0: 'other', 1: 'python', 2: 'javascript'}
    readme_lens.index = readme_lens.sort_index().index.map(index_mapping)
    
    colors = sns.color_palette("Blues_r", len(readme_lens))
    sns.barplot(x = readme_lens.values,
                y = readme_lens.index,
                palette=colors)
    plt.title('README Lengths by Language')
    
    plt.tick_params(axis='both', left=False)
    plt.ylabel('')
    plt.xlabel('README Length')
    sns.despine()
    plt.show()
    
    
    
def plot_unique_word_averages():
    
    df['unique_words'] = df['readme_contents'].apply(lambda x: len(set(x.split())))
    unique_word_avgs = df.groupby('target')['unique_words'].median()
    index_mapping = {0: 'other', 1: 'python', 2: 'javascript'}
    unique_word_avgs.index = unique_word_avgs.sort_index().index.map(index_mapping)
    
    colors = sns.color_palette("Blues_r", len(unique_word_avgs))
    sns.barplot(x = unique_word_avgs.sort_values(ascending=False).values,
                y = unique_word_avgs.sort_values(ascending=False).index,
                palette=colors)
    plt.title('Average Number of Unique Words by Language')
    
    plt.tick_params(axis='both', left=False)
    plt.ylabel('')
    plt.xlabel('Number of Unique Words')
    sns.despine()
    plt.show()
    
def plot_unique_word_averages():
    
    df['unique_words'] = df['readme_contents'].apply(lambda x: len(set(x.split())))
    unique_word_avgs = df.groupby('target')['unique_words'].median()
    index_mapping = {0: 'other', 1: 'python', 2: 'javascript'}
    unique_word_avgs.index = unique_word_avgs.sort_index().index.map(index_mapping)
    
    colors = sns.color_palette("Blues_r", len(unique_word_avgs))
    sns.barplot(x = unique_word_avgs.sort_values(ascending=False).values,
                y = unique_word_avgs.sort_values(ascending=False).index,
                palette=colors)
    plt.title('Average Number of Unique Words by Language')
    
    plt.tick_params(axis='both', left=False)
    plt.ylabel('')
    plt.xlabel('Number of Unique Words')
    sns.despine()
    plt.show()
    
def identify_unique_words(plot=True):
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['lemmatize'])
    
    # Convert TF-IDF matrix to DataFrame
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
                            columns=tfidf_vectorizer.get_feature_names_out())
    
    # Add the 'target' column back to the DataFrame
    tfidf_df['target'] = df['target']
    
    # Group the data by programming language and calculate the mean TF-IDF scores for each word
    language_tfidf = tfidf_df.groupby('target').mean()
    
    for i, language in enumerate(['Other', 'Python', 'Javascript']):
        # Get words with the highest mean TF-IDF score for the specified programming language
        
        top_words = language_tfidf.loc[i].sort_values(ascending=False).head(5)
        print(f"{language}:")
        print(top_words, '\n')
    
        if plot:
            plt.figure(figsize=(6,3))
            colors = sns.color_palette("Blues", len(top_words))[::-1]
            sns.barplot(x = top_words.sort_values(ascending=False).values,
                        y = top_words.sort_values(ascending=False).index,
                        palette=colors)
            language_titles = {1: 'Python', 2: 'Javascript', 0: 'Other'}

            lang_title = language_titles.get(language, 'All')
            plt.title(f'Unique Words for {language}')

            plt.tick_params(axis='both', left=False, bottom=False)
            plt.xlabel('TF-IDF score')
            sns.despine()
            plt.show()
        print('-'*50)
        
        
def anova_test(df):
    
    df['unique_words'] = df['readme_contents'].apply(lambda x: len(set(x.split())))

    # Group the data by programming language and calculate the mean number of unique words
    language_unique_words = df.groupby('target')['unique_words'].mean()

    # Extract unique words data for each programming language
    python_unique_words = df[df['target'] == 1]['unique_words']
    javascript_unique_words = df[df['target'] == 2]['unique_words']
    other_unique_words = df[df['target'] == 0]['unique_words']

    # Perform ANOVA test
    f_statistic, p_value = f_oneway(python_unique_words, javascript_unique_words, other_unique_words)

    # Interpret the results
    if p_value < 0.05:
        print("There are significant differences in the mean number of unique words among different programming languages.")
    else:
        print("There are no significant differences in the mean number of unique words among different programming languages.")