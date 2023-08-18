from collections import Counter
from nltk.tokenize import word_tokenize

import seaborn as sns
import matplotlib.pyplot as plt


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
    
