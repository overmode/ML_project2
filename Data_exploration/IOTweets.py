import pandas as pd

def build_df(filepath):
    """from a cut_vocab return a dataframe which is a mapping of words in tweets
    with their occurences in all tweets
    take the path of the file of the tweets
    """

    df = pd.read_table(filepath_or_buffer = filepath, header=None, names=["word"])
    df["occurence"] = df["word"].map(lambda x:  int(x.split()[0]))
    df["word"] = df["word"].map(lambda x:  x.split()[1])
    return df

def import_(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip() for line in f]     # Make sure to withdraw the "nbr",
    return tweets

def import_without_comma(path):
    with open(path, 'r', encoding="utf-8") as f:
        tweets = [line.strip()[line.find(",")+1:] for line in f]     # Make sure to withdraw the "nbr",
    return tweets

def export(tweets, name):
    with open(name, 'w', encoding="utf-8") as f:
        f.write("\n".join(tweets))
