# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 07:27:52 2018

@author: Bill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def is_int(test):

    try:
        int(test)
        return True

    except:
        return False


def not_int(test):
    try:
        int(test)
        return False
    except:
        return True


def get_ticket_digits(test):
    '''Accepts full ticket string, returns digital portion'''
    tmp = str(test).split(' ')
    return int(tmp[-1])


def guess_decks(df):
    '''Parses known cabins for deck data'''

    decks = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for i in df.index.values:
        if df["Cabin"].isnull().loc[i]:
            df.loc[i, "Deck"] = "Unknown"

        else:

            if (df.loc[i, "Cabin"][0] in decks):
                df.loc[i, "Deck"] = df.loc[i, "Cabin"][0]
            else:
                df.loc[i, "Deck"] = df.loc[i, "Cabin"]

        if str(df.loc[i, "Cabin"]).rfind('F G') >= 0:
            df.loc[i, "Deck"] = "Unknown"
    return df


def clean_data_best_assumptions(df):
    '''Uses best known assumptions, including:
        Salultation translation from Rajitha
        Last Name strip from Pranesh
        '''
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Last_Name'] = df.Name.str.extract('([A-Za-z]+)\,', expand=False)
    # Process titles

    titles = df.Title.unique()
    for t in titles:
        age = df[df['Title'] == t]['Age'].mean()
        df.loc[df['Title'] == t, 'Age'] = \
            df.loc[df['Title'] == t, 'Age'].fillna(age)

    for p in range(1, 4):
        fare = df[df['Pclass'] == p]['Fare'].mean()
        df.loc[(df['Fare'] == 0) & (df['Pclass'] == p), 'Fare'] = fare

    df['Embarked'] = df['Embarked'].fillna('S')

    return df


def survived_or_not(df, bins):
    plt.close()
    tmp_df = df.copy()
    tmp_df = tmp_df.loc[(tmp_df.Survived == 1), :]
    plt.hist(tmp_df.Tick_dig, bins=bins, color='b', alpha=.75, label="Lived")

    tmp2_df = df.copy()
    tmp2_df = tmp2_df.loc[(tmp2_df.Survived == 0), :]
    plt.hist(tmp2_df.Tick_dig, bins=bins, color='r', alpha=.5, label="Died")

    plt.legend(loc="best")
    plt.show()

    return


def translate_categorical_variables(df):
    df["Sex_c"] = df["Sex"].apply(lambda x: 1 if x == "female" else 2)

    edict = {"S": 1, "C": 2, "Q": 3}
    df["Embarked_c"] = df.Embarked.apply(lambda x: edict[x])

    deck_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7,
                 "H": 8, "Unknown": 9, "Cabin": 10, 'F G73': 11, "T": 11}
    df["Deck_c"] = df.Deck.apply(lambda x: deck_dict[x])

    df["Tick_dig_c"] = df["Tick_dig"].apply(lambda x: int(x * 10000)/10000)

    title_dict = {'Sir': 1, 'Major': 1, 'Don': 1, 'Mlle': 3, 'Capt': 1,
                  'Dr': 1, 'Lady': 2, 'Rev': 1, 'Mrs': 2, 'Jonkheer': 2,
                  'Countess': 2, 'Master': 3, 'Ms': 2, 'Mr': 1, 'Mme': 2,
                  'Miss': 2, 'Col': 1, 'Dona': 2}
    df["Title_c"] = df.Title.apply(lambda x: title_dict[x])

    # Check of missing ages that correspond to titles not in training file
    # ex: Dona

    for a in range(1, 4):
        for s in [1, 2]:
            age = df.loc[((df.Title_c == a) & (df.Sex_c == s)), "Age"].mean()
            df.loc[((df.Title_c == a) & (df.Sex_c == s)), 'Age'] =\
                df.loc[((df.Title_c == a) & (df.Sex_c == s)),
                       'Age'].fillna(age)
    return df


def prep_analysis(df_raw):
    df_raw = clean_data_best_assumptions(df_raw)
    df_raw = guess_decks(df_raw)
    # df_raw = df_raw.drop('Age', 1)
    df_raw = df_raw.loc[(df_raw.Ticket != 'LINE'), :]
    df_raw["Tick_dig"] = df_raw["Ticket"].apply(lambda x: get_ticket_digits(x))
    df_raw = translate_categorical_variables(df_raw)
    df_raw.dropna(axis=0)

    # Choose variables to analyze and build your dataframe here
    X = df_raw.loc[:, ["Pclass", "Sex_c", "Embarked_c", "Age"]]
    try:
        y = df_raw["Survived"]
    except:
        print "Loading Test Data"
        y = 0

    return X, y


def generate_knn_prediction(df_train, df_test):

    X_tr, y = prep_analysis(df_train)
    X_test, y_tmp = prep_analysis(df_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y)

    prediction = knn.predict(X_test)

    return prediction

def generate_submission(df_test, prediction):
    submission = pd.DataFrame({
        "PassengerId": tst_df["PassengerId"],
        "Survived": prediction
    })

    submission.to_csv('C:\\Users\\Bill\\Desktop\\Springboard\\Titanic\\\
                  mysubmission.csv', index=False)

    print "This trial predicts",np.sum(prediction),"survivors"
    return

#  Program starts

filename = "C:\\Users\\Bill\\Desktop\\Springboard\\Titanic\\titanic_training_data.csv"
train_df = pd.read_csv(filename)
tr_df = train_df

filename = "C:\\Users\\Bill\\Desktop\\Springboard\\Titanic\\titanic_test.csv"
tst_df = pd.read_csv(filename)


# Take all this out after program finalized
tr_df = clean_data_best_assumptions(tr_df)
tr_df = guess_decks(tr_df)

#tr_df = tr_df.drop('Age', 1)
tr_df = tr_df.loc[(tr_df.Sex == 'female') & (tr_df.Ticket != 'LINE'), :]
tr_df["Tick_dig"] = tr_df["Ticket"].apply(lambda x: get_ticket_digits(x))
tr_df = tr_df.loc[(tr_df.Age <= 18), :]
#tr_df = tr_df.loc[tr_df["Ticket"].apply(lambda x: not_int(x)), :]
#tr_df = tr_df.loc[tr_df.Pclass == 1, :]
# tr_df = tr_df.loc[tr_df.Fare <= 50, :]
#tr_df = tr_df.loc[tr_df.Embarked == 'S', :]
#tr_df = tr_df.loc[tr_df.Deck == 'B', :]
bins = [x for x in range(0, 420000, 42000)]
survived_or_not(tr_df, bins)

#tr_df = translate_categorical_variables(tr_df)

tr_df = train_df
pred = generate_knn_prediction(tr_df, tst_df)

generate_submission(tst_df, pred)


## submission steps.
#submission = pd.DataFrame({
#        "PassengerId": tst_df["PassengerId"],
#        "Survived": prediction
#    })

#submission.to_csv('C:\\Users\\Bill\\Desktop\\Springboard\\Titanic\\\
#                  mysubmission.csv', index=False)
#print np.sum(prediction)
print 'done'
