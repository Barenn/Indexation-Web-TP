# -*- coding: utf-8 -*-

"""
Web Indexing and Data Mining - ENSAI 2020
Authors : BERNARD Renan & LETOUQ Mathilde

This  module  contains all  the functions used for  the Web Indexind and
Data Mining course.
"""
import os
import re
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def generate_texts_dataframe():
    """
    This function generates a pandas DataFrame  containing the documents
    and informations about them.

    Returns:
        pandas.DataFrame: "Text" : raw text of the document
                          "Author" : author of the document
                          "Document_d" : id of the document

    TODO:
        Get the path through a parameters.
    """

    documents = []
    texts = []
    authors = []

    for folder_name in os.listdir("./data/"):
        for file in os.listdir("./data/" + folder_name):
            documents.append(file)
            author = " ".join(re.findall(
                r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', folder_name))
            with open("./data/" + folder_name + "/" + file, "r") \
                as current_file:
                texts.append(current_file.read())
                authors.append(author)

    document_ids = [re.findall(r'\d+', document_id)[0] \
        for document_id in documents]

    texts = pd.DataFrame({'Text' : texts,
                          "Author" : authors,
                          "DocumentId" : document_ids})
    return texts

def tokenize_text(text):
    """
    This function transforms a raw text into a list of tokens.

    The words are  found using a  regular expression, then the stopwords
    are removed, and finally the words are stemmed.

    Args:
        text (string): the raw text to transform.

    Returns:
        list: a list containing the tokens.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    reg_ex = re.compile(r'[A-Za-z]+')
    bag_of_tokens = []
    text = reg_ex.findall(text.lower())
    for word in text:
        if len(word) > 1 and word not in stop_words:
            bag_of_tokens.append(stemmer.stem(word))
    return bag_of_tokens

def generate_query_dataframe(query, index):
    """
    This function generates a pandas.DataFrame containing the number of
    occurences of every tokens in every documents, if said document as
    at least one occurence of every token.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.

    Returns:
        pandas.DataFrame : a dataframe with the documents as index, and
        the tokens as columns.
    """
    query = tokenize_text(query)
    df_results_list = []

    if len(query) == 0:
        print("Invalid query.")
        return None

    for token in query:
        if token in index.keys():
            df_results = pd.DataFrame(index[token])
            df_results = df_results.drop(["total_occurences"], axis=1)
            df_results = df_results.drop(["locations"], axis=0)
            df_results = df_results.T
            df_results = df_results.rename(
                columns={'occurences' : token})
            df_results_list.append(df_results)
        else:
            print(token, "not in index.")
    # then join by keeping only documents with every token at least once
    df_results = df_results_list[0]
    for df_current in df_results_list[1:]:
        df_results = df_results.join(df_current, how='inner')

    return df_results

def query_by_occurences(query, index):
    """
    This function generates the results of a query by total number of
    occurences over every token.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index, and the tokens as columns, ordered by total number of
        tokens.
    """
    df_query = generate_query_dataframe(query, index)
    df_query_sum = pd.DataFrame(df_query.sum(axis=1), columns=['Total'])
    df_query = pd.concat([df_query, df_query_sum], axis=1)
    df_query = df_query.sort_values('Total', ascending=False)
    return df_query

def query_by_ordered_occurences(query, index):
    """
    This function generates the results of a query respecting the
    tokens' order of the query. The results are ordered by number of
    occurences.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index and a single column containing the number of occurences,
        ordered by total number of tokens.
    """
    query_results = query_by_occurences(query, index)
    tokens = list(query_results.columns)[:-1]

    list_results = []

    no_result = True

    for doc in query_results.index:
        positions = np.asarray(
            index[tokens[0]][doc]['locations'])
        positions = set(positions)

        for i in range(1, len(tokens)):
            token = tokens[i]
            positions_current = np.asarray(
                index[token][doc]['locations'])
            positions_current = positions_current - i
            positions_current = set(positions_current)
            positions = positions.intersection(
                positions_current)

        if len(positions) != 0:
            to_add = {'Document' : doc, 'Occurences' : len(positions)}
            list_results.append(to_add)
            no_result = False

    if no_result:
        print("No match found.")
        return 0

    df_results = pd.DataFrame(list_results)
    df_results = df_results.sort_values('Occurences', ascending=False)
    df_results = df_results.set_index('Document')
    return df_results

def search_words(target, index):
    """
    TODO : docstring
    """
    target_list = tokenize_text(target)
    all_words = []
    res = {'words': dict(), "all_words": set()}
    positions = []

    if len(target_list) == 0:
        print("Invalid query.")
        return None

    for word in target_list:
        if word in index.keys():
            print(str("\"" + word + "\""), "appears in",
                  len(index[word].keys()), "documents.")
            all_words.append(set(index[word]))
            res['words'][word] = set(index[word].keys())

        else:
            print(word, "not in index.")

    for i in range(1, len(all_words)):
        all_words[0] = all_words[0].intersection(all_words[i])

    res['all_words'] = set(all_words[0])
    print("All words in", len(res['all_words']), "documents.")

    if len(target_list) > 1:
        res['exact_matches'] = set()

        for doc in res['all_words']:
            if doc != "total_occurences":
                positions = np.asarray(
                    index[target_list[0]][doc]['locations'])
                positions = set(positions)

                for i in range(1, len(target_list)):
                    positions_current = np.asarray(
                        index[target_list[i]][doc]['locations'])
                    positions_current -= i
                    positions_current = set(positions_current)
                    positions = positions_current.intersection(
                        positions)

                if len(positions) != 0:
                    res['exact_matches'].add(doc)

        print("Exact matches in ",
              len(res['exact_matches']), "documents.")

    return res

def view_article(article_index, texts):
    """
    This function prints the  specified article through its index in the
    DataFrame.

    Args:
        article_index (int): the index of the target article in texts.
        texts (pandas.DataFrame): the DataFrame containing the article,
        generated through generate_texts_dataframe(...).
    """
    text = texts.Text.values[article_index]
    author = texts.Author.values[article_index]
    document_id = texts.DocumentId[article_index]
    to_print = "\n--------------------------------------------"
    to_print += "\nAuthor : " + str(author) + "\nId : "
    to_print += str(document_id)
    to_print += "\n--------------------------------------------"
    print(to_print)
    print(text)

def create_index_from_text(text, text_id):
    """
    This function creates an index through the raw text provided.

    It acts as the map function for index creation.

    Args:
        text (string): the raw text.
        text_id (string): an idea for the text, when using text from the
        DataFrame  generated  with  generate_texts_dataframe(...),  this
        should be its index in said DataFrame.

    Results:
        dict: the resulting index, it has the following architecture:
            {"token1" : {"total_occurences" : 4,
                         ind1 : {"locations" : [12, 13, 14],
                                 "occurences" : 3},
                         ind2 : {"locations" : [9],
                                 "occurences" : 1}}}
    """
    text = tokenize_text(text)
    index = dict()
    for i, token in enumerate(text):
        if token in index:
            index[token][text_id]["locations"].append(i)
            index[token][text_id]["occurences"] += 1
            index[token]["total_occurences"] += 1
        else:
            index[token] = {"total_occurences" : 1,
                            text_id : {"locations" : [i],
                                       "occurences" : 1}}
    return index

def sum_two_indexes(index1, index2):
    """
    This function is a summation of two indexes as used in this module.
    It serves as a reducer for index creation.

    It  adds to the  first index the locations  contained in  the second
    index,  respecting  the   index architecture. It   also  updates the
    "total_occurences" values.

    Args:
        index1 (dict): the first index to sum.
        index2 (dict): the second index.

    Returns:
        dict: the indexes' sum.

    TODO:
        A more accurate dosctring.

    """
    for token in list(index2.keys()):
        if token in index1:
            total_occurences = index1[token]["total_occurences"]
            index1[token].update(index2[token])
        else:
            total_occurences = 0
            index1[token] = index2[token]
        index1[token]["total_occurences"] += total_occurences
    return index1

def generate_tokens_count(index):
    """
    This  function  counts the tokens in  each documents index, and also
    the  total   number of  tokens in  the  entire  index  by  using the
    corresponding index.

    Args:
        index (dict): the index from which we want to count the tokens.

    Returns:
        dict: the tokens count in a dictionnary such as follow :
                {"total_occurences" : number of tokens in the corpus,
                 ind1 : number of tokens in the document ind1}
    """
    tokens_count = dict()
    for token in list(index.keys()):
        for key in list(index[token].keys()):
            if (key == "total_occurences") and \
                ("tokens_total" in tokens_count):
                tokens_count["tokens_total"] += index[token][key]
            elif key == "total_occurences":
                tokens_count["tokens_total"] = index[token][key]
            elif key in tokens_count:
                tokens_count[key] += index[token][key]["occurences"]
            else:
                tokens_count[key] = index[token][key]["occurences"]
    return tokens_count

def calculate_tf_idf_for_token(index_token, tokens_count):
    """
    This function calculates the tf_idf for a token in each document.
    It returns the coordinates of each document for the token  dimension
    in the tfidf "space".
    Args:
        index_token (dict): the specific index part for a token i, it is
        index[token].
        tokens_count (dict): the  tokens_count  generated with generate_
        tokens_count(...)

    Returns:
        numpy.array: contains the tfidf for the specified token for each
        documents, its shape is (number of documents, 1).

    """
    tfidf_array = np.zeros((1, len(tokens_count) - 1))
    for text_id in list(index_token.keys())[1:]:
        term_freq = index_token[text_id]["occurences"]
        term_freq *= 1 / tokens_count[text_id]
        inverse_document_freq = (len(tokens_count) - 1)
        inverse_document_freq *= 1 / (len(index_token) - 1 + 1)
        inverse_document_freq = np.log(inverse_document_freq)
        tfidf_array[0][text_id] = term_freq * inverse_document_freq
    return tfidf_array.T

def vectorize_tfidf(query, index):
    """
    This  function  vectorizes a text  in the same  space (tfidf) as the
    documents.

    Args:
        query (string): raw text to vectorize.
        index (dict): this index of the corresponding tfidf space.

    Returns:
        numpy.array: contains the vector in the tfidf space of the text.
    """
    tfidf_array = np.zeros((len(index.keys())))
    query_index = create_index_from_text(query, "query")
    tokens_count = generate_tokens_count(query_index)
    counter = 0
    for token in list(index.keys()):
        if token in query_index:
            term_freq = query_index[token]["query"]["occurences"]
            term_freq = term_freq / tokens_count["query"]
            inverse_document_freq = 2500
            inverse_document_freq *= 1 / (len(index[token]) - 1 + 1)
            inverse_document_freq = np.log(inverse_document_freq)
            tfidf_array[counter] = term_freq
            tfidf_array[counter] *= inverse_document_freq
        counter += 1
    return tfidf_array

def query_corpus(query, index, matrix_tfidf, texts, nb_to_show=3):
    """
    This function queries the corpus with a raw text.
    It returns the corpus' documents that have the highest dot product.

    Args:
        query (string): raw text to query.
        index (string): the index corresponding to the corpus.
        matrix_tfidf (numpy.array): the matrix  that  correspond  to the
        corpus vectors in the tfidf space.
        texts (pandas.DataFrame): the corpus dataframe.
        nb_to_show (int): numbers of documents to provide.
    """
    dot_product = np.dot(matrix_tfidf, vectorize_tfidf(query, index))
    inds = np.argsort(dot_product)[::-1][:nb_to_show]
    dot_product = np.sort(dot_product)[::-1][:nb_to_show]
    results_df = pd.DataFrame({"DotProduct" : dot_product,
                               "Text" : texts.Text[inds],
                               "Author" : texts.Author[inds],
                               "DocumentId" : texts.DocumentId[inds]})
    return results_df
