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

def view_article(article_ind, texts):
    """
    This function prints the  specified article through its index in the
    DataFrame.

    Args:
        article_ind (int): the indice of the target article in texts.
        texts (pandas.DataFrame): the DataFrame containing the article,
        generated through generate_texts_dataframe(...).
    """
    text = texts.Text.values[article_ind]
    author = texts.Author.values[article_ind]
    document_id = texts.DocumentId[article_ind]
    to_print = "\n--------------------------------------------"
    to_print += "\nAuthor : " + str(author) + "\nId : "
    to_print += str(document_id)
    to_print += "\n--------------------------------------------"
    print(to_print)
    print(text)

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

def query_by_occurences(query, index, texts):
    """
    This function generates the results of a query by total number of
    occurences over every token.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index, and the tokens as columns, ordered by total number of
        tokens.
    """
    df_query = generate_query_dataframe(query, index)
    df_query_sum = pd.DataFrame(df_query.sum(axis=1), columns=['Total'])
    df_query = pd.concat([df_query, df_query_sum], axis=1)
    df_query = df_query.sort_values('Total', ascending=False)
    df_query = df_query.join(texts)
    return df_query

def query_by_ordered_occurences(query, index, texts):
    """
    This function generates the results of a query respecting the
    tokens' order of the query. The results are ordered by number of
    occurences.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index and a single column containing the number of occurences,
        ordered by total number of tokens.
    """
    # Get the results of a simple query to get occurences of tokens.
    query_results = query_by_occurences(query, index, texts)
    columns_to_drop = ['Total', 'Text', 'DocumentId', 'Author']
    query_results = query_results.drop(columns_to_drop, axis=1)
    tokens = list(query_results.columns)

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
    df_results = df_results.join(texts)
    df_results.index.name = None
    return df_results

def query_by_frequences(query, index, texts):
    """
    This function generates the results of a query ordered by the
    frequence of every token inside the documents.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index, and the tokens as columns, ordered by frequence of
        tokens.
    """
    tokens_count = generate_tokens_count(index)
    tokens_count.pop("tokens_total")
    tokens_count = pd.DataFrame(tokens_count.items(),
                                columns=["Document", "TokensTotal"])
    tokens_count = tokens_count.set_index('Document')
    tokens_count.index.name = None

    df_query = generate_query_dataframe(query, index)
    for token in df_query.columns:
        df_query[token] = df_query[token].divide(
            tokens_count.loc[df_query.index].TokensTotal)
    df_query_sum = pd.DataFrame(df_query.sum(axis=1), columns=['Total'])
    df_query = pd.concat([df_query, df_query_sum], axis=1)
    df_query = df_query.sort_values('Total', ascending=False)
    df_query = df_query.join(texts)
    df_query = df_query.dropna()
    return df_query

def generate_idf_dataframe(index, texts):
    """
    This function generates a pandas.DataFrame containing the inverse
    document frequency for every token.

    Args:
        index (dict) : the index as created in this .py.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        pandas.DataFrame : contains two columns, the tokens and their
        corresponding inverse document frequency.
    """
    inverse_doc_freq = [{"Token" : key,
                         "IDF" : len(values.keys()) - 1}
                        for key, values in index.items()]
    inverse_doc_freq = pd.DataFrame(inverse_doc_freq)
    inverse_doc_freq['IDF'] = inverse_doc_freq['IDF'].apply(
        lambda x: np.log(len(texts) / x + 1))
    return inverse_doc_freq

def query_by_tfidf(query, index, texts):
    """
    This function generates the results of a query ordered by the tf-idf
    of every token inside the documents.

    Args:
        query (string): the queried text.
        index (dict) : the index as created in this .py.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        pandas.DataFrame : the dataframe containing the documents as
        index, and the tokens as columns, ordered by tf-idf of tokens.
    """
    results_query = query_by_frequences(query, index, texts)
    results_query = results_query.drop(['Total'], axis=1)
    inverse_doc_freq = generate_idf_dataframe(index, texts)
    query = tokenize_text(query)
    inverse_doc_freq = inverse_doc_freq[
        inverse_doc_freq.Token.isin(query)]
    inverse_doc_freq = list(inverse_doc_freq['IDF'])
    results_query[query] = results_query[query].mul(
        inverse_doc_freq, axis=1)
    results_query_sum = pd.DataFrame(
        results_query[query].sum(axis=1), columns=['Total'])
    results_query['Total'] = results_query_sum
    results_query = results_query.sort_values('Total')
    return results_query

def generate_tokens_count(index):
    """
    This  function  counts the tokens in  each documents index, and also
    the  total   number of  tokens in  the  entire  index  by  using the
    corresponding index.

    Args:
        index (dict): the index from which we want to count the tokens.

    Returns:
        dict: the tokens count in a dictionnary such as follow :
                {"tokens_total" : number of tokens in the corpus,
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

def calculate_tf_idf_matrix(index, texts):
    """
    This function calculates the TF-IDF matrix, the representation of
    all documents in the TF-IDF space.

    Args:
        index (dict): the index from which we want to count the tokens.
        texts (pandas.DataFrame) : the dataframe generated using the
        generate_texts_dataframe function.

    Returns:
        numpy.array : shape (number of documents, number of tokens).
    """
    tf_idf_matrix = np.zeros((len(texts), len(index.keys())))
    # Fetching number of occurences for every tokens by documents.
    for i, value in enumerate(index.values()):
        values = [d['occurences'] for d in list(value.values())[1:]]
        tf_idf_matrix[np.array(list(value.keys())[1:]), i] = values
    # Divide each lines by its sum to get the frequences.
    tf_idf_matrix = tf_idf_matrix / tf_idf_matrix.sum(
        axis=0, keepdims=1)
    idf_matrix = np.array(
        generate_idf_dataframe(index, texts)['IDF'])
    tf_idf_matrix = np.multiply(tf_idf_matrix, idf_matrix)
    return tf_idf_matrix

def vectorize_tfidf(query, index, texts):
    """
    This  function  vectorizes a text  in the same  space (tfidf) as the
    documents.

    Args:
        query (string): raw text to vectorize.
        index (dict): this index of the corresponding tfidf space.

    Returns:
        numpy.array: contains the vector in the tfidf space of the text.
    """
    index_query = create_index_from_text(query, 'query')
    tokens_count = generate_tokens_count(index_query)['tokens_total']
    idf_df = generate_idf_dataframe(index, texts).set_index('Token')

    for token in index_query:
        idf_df.loc[token] = (idf_df.loc[token] *
                             index_query[token]['total_occurences'])
        idf_df.loc[token] = idf_df.loc[token] / tokens_count

    idf_df[~idf_df.index.isin(index_query.keys())] = 0

    return np.array(idf_df['IDF'])

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
    query_vector = vectorize_tfidf(query, index, texts)
    dot_product = np.dot(matrix_tfidf, query_vector)
    inds = np.argsort(dot_product)[::-1][:nb_to_show]
    dot_product = np.sort(dot_product)[::-1][:nb_to_show]
    results_df = pd.DataFrame({"DotProduct" : dot_product,
                               "Text" : texts.Text[inds],
                               "Author" : texts.Author[inds],
                               "DocumentId" : texts.DocumentId[inds]})
    return results_df
