import pandas as pd
import re
import gensim
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import logging

logger = logging.getLogger(__name__)



# create sample documents

def get_questions_text(questions_df):
# Given a pandas dataframe containing the questions, this function merges text and descriptions
# filename: this is the name of the csv file containing the questions
# the return value is an array containing the merge of text and description for each question
    doc_desc = questions_df.q_desc.values.tolist()
    doc_quest = questions_df.q_text.values.tolist()
    doc_set = []
    for i, v in zip(doc_quest, doc_desc):
        doc_set.append(i + v)
    return doc_set



def preprocess_text(doc_set):
# This function removes the stopwords and useless words/strings
# doc_set: this is an array containing the question data, each element is a question
# the return value is the array of question data but cleaned from all the stopwords and words we do not want
	list_of_removal_words = ['\'s','e.g.','(', ')','-','_',',',';',':','i.e.','*','.','\''] 
	list_of_removal_regex = [r"http\S+", r"Http\S+", r"HTTP\S+", r"www\S+", r"WWW\S+"]
	stopwords = ['will','\'s','e.g.,','i.e.,']
	    
	for i in range(len(doc_set)):
	     question = doc_set[i]
	     for string in list_of_removal_words:
	         question = question.replace(string, " ")

	     for regex in list_of_removal_regex:
	        question = re.sub(regex, "", question) 

	     querywords = question.split()
	     resultwords  = [word for word in querywords if word.lower() not in stopwords]
	     doc_set[i] = ' '.join(resultwords)
	return doc_set



def get_corpus(doc_set):
# Gives corpus of text in output after having applied tokenization and stem-ization
# doc_set: is the set of documents already cleaned after a preprocessing (preprocess text)
# return value is an array of corpus text which are cleaned tokenized and stemmed
    tokenizer = RegexpTokenizer(r'\w+')
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    texts = []
    # loop through document list

    for i in doc_set:
        i = i.decode('utf-8', 'ignore').encode("utf-8")
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
    
    	# remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
    	    
    	# stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    	    
        # add tokens to list
        texts.append(stemmed_tokens)
    
        # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    	 
    	# convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary 
    

def generate_lda_model(num_topics, corpus, passes, dictionary):
# generate LDA model
    num_topics = 6
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word =
            dictionary, passes=20)
    return ldamodel


def save_topics_to_file(filename, num_words = 20):
    with open(filename, 'a') as out:
    	for i in ldamodel.print_topics(num_words=20):
    		out.write(str(i) + '\n')


#if __name__ == "__main__":
filename_in         = "data/ifps.csv"
filename_out_topics = "data/topics_structure.txt"

questions = pd.read_csv(open(filename_in, 'rU'), sep=None, engine='python')

doc_set = get_questions_text(questions)
doc_set = preprocess_text(doc_set)

corpus, dictionary	= get_corpus(doc_set)
num_topics = 7
passes = 20

ldamodel = generate_lda_model(num_topics, corpus, passes, dictionary)
save_topics_to_file(filename_out_topics)

for i in range(num_topics):
    questions['topic_' + str(i)] = 0

# Let's see how the model assigns topic on a new question string
doc_lda = []

for i in range(len(doc_set)):
    #print("The question is {}".format(doc_set[i]))
    doc_lda.append(ldamodel[corpus[i]])
    #print("The result is {}".format(doc_lda))



for index, row in questions.iterrows():
    for i in doc_lda[index]:
            questions.loc[index,'topic_' + str(i[0])] = i[1]


questions.to_csv('data/questions_w_topics.csv', index=False)
