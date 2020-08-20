
import gzip

from flask import Flask, render_template, request, redirect
app = Flask(__name__)


import numpy as np
import pandas as pd
import sklearn
# import matplotlib.pyplot as plt
# import seaborn as sns
# import nltk
from gensim.models import Word2Vec
# from gensim.models import KeyedVectors

import pickle
import re


# w2v_model= pickle.load(open( "w2v.pkl", "rb" ))

# with gzip.open('w2v_new.pkl.gz', 'wb') as f:
# 	pickle.dump(w2v_model, f, protocol=-1)



classifier= pickle.load(open( "classifier_new.pkl", "rb" ))

# with gzip.open('classifier.pkl.gz', 'wb') as f:
# 	pickle.dump(classifier, f, protocol=-1)




filename='w2v_new.pkl.gz'
with gzip.open(filename, 'rb') as f:
	w2v_model=pickle.load(f)
	

# filename='classifier.pkl.gz'
# with gzip.open(filename, 'rb') as f:
# 	classifier=pickle.load(f)












def remove_punctuations(txt):
    txt= re.sub('[^a-zA-Z]', ' ', txt)
    txt= re.sub(' [a-zA-Z][a-zA-Z] ', ' ', txt)
    txt= re.sub(' [a-zA-Z] ', ' ', txt)
    txt= re.sub(r'\s+', " ", txt)
    txt= txt.lower()
    
    return txt



def vectorize_text(text):
	global vectorizer
	words= text.split()
	w2v_words= list(w2v_model.wv.vocab)
	sent_vec = np.zeros(100) # as word vectors are of zero length 50, you might need to change this to 300 if you use google's w2v
	cnt_words =0 # num of words with a valid vector in the sentence/review
	for word in words: # for each word in a review/sentence
		if word in w2v_words:
			vec = w2v_model.wv[word]
			sent_vec += vec
			cnt_words += 1
	if cnt_words != 0:
		sent_vec /= cnt_words

	return sent_vec


#     # sent_vectors.append(sent_vec)






def classify(vec):
	global classifier
	
	result=classifier.predict([vec])

	if result[0]==0:
		return 'Negative Review'
	else:
		return 'Positive Review'

	# return predict_this



# review= 'Absolutely loved the movie and its character so well directed. Enjoyed it thoroughly. Great work! hope such work continues.'

# cleaned= remove_punctuations(review)
# print(cleaned)

# vec= vectorize_text(cleaned)
# print(vec)

# print(classify(vec))





@app.route("/")
@app.route("/Home" , methods=["GET","POST"])
def home():
	
	if request.method=="POST":
		
		review=str(request.form["user_review"])
		cleaned= remove_punctuations(review)
		vec= vectorize_text(cleaned)
		a=classify(vec)
		return render_template("movie_review_nlp.html", ans=a)
		# except:
		# 	return "Something went wrong."

	return render_template("movie_review_nlp.html", ans='ok')




# app.run(debug=True)

if __name__ == "__main__":
	app.run(debug=True)





