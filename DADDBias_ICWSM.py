import pandas as pd
import gensim 
from gensim.models import Word2Vec
import numpy as np
import nltk
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy import spatial
from sklearn.cluster import KMeans
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

def TrainModel(csv_document, csv_comment_column='body', outputname='outputModel', window = 4, minf=10, epochs=100, ndim=200, lemmatiseFirst = False, verbose = True):
	'''
	Load the documents from csv_document and column csv_comment_column, trains a skipgram embedding model with given parameters and saves it in outputname.
	csv_document <str> : path to reddit csv dataset
	csv_comment_column <str> : column where comments are stored
	outputname <str> : output model name
	window = 4, minf=10, epochs=100, ndim=200, lemmatiseFirst = False, tolower= True : Training and preprocessing parameters
	'''

	def loadCSVAndPreprocess(path, column = 'body', nrowss=None, verbose = True):
		'''
		input:
		path <str> : path to csv file
		column <str> : column with text
		nrowss <int> : number of rows to process, leave None if all
		verbose <True/False> : verbose output
		tolower <True/False> : transform all text to lowercase
		returns:
		list of preprocessed sentences
		'''
		trpCom = pd.read_csv(path, lineterminator='\n', nrows=nrowss)
		documents = []
		for i, row in enumerate(trpCom[column]):
			

			if i%500000 == 0 and verbose == True:
				print('\t...processing line {}'.format(i))
			try:
				pp = gensim.utils.simple_preprocess (row)
				if(lemmatiseFirst == True):
					pp = [wordnet_lemmatizer.lemmatize(w, pos="n") for w in pp]
				documents.append(pp)
			except:
				print('\terror with row {}'.format(row))
		print('Done reading all documents')
		return documents

	def trainWEModel(documents, outputfile, ndim, window, minfreq, epochss):
		'''
		documents list<str> : List of texts preprocessed
		outputfile <str> : final file will be saved in this path
		ndim <int> : embedding dimensions
		window <int> : window when training the model
		minfreq <int> : minimum frequency, words with less freq will be discarded
		epochss <int> : training epochs
		'''
		starttime = time.time()
		print('->->Starting training model {} with dimensions:{}, minf:{}, epochs:{}'.format(outputfile,ndim, minfreq, epochss))
		model = gensim.models.Word2Vec (documents, size=ndim, window=window, min_count=minfreq, workers=5)
		model.train(documents,total_examples=len(documents),epochs=epochss)
		model.save(outputfile)
		print('->-> Model saved in {}'.format(outputfile))     

     
	print('->Starting with {} [{}], output {}, window {}, minf {}, epochs {}, ndim {}'.format(csv_document,csv_comment_column,outputname, window, minf, epochs, ndim))
	docs = loadCSVAndPreprocess(csv_document, csv_comment_column, nrowss=None, verbose=verbose)[:1000] #TODO REMOVE
	starttime = time.time()
	print('-> Output will be saved in {}'.format(outputname))
	trainWEModel(docs, outputname, ndim, window, minf, epochs)
	print('-> Model creation ended in {} seconds'.format(time.time()-starttime))


sid = SentimentIntensityAnalyzer()
def GetTopMostBiasedWords(modelpath, topk, c1, c2, pos = ['JJ','JJR','JJS'], verbose = True):
	'''
	modelpath <str> : path to skipgram w2v model
	topk <int> : topk words
	c1 list<str> : list of words for target set 1
	c2 list<str> : list of words for target set 2
	pos list<str> : List of parts of speech we are interested in analysing
	verbose <bool> : True/False
	'''

	def calculateCentroid(model, words):
		embeddings = [np.array(model[w]) for w in words if w in model]
		centroid = np.zeros(len(embeddings[0]))
		for e in embeddings:
			centroid += e
		return centroid/len(embeddings)

	def getCosineDistance(embedding1, embedding2):       
		return spatial.distance.cosine(embedding1, embedding2)


	#select the interesting subset of words based on pos
	model = Word2Vec.load(modelpath)
	words_sorted = sorted( [(k,v.index, v.count) for (k,v) in model.wv.vocab.items()] ,  key=lambda x: x[1], reverse=False)
	words = [w for w in words_sorted if nltk.pos_tag([w[0]])[0][1] in pos]

	if len(c1) < 1 or len(c2) < 1 or len(words) < 1:
		print('[!] Not enough word concepts to perform the experiment')
		return None

	centroid1, centroid2 = calculateCentroid(model, c1),calculateCentroid(model, c2)
	winfo = []
	for i, w in enumerate(words):
		word = w[0]
		freq = w[2]
		rank = w[1]
		pos = nltk.pos_tag([word])[0][1]
		wv = model[word]
		sent = sid.polarity_scores(word)['compound']
		#estimate cosinedistance diff
		d1 = getCosineDistance(centroid1, wv)
		d2 = getCosineDistance(centroid2, wv)
		bias = d2-d1

		winfo.append({'word':word, 'bias':bias, 'freq':freq, 'pos':pos, 'wv':wv, 'rank':rank, 'sent':sent} )

		if(i%100 == 0 and verbose == True):
			print('...'+str(i), end="")

	#Get max and min topk biased words...
	biasc1 = sorted( winfo, key=lambda x:x['bias'], reverse=True )[:min(len(winfo), topk)]
	biasc2 = sorted( winfo, key=lambda x:x['bias'], reverse=False )[:min(len(winfo), topk)]
	return [biasc1, biasc2]


def Cluster(biasc1, biasc2, r, repeatk, verbose = True):
	'''
	biasc1 list<words> : List of words biased towards target concept1 as returned by GetTopMostBiasedWords
	biasc2 list<words> : List of words biased towards target concept2 as returned by GetTopMostBiasedWords
	r <int> : reduction factor used to determine k for the kmeans; k = r * len(voc) 
	repeatk <int> : Number of Clustering to perform only to keep the partition with best intrasim
	'''
	def getCosineDistance(embedding1, embedding2): 
		return spatial.distance.cosine(embedding1, embedding2)
	def getIntraSim(partition):
		iS = 0
		for cluster in partition:
			iS += getIntraSimCluster(cluster)
		return iS/len(partition)
	def getIntraSimCluster(cluster):
		if(len(cluster)==1):
			return 0
		sim = 0; c = 0
		for i in range(len(cluster)):
			w1 = cluster[i]['wv']
			for j in range(i+1, len(cluster)):
				w2 = cluster[j]['wv']
				sim+= 1-getCosineDistance(w1,w2)
				c+=1
		return sim/c
	def createPartition(embeddings, biasw, k):
		preds = KMeans (n_clusters=k).fit_predict(embeddings)
		#first create the proper clusters, then estiamte avg intra sim
		all_clusters = []
		for i in range(0, k):
			clust = []
			indexes = np.where(preds == i)[0]
			for idx in indexes:
				clust.append(biasw[idx])
			all_clusters.append(clust)
		score = getIntraSim(all_clusters)
		return [score, all_clusters]


	k = int(r * (len(biasc1)+len(biasc2))/2)
	emb1, emb2  = [w['wv'] for w in biasc1], [w['wv'] for w in biasc2]
	mis1, mis2 = [0,[]], [0,[]]	#here we will save partitions with max sim for both target sets
	for run in range(repeatk):
		p1 = createPartition(emb1, biasc1, k)
		if(p1[0] > mis1[0]):
			mis1 = p1
		p2 = createPartition(emb2, biasc2, k)
		if(p2[0] > mis2[0]):
			mis2 = p2
		if(verbose == True):
			print('New partition for ts1, intrasim: ', p1[0])
			print('New partition for ts2, intrasim: ', p2[0])

	print('[*] Intrasim of best partition found for ts1, ', mis1[0])
	print('[*] Intrasim of best partition found for ts2, ', mis2[0])
	return [mis1[1], mis2[1]]
		
