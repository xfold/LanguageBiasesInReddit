import DADDBias_ICWSM

print('''
	*************************************************************************
	Test run using a toy dataset of 1000 comments collected from r/TheRedPill
	*************************************************************************
	''')

#config
csvpath = 'Datasets/toy_1000_trp.csv'
outputpath = 'Models/toy_trp_model'


'''
Train new model
'''
print('Training new model', csvpath)
DADDBias_ICWSM.TrainModel(csvpath, 
           'body',
           outputname = outputpath,
           window = 4,
           minf = 10,
           epochs = 5,
           ndim = 200,
           verbose = False
           )
print('Training finished, saved ', outputpath)


'''
Find biased words
'''
print()
print('Finding biases...')
female=["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
male=["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"]  
[b1,b2] = DADDBias_ICWSM.GetTopMostBiasedWords(
			outputpath,			#model path
            50,					#topk biased words
            female,				#target set 1
            male,				#target set 2
            ['JJ','JJR','JJS'], #nltk pos to be considered
            verbose = False)		

print('Biased words towards ', female)
print([b['word'] for b in b1])

print('Biased words towards ', male)
print([b['word'] for b in b2])


'''
cluster words
'''
print()
print('Clustering words into concepts...')
[cl1,cl2] = DADDBias_ICWSM.Cluster(
	b1,			#set of words biased towards target set 1
	b2, 		#set of words biased towards target set 2
	0.15, 		#r 
	10,			#repeat
	verbose = False)	

print('Resulting clusters')
print('Clusters biased towards ', female)
for cluster in cl1:
    print( [k['word'] for k in cluster] )

print('Clusters biased towards ', male)
for cluster in cl2:
    print( [k['word'] for k in cluster] )

print()
print('*Finished')