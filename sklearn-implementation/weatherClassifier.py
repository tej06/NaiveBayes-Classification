import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def data_process(input_file, label):
	df = pd.read_csv(input_file)
	df.set_index('Date', inplace=True)
	df.fillna(method='ffill', inplace=True)
	df.RainToday=df.RainToday.map({'Yes':1, 'No':0})
	df.RainTomorrow=df.RainTomorrow.map({'Yes':1, 'No':0})
	df.dropna(axis=1, how='any', inplace=True)
	features = list(df.select_dtypes(include='float64'))
	target = df[label]
	feats = df[features].values
	y_true = target.values
	y_true = y_true.reshape((len(target), 1))
	return feats, y_true

def getAccuracy(y_true, y_pred):
	return accuracy_score(y_true, y_pred)*100.0

def classifier(input_file, label):
	feats, y_true = data_process(input_file, label)
	gnb = GaussianNB()
	y_pred = gnb.fit(feats, y_true).predict(feats)
	y_pred = y_pred.reshape((y_true.shape[0],1))
	accuracy = getAccuracy(y_true, y_pred)
	return accuracy

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_file', default=None, dest='input_file', help="""input file for data preprocessing [default : %(default)s]""")
	parser.add_argument('-l', '--target_label', default='RainToday', dest='label', help="""target label [default : %(default)s]""")
	args = parser.parse_args()
	if args.input_file is None:
		print("Error : Please provide an input file")
		parser.print_help()
	else:
		accuracy = classifier(args.input_file, args.label)
		print("Accuracy:",accuracy)
