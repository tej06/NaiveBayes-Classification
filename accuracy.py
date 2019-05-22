import json
import predict
import pandas as pd

def getClass(summary, test_dict):
	bestClass,_ = predict.classifier(summary, test_dict)
	return int(bestClass)

def getAccuracy(summary, test, label):
	features = list(test.select_dtypes(include='float64'))
	features.append(label)
	test_data = test[features].to_dict('list')
	correct_count = 0
	for i in range(len(test)):
		test_dict = {}
		for feats in features:
			if not feats==label:
				test_dict[feats] = test_data[feats][i]
		if getClass(summary, test_dict) == test_data[label][i]:
			correct_count += 1
	acc = (correct_count/float(len(test)))*100.0
	return acc

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--summary_file', default='summary.json', dest='summary_file', help="""summary file [default : %(default)s]""")
	parser.add_argument('-t', '--test_csv', default='test-data.csv', dest='test_file', help="""test file in csv [default : %(default)s]""")
	parser.add_argument('-l', '--target_label', default='RainToday', dest='label', help="""target label [default : %(default)s]""")
	args = parser.parse_args()
	summary = predict.load_data(args.summary_file)
	test = pd.read_csv(args.test_file)
	test.set_index('Date', inplace=True)
	acc = getAccuracy(summary, test, args.label)
	print("Accuracy:", acc)
