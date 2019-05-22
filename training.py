import pandas as pd
import json

def summarize_data(train_data, features, label):
	neg_x = train_data[train_data[label]==0][features]
	pos_x = train_data[train_data[label]==1][features]
	mean_neg_x = dict(neg_x.mean(axis=0))
	stdev_neg_x = dict(neg_x.std(axis=0))
	mean_pos_x = dict(pos_x.mean(axis=0))
	stdev_pos_x = dict(pos_x.std(axis=0))
	summary = {}
	summary[0] = {}
	summary[1] = {}
	for feat in features:
		summary[0][feat] = [mean_neg_x[feat], stdev_neg_x[feat]]
		summary[1][feat] = [mean_pos_x[feat], stdev_pos_x[feat]]
	return summary	

def training(input_file, output_file, label):
	train_data = pd.read_csv(input_file)
	train_data.set_index('Date', inplace=True)
	features = list(train_data.select_dtypes(include='float64'))
	summary = summarize_data(train_data, features, label)
	# print(summary)
	with open(output_file, 'w+') as out_file:
		json.dump(summary, out_file)

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_file', default='train-data.csv', dest='input_file', help="""input file for training [default : %(default)s]""")
	parser.add_argument('-o', '--output_file', default='summary.json', dest='output_file', help="""output summary file [default : %(default)s]""")
	parser.add_argument('-l', '--target_label', default='RainToday', dest='label', help="""target label [default : %(default)s]""")
	args = parser.parse_args()
	training(args.input_file, args.output_file, args.label)
