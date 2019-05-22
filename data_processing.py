import pandas as pd

def preprocess(input_file, output_file, test_file, split):
	df = pd.read_csv(input_file)
	df.set_index('Date', inplace=True)
	df.fillna(method='ffill', inplace=True)
	df.RainToday=df.RainToday.map({'Yes':1, 'No':0})
	df.RainTomorrow=df.RainTomorrow.map({'Yes':1, 'No':0})
	df.dropna(axis=1, how='any', inplace=True)
	split_count = int(split*len(df)/100.0)
	train = df[:split_count]
	test = df[split_count:]
	train.to_csv(output_file)
	test.to_csv(test_file)

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_file', default=None, dest='input_file', help="""input file for data preprocessing [default : %(default)s]""")
	parser.add_argument('-o', '--output_file', default='train-data.csv', dest='output_file', help="""processed csv file [default : %(default)s]""")
	parser.add_argument('-s', '--split_train_test', default=80.0, dest='split', help="""train : test split ratio [default : %(default)s]""")
	parser.add_argument('-t', '--test_file', default='test-data.csv', dest='test_file', help="""test csv file [default : %(default)s]""")
	args = parser.parse_args()
	if args.input_file is None:
		print("Error : Please provide an input file")
		parser.print_help()
	else:
		preprocess(args.input_file, args.output_file, args.test_file, float(args.split))
