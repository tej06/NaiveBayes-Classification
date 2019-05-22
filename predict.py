import json
import math

def load_data(file):
	with open(file) as json_file:
		data = json.load(json_file)
	return data

def calcGaussianProb(params, x):
	prob_exp = math.exp(-(math.pow(x-params[0], 2)/(2*math.pow(params[1],2))))
	probability = (1.0/(math.sqrt(2*math.pi)*params[1]))*prob_exp
	return probability

def normalize(probabilities):
	total = sum(probabilities.values())
	probabilities['0'] /= total
	probabilities['1'] /= total
	return probabilities

def classProbabilities(summary, inputDict):
	probabilities = {}
	for group, parameters in summary.items():
		if group not in probabilities:
			probabilities[group] = 1
		for key, params in parameters.items():
			probabilities[group] *= calcGaussianProb(params, inputDict[key])
	# print(probabilities)
	return normalize(probabilities)

def classifier(summary, inputDict):
	probabilities = classProbabilities(summary, inputDict)
	bestClass, bestProb = None, -1
	for group, prob in probabilities.items():
		if bestClass is None or prob > bestProb:
			bestClass = group
			bestProb = prob
	return bestClass, bestProb

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--summary_file', default='summary.json', dest='summary_file', help="""summary file [default : %(default)s]""")
	parser.add_argument('-i', '--input_dictionary', default=None, dest='input_dict', help="""input file with dictionary to predict [default : %(default)s]""")
	args = parser.parse_args()
	if args.input_dict is None:
		print("Error : Please provide an input file for prediction")
		parser.print_help()
	else:
		summary = load_data(args.summary_file)
		inputDict = load_data(args.input_dict)
		bestClass, bestProb = classifier(summary, inputDict)
		print("Class: {0}\tProbability: {1}".format(bestClass, bestProb))
