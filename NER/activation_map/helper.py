def list2dict(lst):
	output = dict()
	for i, e in enumerate(lst):
		output[e] = i
	return output

def dictReverse(dic):
	output = dict()
	for key, value in dic.items():
		output[value] = key
	return output