if __name__ == '__main__':
	# import required libraries
	import h5py as h5
	import numpy as np
	import matplotlib.pyplot as plt
	import pandas as pd

	#filename = "/mnt/c/Users/Ebru-as-User/Downloads/combined_test.h5"
	filename = "/mnt/c/Users/Ebru-as-User/Downloads/pacific.pacific_9mers.01.h5"
	# Read H5 file
	f = h5.File(filename, "r")
	#n1 = np.array(f['sequence'])
	#n1 = f.get('input_ids')
	#n1 = np.array(n1[0:100])
	#print(n1[0:100]) 
	#df = pd.DataFrame(n1)
	#df.head()
	#print(df)
	# Get and print list of datasets within the H5 file
	datasetNames = [n for n in f.keys()]
	for n in datasetNames:
			print(n)
	def print_name(name, obj):
		if isinstance(obj, h5.Dataset):
			print('Dataset:', name)
		elif isinstance(obj, h5.Group):
			print('Group:', name)

	#with h5.File(filename, 'r') as h5f:  # file will be closed when we exit from WITH scope
    #    h5f.visititems(print_name)

    #with h5.File(filename, 'r') as h5f:  # file will be closed when we exit from WITH scope
	#	sequences = h5f['sequence'].value
	#	input_ids = h5f['input_id'].value
