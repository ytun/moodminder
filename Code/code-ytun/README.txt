ReadMe for MoodMinder Classification
Author: Yamin Tun

To test the main scripts for 2 experiments mentioned below:

-Download Sentiment140 dataset at:
http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
-Rename training.1600000.processed.noemoticon.csv into train.csv
-Copy train.csv into stanford_data folder inside our code_evaluation folder



==Modules==
Classifier.py- Classifier class 
Feat_extractor.py- Feature Extractor class
Util_data.py- Module with useful functions to load, split and sample dataset
Util_plot.py- Module with useful functions for plotting bar graphs and saving classification reports

==Folders==

-RESULTS: Store results from experiment 2 in one folder for each classifier

-stanford_data
	test.txt- dummy short test corpus
	train.csv- full stanford training corpus
	**(See the top instruction)

-main: Scripts for 2 experiments

	main_exp1.py
	Experiment 1: Cross-Validation for NB and SVC with unigrams only and unigram		+bigram

	main_exp2.py
	Experiment 2: comparison among svc and nb in terms of training time, precision, 	accuracy, recall

-unitTests: 

	Test_Feat_extractor.py- Unit test for feature extractor
	Test_Util_data.py- Unit test for Util_data.py