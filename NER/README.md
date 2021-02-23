all final codes implemented in NumPy are in folder numpy

run python Features1.py to check the training log on loss and evalution results
Features1.py, Features2.py, Features3.py are for CRF
Features1HMM.py is for HMM
Features1StrP.py and Features3StrP.py are for Structured Perceptron

ValTestSplit.py is used to generate validation set

to check the result of lstm-crf, change directory to pytorch_lstmcrf folder and run the following command:
python trainer.py --device=cuda:0 --dataset=partial --model_folder=model_name --embedder_type=bert-base-cased

all .ipynb files, Features1NumpyLog.py, NumpyTest.py and Features1Numpy.py are for development

code in activation_map folder is another implementation which is formal, easier for generalization but slower for training and decoding
there are some bugs for the activation_map implementation currently
