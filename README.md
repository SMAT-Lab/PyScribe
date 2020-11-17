# PyScribe
Learning to Describe Python Code

Runtime Environment in our experiments:
10 NVIDIA 2080 Ti GPUs
CentOS 7.7
CUDA 10.1 (with CuDNN of the corresponding version)
Python 3.7
PyTorch 1.2.0


The dataset we built is in the direcotry "data/code_to_description_raw_data/"
To run the experiment, step into the directory "src_code/code_to_description/":
1. Run "python  preprocessor.py" for preprocessing.
2. Run "python code2dsc_model.py" to train and test the model.
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path "data/test_result/test_result.json", with ground truth and code involved for comparison.

Note that: all the parameters are st in "src_code/code_to_description/config.py"
