# PYSCRIBE
In this work, we propose PyScribe, which expands the **Transformer** model to a new encoder-decoder architecture to incorporate **triplet positions** into the **node-level** and **edge-level features** of AST towards automatically generating comments for Python code. In the decoder, we perform a **two-stage decoding process** over the extracted node and edge features to generate comments.

# Runtime Environment
Runtime Environment in our experiments:
- 4 NVIDIA 2080 Ti GPUs
- Ubuntu 16.04
- CUDA 10.0 (with CuDNN of the corresponding version)
- Python 3.7
- PyTorch 1.2.0

# Dataset
The dataset we built is in the direcotry `data/code_to_description_raw_data/`

# Run the experiment
1. step into the directory `src_code/code_to_description/`:
   ```angular2html
   cd src_code/code_to_description/
    ```
2. Proprecess the train/valid/test data:
    ```angular2html
    python  preprocessor.py
    ```
3. Run the model for training and testing:
   ```angular2html
   python code2dsc_model.py
    ```
After running, the performance will be printed in the console, and the predicted results of test data and will be saved in the path `data/test_result/test_result.json`, with ground truth and code involved for comparison.

**Note that**: all the parameters are set in `src_code/code_to_description/config.py`
If the model has been trained, you can set the parameter "train" in line 92 in `config.py` to "False". Then you can predict the test data directly by using the model saved in `data/model/`.

Besides, we provide a **supplementary experiment** (click **[HERE](https://github.com/GJCEXP/PYSCRIBE_supplementary)**) on two other datasets including a Python dataset[1] and a Java dataset[2].

<font size=2>[1] Wan, Y., Zhao, Z., Yang, M., Xu, G., Ying, H., Wu, J., Yu, P.S.: Improving automatic source code summarization via deep reinforcement learning. In: Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering, pp. 397–407 (2018).

[2] Hu, X., Li, G., Xia, X., Lo, D., Lu, S., Jin, Z.: Summarizing source code with transferred api knowledge. IJCAI’18, pp. 2269–2275 (2018).</font>

***This paper is still under review, please do not distribute it.***
