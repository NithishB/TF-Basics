# Seq2Seq_Model class
The python file has a class named Seq2Seq_Model. It takes params as a dictionary or parameters to contruct a seq2seq model and return the graph object. <br>

An example run is shown in the Seq2Seq_final notebook. The same code can be used to create both training and inference graphs. The user must use saving and loading variables in graphs to use the inference based on the trained model. Inference can use Greedy and Beam search decoding mechanisms and attention mechanisms can be Bahdanau, Luong or their normalized and scaled versions.
