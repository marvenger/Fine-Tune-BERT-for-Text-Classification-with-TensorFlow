# Fine Tune BERT for Text Classification with TensorFlow
 Prediction of Insincerity of the Questions of the Quora




The problem statement here deals with the QUORA dataset. It is a platform where the queries are posted that are meant to be answered. But, there are some cases in which the posted text is not the query but the general statement based on a particular topic or thing. This type of text is referred to as insincere. The problem here is to predict the insincerity of the text posted.

Firstly, I installed all the dependencies and imported the required libraries.I used the ‘TensorflowHub’ library to deal with the problem. I set up the ‘tensorflow’ and the ‘collab’ runtime. I downloaded the required dataset and imported the same and began the analysis of the data. The data consists of the features ‘Ids of the text’, ‘Queries’ which consist of the texts and ‘Insincerity’ feature which is a binary variable for which 0 depicts the text being sincere and 1 depicts the text being insincere.

Then, I created the ‘tensorflow.Data.Datasets’ for training and evaluation and downloaded a pre-trained BERT Model from the ‘tensorflowHub’ library. The bert layer is defined using the layer present in the library and the tokenizer is also defined using the BERT tokenizer present in the library itself.

For the next step I tokenized and preprocessed the text for the BERT as it is required to transform the text into the inputs the BERT layer understands. These inputs are represented with the three sets of features which are ‘Token Ids’, ‘Input Mask’ and the ‘input Mask Ids’ or the ‘Segment Ids’.
It is done using the two steps. First step is to create InputExamples using classifier_data_lib's constructor InputExample provided in the BERT library. 
It allows the creation of a function that converts raw input features into the desired inputs.

Next step is to map the feature to each element of the dataset. It can be done by ‘Dataset.Map’, but Dataset.map runs in graph mode.
Graph tensors do not have a value.
In graph mode we can only use TensorFlow Ops and functions 
So, the function cannot be mapped directly. Instead of this I wrapped a Python function into a Tensorflow op for Eager Execution.
Then, I created a Tensorflow input pipeline with tensorflow.data where I defined the training and validation data.
Next, I built the model, I defined the three input layers the BERT layer understands and pooled these inputs into the BERT Layer using the BERT layer defined earlier. Then, I added a dropout layer to the BERT layer to prevent overfitting and finally added a classification layer with sigmoid as the activation function.

Then, I fine tuned the model after compilation using the ‘ADAM’ Optimizer and loss as Binary CrossEntropy and the metrics as the BInary Accuracy.
Finally, I evaluated the model after fitting it.


