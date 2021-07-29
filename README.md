<h1>Introduction</h1>
With this program, we explore a different way to train a classifier to try and increase its robustness
to adversarial attacks on the MNIST data base. The robustness will be tested on the Fast Gradient Sign Method using Foolbox(1).


<h1>Principle of the procedure</h1>
Instead of training a CNN on the data base directly, we take a few steps :


Step 1: We train an autoencoder on the data base

Step 2: We create a new data base out of the latent representations of the images (h_train)

Step 3: We make linear combinations of the latent vectors of the same class to generate new data (augmentation)

Step 4: We train a classifier on h_train and then on augmented h_train

Step 5: We concatenate the encoder of the autoencoder and the classifier to make a CNN


<h1>How to use the program</h1>
Use the git clone command to get the files or manually download them.

You can then lauch the program by executing 'main.py'. Here are the arguments:


--phase : Choose between 'train' or 'test'

--procedure : For training: 'True' will train a model using the procedure and 'False' will train a model without the procedure

--epoch : For training: Choose the number of epoch

--batch_size : For training: Choose the batch size

--nb_samples : For training: Choose the size of the training dataset

--name : For training: Choose the name of the model, if you put '', the name will be automatically generated

--epsilons : For testing: Choose list of epsilon to test robustness


<h1>Citation:</h1>
(1) Rauber et al., (2020). Foolbox Native: Fast adversarial attacks to benchmark the robustness of machine learning models in PyTorch, TensorFlow, 
and JAX. Journal of Open Source Software, 5(53), 2607, https://doi.org/10.21105/joss.02607
