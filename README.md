
Assignment work completed for McGill's COMP 551 Applied Machine Learning course.
## Setup Instructions
- 1. Create a python3.10 virtual environment and activate it
`python3.10 -m venv .`
`source ./bin/activate`
- 2. Install the packages
`pip install -r requirements.txt`

## Models and Topics Covered

### Assignment 2: Linear Classification Models

- **Linear Classification**: Explored binary and multi-class classification using logistic regression, decision trees, and K Nearest Neighbors.
- **Feature Selection**: Investigated the impact of selecting features based on their linear coefficients on model performance.
- **Custom Implementation**: Developed custom logistic regression and multi-class regression models from scratch.

### Assignment 3: Implementing a Multilayer Perceptron (MLP)

- **Neural Network Basics**: Implemented a basic MLP from scratch, including the design of its architecture and the application of backpropagation and gradient descent algorithms.
- **Convolutional Neural Networks (ConvNets)**: Experimented with ConvNets to understand their structure and performance on image data classification.
- **Model Optimization**: Explored various network configurations and hyperparameter tunings to optimize the models' performance.

## Tools Used

- **Programming Language**: Python was used as the primary programming language for all assignments due to its extensive support for data analysis and machine learning libraries.
- **Libraries**: 
  - **Numpy**: Essential for implementing custom models and performing numerical computations.
  - **Scikit-learn**: Utilized for data preprocessing, feature extraction, and leveraging built-in machine learning algorithms for comparison.
  - **Keras/TensorFlow and PyTorch**: Employed for creating ConvNet models and experimenting with deep learning techniques.

## Datasets Used

- **The Large Movie Review Dataset**: A collection of movie reviews for binary sentiment classification tasks, used in Assignment 2.
- **The 20 Newsgroups Dataset**: Comprises roughly 20,000 newsgroup documents, utilized for multi-class text classification, also in Assignment 2.
- **Sign Language MNIST**: An image dataset used in Assignment 3 for classifying sign language digits, demonstrating the application of MLPs and ConvNets to image data.

### Setup Instructions ###
Run the following to have the jupyter notebook outputs cleared
`python3 -m pip install nbstripout`
`python3 -m nbstripout --install --global`
If it doesn't work just make sure you always clear outputs before committing
