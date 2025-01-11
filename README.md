# Logistic Regression

Logistic regression is a statistical method used for binary classification, which means it's used to predict the probability of a binary outcome (e.g., yes/no, true/false, 1/0).it is a classification algorithm .The term "logistic" refers to the logistic function(or sigmoid function) that is used to model the relationship between the independent variables (predictors) and the dependent variable (response or outcome). The logistic function, also known as the sigmoid function, is an S-shaped curve that maps any real-valued number to a value between 0 and 1.

The logistic function is defined as:

![sigmoid function](https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/3d84a670-2fae-45b7-9328-4349bdf3df74)



where z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ


Logistic regression is  a linear model. The linearity in logistic regression refers to the fact that the decision boundary separating the classes (e.g., positive and negative classes) is a linear function of the input features.

The logistic regression model makes predictions by applying a linear transformation to the input features and then passing the result through a logistic (sigmoid) function to obtain the probability of belonging to a particular class. Mathematically, the logistic regression model can be represented as:

$$
P(y = 1 | \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n)}}
$$

- `P(y = 1 | x)` represents the probability of the outcome variable `y` being 1 given the input features `x`.
- `β₀, β₁, β₂, ..., βₙ` are the coefficients (parameters) of the model.
- `x₁, x₂, ..., xₙ` are the input features.


Logistic regression assumes a linear relationship between the input features and the log-odds of the outcome variable being in a particular category.

## Some basic Geometry 

1. Every line has a positive side and a negative side.
2. Put the point in the equation of line and if the resultant is greater than 0 then it lies on the positive side and if it is less than 0 then it is negative .

 <img src="https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/2c54c967-bbd1-4091-806e-1d620f638479" alt="WhatsApp Image 2024-03-18 at 11 54 58 AM 1" height="400px" width="400"><br>





Now, let's talk about the perceptron trick:

1. **Initialize coefficients**: First, we start with some initial values for the coefficients (often set to 0 or randomly).

2. **Iterate through training data**: We go through each training example one by one.

3. **Compute prediction**: We use the current coefficients to predict the outcome for the current training example.

4. **Update coefficients**: If the prediction is incorrect, we update the coefficients using the perceptron trick.

    - If the prediction is 1 (positive class) but the actual label is 0, we decrease the coefficients.
    - If the prediction is 0 (negative class) but the actual label is 1, we increase the coefficients.
5. **Repeat**: We repeat steps 3 and 4 for each training example, adjusting the coefficients after each prediction.


The perceptron trick essentially adjusts the coefficients based on whether the prediction was correct or not. If the prediction was wrong, it moves the decision boundary (defined by the coefficients) closer to the correct classification. This process continues until the algorithm converges to a set of coefficients that minimize the cost function.

lets write the algorithm now:

W is the weight vector 
X is the input vector 

<img src="https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/254b9114-b858-409a-aea9-f6848bd78e10" height="450px" width="450">

<img src="https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/c8964bbe-33c0-4c95-b5a0-1507aeaa7471" height="450px" width="450"><br>




Now, even though the perceptron trick will work, it has a few limitations. It stops once the classes in the training dataset are classified, which may cause the model to perform poorly on real data. The likelihood function is the product of the predicted probabilities for the actual class of each observation. So, the higher the likelihood, the better the logistic regression model. However, when multiplying the probabilities of each observation, we might encounter a numerical underflow problem, as computers might not be able to accurately represent it. Therefore, we introduce the logarithm to this likelihood function to convert the product into a sum. However, the logarithm of values between 0 and 1 is negative, so we must apply one extra negative to make the result positive. Since we have applied the negative, we then need to minimize the function for a better model instead of maximizing it. That's how we derive log loss
## Log Loss

Log loss, also known as logistic loss or cross-entropy loss, is a measure used to evaluate the performance of a classification model. It quantifies the accuracy of the model by penalizing incorrect classifications.

In binary classification problems, where there are only two classes (e.g., positive and negative), log loss is defined as:

$$
\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right)
$$

Explanation:
- `N` is the number of observations in the dataset.
- `y_i` is the actual label for the `i`th observation (0 or 1).
- `p_i` is the predicted probability that the `i`th observation belongs to the class represented by `y_i`.
  - If `y_i = 1`, `p_i` is the predicted probability of the positive class.
  - If `y_i = 0`, `p_i` is the predicted probability of the negative class.

$$
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n
$$

<br>Now our goal is to find the values of $$\beta_0 ,\beta_1,\beta_2$$ and so on such that log loss is minimized . The log loss function doesn't have any closed form solution . so we will use Gradient descent to minimize the log loss.

<img src="https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/c3e9a85d-b7ed-4784-b650-dfbb4675f508" height="550px" width="450">

<img src="https://github.com/NeuralNoble/Logistic-Regression/assets/156664113/d74c7197-9931-42f5-a4d3-7f6a2837bbab" height="550px" width="450">




