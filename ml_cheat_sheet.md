# Machine Learning Cheat Sheet 📚

*A comprehensive revision guide and interview preparation resource*

---

## 📋 Table of Contents
1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Model Evaluation](#model-evaluation)
4. [Regression Models](#regression-models)
5. [Regularization](#regularization)
6. [Dimensionality Reduction](#dimensionality-reduction)
7. [Decision Trees & Ensemble Methods](#decision-trees--ensemble-methods)
8. [Common Interview Questions](#common-interview-questions)

---

## 🤖 Introduction to Machine Learning

### Key Definitions
- **Machine Learning**: A subset of AI where algorithms learn patterns from data without explicit programming
- **Algorithm**: A set of rules/instructions for solving problems
- **Model**: The output of an algorithm trained on data
- **Training**: Process of teaching the algorithm using historical data
- **Prediction**: Using the trained model to make forecasts on new data

### Types of Learning

#### Supervised Learning
- **Definition**: Learning from labeled data (input-output pairs)
- **Goal**: Learn a mapping function from input to output
- **Types**:
  - **Classification**: Predicting categories/classes (discrete output)
    - *Examples*: Email spam detection, image recognition, medical diagnosis
  - **Regression**: Predicting continuous numerical values
    - *Examples*: House price prediction, stock price forecasting, temperature prediction

#### Unsupervised Learning
- **Definition**: Finding hidden patterns in data without labels
- **Types**:
  - **Clustering**: Grouping similar data points
    - *Examples*: Customer segmentation, gene sequencing, market research
  - **Association**: Finding relationships between variables
  - **Dimensionality Reduction**: Reducing number of features while preserving information

### Training Process Components
- **Dataset**: Collection of examples used for training
- **Features**: Input variables (independent variables)
- **Labels**: Output variables (dependent variables) - only in supervised learning
- **Loss Function**: Measures how wrong the model's predictions are
- **Training Algorithm**: Method to minimize the loss function

---

## 🧮 Mathematical Foundations

### Linear Algebra Essentials

#### Vectors
```
Vector: [x₁, x₂, ..., xₙ]
```
- **Dot Product**: a·b = Σ(aᵢ × bᵢ)
- **Vector Norm**: ||v|| = √(Σvᵢ²) (L2 norm)
- **Unit Vector**: Vector with magnitude 1

#### Matrices
```
Matrix A (m×n):
[a₁₁  a₁₂  ...  a₁ₙ]
[a₂₁  a₂₂  ...  a₂ₙ]
[...  ...  ...  ...]
[aₘ₁  aₘ₂  ...  aₘₙ]
```

**Key Operations**:
- **Addition**: Element-wise addition
- **Multiplication**: (A×B)ᵢⱼ = Σ(Aᵢₖ × Bₖⱼ)
- **Transpose**: Aᵀ (flip rows and columns)
- **Inverse**: A⁻¹ (exists only for square, non-singular matrices)
- **Determinant**: Scalar value representing matrix properties

**Important Concepts**:
- **Eigenvalues (λ)**: Av = λv (scaling factor)
- **Eigenvectors (v)**: Direction that doesn't change under transformation
- **SVD**: A = UΣVᵀ (factorization into orthogonal matrices)
- **Covariance Matrix**: Measures how variables vary together

### Calculus & Optimization

#### Derivatives
- **Definition**: Rate of change of a function
- **Partial Derivative**: ∂f/∂x (derivative with respect to one variable)
- **Gradient**: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ] (vector of all partial derivatives)
- **Chain Rule**: d/dx[f(g(x))] = f'(g(x)) × g'(x)

#### Optimization
- **Gradient Descent**: x_{new} = x_{old} - α∇f(x)
  - α = learning rate
  - Move in opposite direction of gradient (steepest descent)
- **Convex Function**: Has single global minimum (bowl-shaped)
- **Non-Convex Function**: Multiple local minima (challenging to optimize)
- **Learning Rate**: Step size in optimization
  - Too high: May overshoot minimum
  - Too low: Slow convergence

### Probability & Statistics

#### Probability Basics
- **Sample Space**: Set of all possible outcomes
- **Event**: Subset of sample space
- **P(A)**: Probability of event A (0 ≤ P(A) ≤ 1)
- **Conditional Probability**: P(A|B) = P(A∩B)/P(B)
- **Independence**: P(A∩B) = P(A)×P(B)

#### Distributions
- **Normal Distribution**: N(μ, σ²)
  - Bell-shaped curve
  - Parameters: mean (μ), variance (σ²)
  - 68-95-99.7 rule

#### Key Concepts
- **Maximum Likelihood Estimation (MLE)**: Find parameters that maximize likelihood of observed data
- **Entropy**: H(X) = -Σ P(x)log₂P(x) (measure of uncertainty)
- **Cross-Entropy**: Measures difference between two probability distributions
- **Correlation**: Linear relationship strength (-1 to 1)
- **Covariance**: How two variables change together

---

## 📊 Model Evaluation

### Train-Test Split Issues
- **Data Leakage**: Information from future leaks into training
- **Temporal Dependency**: Time-series data requires chronological splits
- **Distribution Shift**: Training and test data from different distributions

### Cross-Validation Techniques

#### K-Fold Cross-Validation
```
Data split into K folds
For each fold:
  - Use as test set
  - Train on remaining K-1 folds
  - Calculate performance
Average performance across all folds
```

#### Stratified K-Fold
- Maintains class distribution in each fold
- Important for imbalanced datasets

#### Time Series CV
- Respect temporal order
- Use expanding or sliding window approach

### Classification Metrics

#### Confusion Matrix
```
                Predicted
              Pos    Neg
Actual  Pos   TP    FN
        Neg   FP    TN
```

#### Performance Metrics
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - "Of predicted positives, how many are correct?"
- **Recall (Sensitivity)**: TP / (TP + FN) - "Of actual positives, how many did we catch?"
- **Specificity**: TN / (TN + FP) - "Of actual negatives, how many did we correctly identify?"
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

#### ROC Curve & AUC
- **ROC**: Receiver Operating Characteristic (True Positive Rate vs False Positive Rate)
- **AUC**: Area Under Curve (0.5 = random, 1.0 = perfect)
- **Interpretation**: Probability that model ranks random positive higher than random negative

### Regression Metrics
- **MSE**: Mean Squared Error = Σ(yᵢ - ŷᵢ)²/n
- **RMSE**: Root Mean Squared Error = √MSE
- **MAE**: Mean Absolute Error = Σ|yᵢ - ŷᵢ|/n
- **R²**: Coefficient of Determination = 1 - (SSres/SStot)

### Bias-Variance Tradeoff
- **Bias**: Error from oversimplifying assumptions
  - High bias → Underfitting
- **Variance**: Error from sensitivity to training data fluctuations
  - High variance → Overfitting
- **Total Error**: Bias² + Variance + Irreducible Error
- **Sweet Spot**: Balance between bias and variance

---

## 📈 Regression Models

### Linear Regression

#### Mathematical Formulation
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
- **β₀**: Intercept
- **βᵢ**: Coefficients (slopes)
- **ε**: Error term

#### Cost Function (MSE)
```
J(β) = (1/2m) Σ(hβ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

#### Normal Equation Solution
```
β = (XᵀX)⁻¹Xᵀy
```

#### Key Properties
- **Assumptions**:
  - Linear relationship
  - Independence of errors
  - Homoscedasticity (constant variance)
  - Normal distribution of errors
- **Convex Optimization**: Single global minimum
- **R-Squared Interpretation**: Proportion of variance explained

### Logistic Regression

#### Sigmoid Function
```
σ(z) = 1 / (1 + e⁻ᶻ)
where z = β₀ + β₁x₁ + ... + βₙxₙ
```

#### Properties
- **Output Range**: (0, 1) - represents probabilities
- **Decision Boundary**: P(y=1) = 0.5 when z = 0
- **Log-Odds**: ln(p/(1-p)) = z (linear relationship)

#### Cost Function (Cross-Entropy)
```
J(β) = -(1/m) Σ[y⁽ⁱ⁾log(hβ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-hβ(x⁽ⁱ⁾))]
```

#### Maximum Likelihood Estimation
- **Goal**: Find parameters that maximize likelihood of observed data
- **Why Log-Likelihood**: Converts multiplication to addition, easier to optimize
- **Convex Optimization**: Guaranteed global minimum

---

## 🎯 Regularization

### The Problem
- **Overfitting**: Model learns training data too well, poor generalization
- **High Variance**: Model sensitive to small changes in training data
- **Complex Models**: Too many parameters relative to data

### Regularization Techniques

#### Ridge Regression (L2)
```
Cost = MSE + α Σβᵢ²
```
- **Effect**: Shrinks coefficients toward zero
- **Properties**: Keeps all features, reduces their impact
- **Best for**: When all features are somewhat relevant

#### Lasso Regression (L1)
```
Cost = MSE + α Σ|βᵢ|
```
- **Effect**: Can set coefficients exactly to zero
- **Properties**: Automatic feature selection
- **Best for**: When only subset of features are relevant

#### Elastic Net
```
Cost = MSE + α₁Σ|βᵢ| + α₂Σβᵢ²
```
- **Combines**: L1 and L2 penalties
- **Benefits**: Feature selection + coefficient shrinkage
- **Best for**: High-dimensional data with grouped features

### Hyperparameter α (Lambda)
- **α = 0**: No regularization (standard regression)
- **α → ∞**: Maximum regularization (coefficients → 0)
- **Selection**: Use cross-validation to find optimal value

---

## 📉 Dimensionality Reduction

### Principal Component Analysis (PCA)

#### Purpose
- **Reduce Dimensions**: Keep most important information
- **Remove Redundancy**: Eliminate correlated features
- **Visualization**: Project high-D data to 2D/3D
- **Noise Reduction**: Focus on signal, reduce noise

#### Mathematical Process
1. **Standardize Data**: Zero mean, unit variance
2. **Covariance Matrix**: C = (1/n)XᵀX
3. **Eigendecomposition**: C = PΛPᵀ
4. **Select Components**: Choose top k eigenvectors
5. **Transform Data**: Y = XP_k

#### Key Concepts
- **Principal Components**: Eigenvectors of covariance matrix
- **Explained Variance**: Eigenvalues show information content
- **Cumulative Explained Variance**: Choose k to retain 95% of variance
- **Loading Scores**: How much each original feature contributes to PC

#### Geometric Intuition
- **First PC**: Direction of maximum variance
- **Second PC**: Perpendicular to first, next highest variance
- **Orthogonal**: All PCs are perpendicular to each other

#### When to Use PCA
- **High-dimensional data** with multicollinearity
- **Preprocessing** before other algorithms
- **Visualization** of complex datasets
- **Storage/Speed** optimization

---

## 🌲 Decision Trees & Ensemble Methods

### Decision Trees

#### How They Work
- **Recursive Splitting**: Divide data based on feature values
- **Greedy Algorithm**: Choose best split at each node
- **Tree Structure**: Root → Internal Nodes → Leaves

#### Splitting Criteria

##### Gini Impurity
```
Gini = 1 - Σpᵢ²
```
- **Range**: 0 (pure) to 0.5 (maximum impurity for binary)
- **Interpretation**: Probability of misclassifying randomly chosen element

##### Entropy
```
Entropy = -Σpᵢ log₂(pᵢ)
```
- **Range**: 0 (pure) to log₂(classes) (maximum uncertainty)
- **Interpretation**: Amount of information needed to classify

##### Information Gain
```
IG = Entropy(parent) - Σ(|child|/|parent|) × Entropy(child)
```

#### Advantages
- **Interpretable**: Easy to understand and visualize
- **No Assumptions**: Non-parametric, handles non-linear relationships
- **Feature Selection**: Automatically selects relevant features
- **Mixed Data Types**: Handles numerical and categorical features

#### Disadvantages
- **Overfitting**: Can create overly complex trees
- **Instability**: Small data changes can create different trees
- **Bias**: Favors features with more levels

### Random Forest

#### Concept
- **Ensemble**: Combination of multiple decision trees
- **Bagging**: Bootstrap Aggregating - train on random subsets
- **Feature Randomness**: Consider random subset of features at each split
- **Voting**: Majority vote (classification) or average (regression)

#### Algorithm
1. **Bootstrap Sampling**: Create B bootstrap samples
2. **Train Trees**: Build tree on each sample with feature randomness
3. **Aggregate**: Combine predictions from all trees

#### Advantages
- **Reduces Overfitting**: Averaging reduces variance
- **Robust**: Less sensitive to outliers and noise
- **Feature Importance**: Ranks feature importance
- **Parallel**: Trees can be trained independently

#### Hyperparameters
- **n_estimators**: Number of trees
- **max_features**: Features considered at each split
- **max_depth**: Maximum tree depth
- **min_samples_split**: Minimum samples to split node

### Gradient Boosting

#### Concept
- **Sequential Learning**: Trees learn from previous trees' mistakes
- **Additive Model**: F(x) = f₁(x) + f₂(x) + ... + fₘ(x)
- **Gradient Descent**: Optimize loss function iteratively

#### Algorithm
1. **Initialize**: Start with simple prediction (mean)
2. **Calculate Residuals**: Errors from current model
3. **Train Tree**: Fit tree to residuals
4. **Update Model**: Add tree with learning rate
5. **Repeat**: Until convergence or max iterations

#### Key Concepts
- **Weak Learners**: Simple models (shallow trees)
- **Learning Rate**: Controls contribution of each tree
- **Regularization**: Prevent overfitting through tree constraints

#### Popular Implementations
- **XGBoost**: Extreme Gradient Boosting
- **LightGBM**: Light Gradient Boosting Machine
- **CatBoost**: Categorical Boosting

---

## 🎤 Common Interview Questions

### Conceptual Questions

**Q: Explain the bias-variance tradeoff.**
A: Bias is error from oversimplifying assumptions (underfitting), variance is error from sensitivity to training data (overfitting). Total error = bias² + variance + noise. Need to balance both for optimal performance.

**Q: When would you use logistic regression vs. decision trees?**
A: Logistic regression for linear relationships, interpretable coefficients, and probabilistic outputs. Decision trees for non-linear relationships, mixed data types, and when interpretability is crucial.

**Q: How do you handle overfitting?**
A: Regularization (L1/L2), cross-validation, more training data, feature selection, simpler models, early stopping, ensemble methods.

**Q: Explain PCA in simple terms.**
A: PCA finds the directions (principal components) along which data varies the most. It's like finding the best angle to photograph a 3D object in 2D while preserving maximum information.

**Q: What's the difference between bagging and boosting?**
A: Bagging trains models independently in parallel and averages results (reduces variance). Boosting trains models sequentially, each learning from previous mistakes (reduces bias).

### Technical Questions

**Q: How does gradient descent work?**
A: Iteratively moves in the direction of steepest descent (negative gradient) to minimize cost function. Update rule: θ = θ - α∇J(θ).

**Q: What are eigenvalues and eigenvectors?**
A: For matrix A, eigenvector v satisfies Av = λv, where λ is eigenvalue. Eigenvector's direction doesn't change under transformation, only scaled by eigenvalue.

**Q: Why use cross-entropy loss for classification?**
A: Penalizes confident wrong predictions heavily, provides smooth gradients for optimization, and corresponds to maximum likelihood estimation for probabilistic models.

**Q: How do you choose the number of principal components?**
A: Plot cumulative explained variance, choose k where you retain 95% of variance, or use elbow method on scree plot.

### Practical Questions

**Q: How do you detect overfitting?**
A: Large gap between training and validation performance, high variance in cross-validation scores, model performs well on training but poorly on test data.

**Q: What's your approach to feature selection?**
A: Domain knowledge, correlation analysis, univariate statistical tests, recursive feature elimination, regularization methods (L1), and feature importance from tree-based models.

**Q: How do you handle imbalanced datasets?**
A: Resampling (over/under-sampling), cost-sensitive learning, different evaluation metrics (precision, recall, F1), ensemble methods, and synthetic data generation (SMOTE).

---

## 📝 Quick Reference Formulas

### Linear Regression
- **Prediction**: ŷ = Xβ
- **Cost**: J = (1/2m)||Xβ - y||²
- **Normal Equation**: β = (XᵀX)⁻¹Xᵀy

### Logistic Regression
- **Sigmoid**: σ(z) = 1/(1 + e⁻ᶻ)
- **Cost**: J = -Σ[y log(ĥ) + (1-y) log(1-ĥ)]

### Regularization
- **Ridge**: J = MSE + α||β||₂²
- **Lasso**: J = MSE + α||β||₁

### Evaluation Metrics
- **Precision**: TP/(TP + FP)
- **Recall**: TP/(TP + FN)
- **F1**: 2 × (Precision × Recall)/(Precision + Recall)
- **R²**: 1 - SS_res/SS_tot

---

## 🎯 Tips for Success

### For Exams
1. **Understand Concepts**: Don't just memorize formulas
2. **Practice Problems**: Work through mathematical derivations
3. **Connect Ideas**: Understand relationships between topics
4. **Draw Diagrams**: Visualize concepts (decision boundaries, PCA projections)

### For Interviews
1. **Start Simple**: Begin with basic explanation, then add complexity
2. **Use Examples**: Concrete examples make concepts clearer
3. **Discuss Tradeoffs**: Every method has pros and cons
4. **Ask Clarifying Questions**: Understand the specific context
5. **Practice Coding**: Be ready to implement basic algorithms

### For Projects
1. **Data First**: Understand your data before choosing methods
2. **Baseline Models**: Start simple, then increase complexity
3. **Validation Strategy**: Choose appropriate cross-validation
4. **Feature Engineering**: Often more important than algorithm choice
5. **Interpretability**: Consider stakeholder needs

---

*Happy Learning! 🚀*