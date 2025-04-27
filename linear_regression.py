import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def main():
    print("=== Introduction to Linear Regression ===\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    print("1. Creating sample data...")
    X = 2 * np.random.rand(100, 1)  # House sizes between 0 and 2
    y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise
    
    # Plot the data
    print("\n2. Plotting the data...")
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y)
    plt.title('House Size vs Price')
    plt.xlabel('House Size (thousands of sq ft)')
    plt.ylabel('Price (thousands of dollars)')
    plt.grid(True)
    plt.show()
    
    # Create and train the model
    print("\n3. Training the linear regression model...")
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the model parameters
    print(f"\nModel parameters:")
    print(f"Slope (m): {model.coef_[0][0]:.2f}")
    print(f"Intercept (b): {model.intercept_[0]:.2f}")
    
    # Make predictions
    print("\n4. Making predictions and plotting results...")
    X_new = np.array([[0], [2]])  # Points to predict
    y_pred = model.predict(X_new)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, label='Data Points')
    plt.plot(X_new, y_pred, 'r-', label='Regression Line')
    plt.title('Linear Regression Results')
    plt.xlabel('House Size')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n=== Summary ===")
    print("In this tutorial, we learned:")
    print("1. How to create sample data")
    print("2. How to train a linear regression model")
    print("3. How to make predictions")
    print("4. How to visualize the results")

if __name__ == "__main__":
    main() 