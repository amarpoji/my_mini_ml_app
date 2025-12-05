import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

# simple model for demonstration
def train_model():
    X = np.array([[500], [1000], [1500], [2000], [2500]])  # house size
    y = np.array([100000, 180000, 260000, 330000, 400000])  # price

    model = LinearRegression()
    model.fit(X, y)

    # save model to project root
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    train_model()
