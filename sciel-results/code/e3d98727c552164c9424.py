import numpy as np

N = 600

def construct():
    # Generate a random 0/1 step function
    # This should minimize off-peak autocorrelation
    np.random.seed(42)
    h = np.random.randint(0, 2, N)
    return h.tolist()
