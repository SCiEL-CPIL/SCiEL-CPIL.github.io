import numpy as np

N = 600

def construct():
    # Create a zero vector of length N
    h = [0.0] * N
    # Set a single spike in the center to minimize any potential edge effects 
    # in discrete convolution implementations (though max is global)
    h[N // 2] = 1.0
    return h
