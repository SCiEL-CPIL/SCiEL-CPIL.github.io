import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard: if we are close to the limit, return a safe constant
    # Though this function is called by the optimizer, not directly by the user 
    # in a loop that we control, the evaluator calls it. 
    # However, the task says "The evaluator fits params... minimizing the training MSE".
    # The evaluator calls equation(x, params) many times.
    # We cannot easily guard the *entire* evaluation from inside equation if it's called repeatedly.
    # But we can ensure equation itself is fast.
    
    # Normalize input to [-1, 1] for stable trigonometric functions
    # Domain is [-3, 3]
    t = x / 3.0
    
    # Fourier basis functions
    # We have 10 parameters.
    # Let's use:
    # p0: Constant (DC term)
    # p1: cos(pi * t)
    # p2: sin(pi * t)
    # p3: cos(2 * pi * t)
    # p4: sin(2 * pi * t)
    # p5: cos(3 * pi * t)
    # p6: sin(3 * pi * t)
    # p7: cos(4 * pi * t)
    # p8: sin(4 * pi * t)
    # p9: cos(5 * pi * t)
    
    # Note: Using pi * t means frequency 1 corresponds to period 2 in t, or period 6 in x.
    # Since x is in [-3, 3], t is in [-1, 1].
    # cos(pi * t) goes from cos(-pi) to cos(pi) -> -1 to -1, passing through 1 at t=0.
    # This captures the fundamental frequency of the domain.
    
    c1 = np.cos(np.pi * t)
    s1 = np.sin(np.pi * t)
    
    c2 = np.cos(2 * np.pi * t)
    s2 = np.sin(2 * np.pi * t)
    
    c3 = np.cos(3 * np.pi * t)
    s3 = np.sin(3 * np.pi * t)
    
    c4 = np.cos(4 * np.pi * t)
    s4 = np.sin(4 * np.pi * t)
    
    c5 = np.cos(5 * np.pi * t)
    
    y = (params[0] * np.ones_like(t) +
         params[1] * c1 +
         params[2] * s1 +
         params[3] * c2 +
         params[4] * s2 +
         params[5] * c3 +
         params[6] * s3 +
         params[7] * c4 +
         params[8] * s4 +
         params[9] * c5)
         
    return y
