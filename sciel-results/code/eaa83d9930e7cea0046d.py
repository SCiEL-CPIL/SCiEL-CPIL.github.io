import math

def construct():
    # Based on the "difference-sparse" construction directive, we optimize
    # a subset of an arithmetic progression.
    # Analysis shows that U = {0, 1, 4, 6} (d=1, offsets 0,1,4,6)
    # yields a high score by balancing |U-U|/|U+U| and max(U).
    
    u = [0, 1, 4, 6]
    
    return u
