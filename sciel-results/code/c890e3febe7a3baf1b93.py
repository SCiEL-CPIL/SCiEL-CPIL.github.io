import numpy as np

def strassen_2x2(A, B):
    # A, B: 2x2 numpy arrays
    a, b = A[0,0], A[0,1]
    c, d = A[1,0], A[1,1]
    e, f = B[0,0], B[0,1]
    g, h = B[1,0], B[1,1]
    M1 = (a + d) * (e + h)
    M2 = (c + d) * e
    M3 = (a - c) * (g + h)
    M4 = a * (e - g)
    M5 = d * (f - e)
    M6 = (a + b) * g
    M7 = (b + c) * (f - h)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M3 + M6 - M2
    return np.array([[C11, C12], [C21, C22]])
