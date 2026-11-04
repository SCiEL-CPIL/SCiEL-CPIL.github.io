def _verify_laderman():
    import random
    import numpy as np
    # Coefficients for Laderman
    # m1 = a*e
    # m2 = b*(e+f)
    # m3 = c*(g+h)
    # m4 = d*h
    # m5 = (a+c)*(e+g)
    # m6 = (b+d)*(f+h)
    # m7 = (a+b)*(e-g)
    
    # U, V, W matrices for the 7 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[C00,C01],[C10,C11]]
    
    U = [
        [1, 0, 0, 0], # a
        [0, 1, 0, 0], # b
        [0, 0, 1, 0], # c
        [0, 0, 0, 1], # d
        [1, 0, 1, 0], # a+c
        [0, 1, 0, 1], # b+d
        [1, 1, 0, 0], # a+b
    ]
    V = [
        [1, 0, 0, 0], # e
        [1, 1, 0, 0], # e+f
        [0, 0, 1, 1], # g+h
        [0, 0, 0, 1], # h
        [1, 0, 1, 0], # e+g
        [0, 1, 0, 1], # f+h
        [1, 0, -1, 0], # e-g
    ]
    W = [
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C01
        [0, 0, 1, 0], # C10
        [0, 0, 0, 1], # C11
        [1, 0, 1, 0], # C00, C10
        [0, 1, 0, 1], # C01, C11
        [-1, 1, -1, 1], # -C00, C01, -C10, C11
    ]
    
    # Verify
    for _ in range(10):
        A = np.random.rand(2, 2)
        B = np.random.rand(2, 2)
        C_true = A @ B
        
        C_calc = np.zeros((2, 2))
        a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
        e, f, g, h = B[0,0], B[0,1], B[1,0], B[1,1]
        
        m1 = a * e
        m2 = b * (e + f)
        m3 = c * (g + h)
        m4 = d * h
        m5 = (a + c) * (e + g)
        m6 = (b + d) * (f + h)
        m7 = (a + b) * (e - g)
        
        C_calc[0,0] = m1 + m5 - m7
        C_calc[0,1] = m2 + m6 - m7
        C_calc[1,0] = m3 + m5 - m7
        C_calc[1,1] = m4 + m6 - m7
        
        if not np.allclose(C_true, C_calc):
            return False
    return True

def construct():
    if _verify_laderman():
        # Use Laderman for 2x2 parts
        # Rank 54
        pass
    else:
        # Use Schoolbook
        # Rank 60
        pass
