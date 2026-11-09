import numpy as np

def _laderman_3x3():
    # Returns U, V, W for 3x3x3
    # U: 9x23, V: 9x23, W: 9x23
    # ... hardcoded coefficients ...
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    # ... verification logic ...
    pass

def construct():
    n, m, p = 3, 3, 6
    # Try Laderman
    U3, V3, W3 = _laderman_3x3()
    # Construct 3x3x6
    U = np.zeros((9, 46))
    V = np.zeros((18, 46))
    W = np.zeros((18, 46))
    
    U[:, :23] = U3
    U[:, 23:] = U3
    
    # V block 1
    for j in range(3):
        for k in range(3):
            V[j*6+k, :23] = V3[j*3+k, :]
            
    # V block 2
    for j in range(3):
        for k in range(3):
            V[j*6+(k+3), 23:] = V3[j*3+k, :]
            
    # W block 1
    for k in range(3):
        for i in range(3):
            W[k*3+i, :23] = W3[k*3+i, :]
            
    # W block 2
    for k in range(3):
        for i in range(3):
            W[(k+3)*3+i, 23:] = W3[k*3+i, :]
            
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback
        return _schoolbook()
