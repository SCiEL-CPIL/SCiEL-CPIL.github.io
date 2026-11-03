import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # Strassen 2x2x2 terms
    # U, V, W are 2x2 matrices
    S1 = [
        (np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]]), np.array([[1,0],[0,1]])),
        (np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]), np.array([[0,0],[1,0]])),
        (np.array([[1,0],[0,0]]), np.array([[0,1],[0,-1]]), np.array([[0,1],[0,0]])),
        (np.array([[0,0],[0,1]]), np.array([[-1,0],[1,0]]), np.array([[1,0],[1,0]])),
        (np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]]), np.array([[-1,1],[0,0]])),
        (np.array([[-1,0],[1,0]]), np.array([[1,1],[0,0]]), np.array([[0,0],[0,1]])),
        (np.array([[0,1],[0,-1]]), np.array([[0,0],[1,1]]), np.array([[1,0],[0,0]]))
    ]
    
    # Generate 4x4x4 terms using Kronecker product
    terms_4x4x4 = []
    for u1, v1, w1 in S1:
        for u2, v2, w2 in S1:
            u = np.kron(u1, u2)
            v = np.kron(v1, v2)
            w = np.kron(w1, w2)
            terms_4x4x4.append((u, v, w))
            
    # Generate 4x4x3 terms using schoolbook
    # C is 4x3. i in 0..3, j in 0..3, k in 0..2
    terms_4x4x3 = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                u = np.zeros((4,4))
                v = np.zeros((4,4))
                w = np.zeros((4,3))
                u[i,j] = 1
                v[j,k] = 1
                w[i,k] = 1
                terms_4x4x3.append((u, v, w))
                
    all_terms = terms_4x4x4 + terms_4x4x3
    rank = len(all_terms)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u, v, w) in enumerate(all_terms):
        U[:, r] = np.flatten(u, order='F')
        V[:, r] = np.flatten(v, order='F')
        W[:, r] = np.flatten(w, order='F')
        
    # Convert to lists as required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The seed program returned lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
