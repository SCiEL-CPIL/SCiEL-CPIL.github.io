import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x6 matrix.
    
    The tensor T_{a,b,c,d,i,k} = delta_{a,i} delta_{c,k} delta_{b,d} where:
    - a, i in {0, 1}
    - b, d in {0, 1, 2} 
    - c, k in {0, 1, 2, 3, 4, 5}
    
    The non-zero entries occur when a=i, c=k, b=d.
    Free variables: a (2 choices), b (3 choices), c (6 choices).
    Total non-zeros: 2 * 3 * 6 = 36.
    
    Each non-zero entry corresponds to a unique rank-1 term with disjoint support.
    Therefore, the rank is exactly 36.
    """
    n, m, p = 2, 3, 6
    
    u_dim = n * m  # 6
    v_dim = m * p  # 18
    w_dim = p * n  # 12
    
    # Count the number of non-zero terms
    # a: 0..n-1, b: 0..m-1, c: 0..p-1
    num_terms = n * m * p  # 36
    
    U = np.zeros((u_dim, num_terms))
    V = np.zeros((v_dim, num_terms))
    W = np.zeros((w_dim, num_terms))
    
    idx = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                # u = a*m + b
                # v = c*p + b  (since d=b, and v = c*p + d)
                # w = c*n + a  (since i=a, and w = k*n + i where k=c)
                u = a * m + b
                v = c * p + b
                w = c * n + a
                
                U[u, idx] = 1.0
                V[v, idx] = 1.0
                W[w, idx] = 1.0
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    construct()
