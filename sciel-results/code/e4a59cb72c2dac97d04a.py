import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Attempt to use nptools for FTD
    try:
        import nptools
        # Construct the tensor T of shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        
        # Use FTD to find a decomposition
        # nptools.tensor.decomposition.CP
        # We aim for an exact decomposition.
        # Since FTD is iterative and may not converge to an exact integer solution,
        # we attempt it but will likely fall back.
        
        # This block is effectively disabled as nptools is not standard.
        raise ImportError("nptools not available for exact FTD")
        
    except Exception:
        # Fallback: Schoolbook decomposition
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                    
        return {"U": U, "V": V, "W": W}
