import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition by summing decompositions for sub-problems.
    # The total rank is the sum of the ranks of the sub-problems.
    # We split the p=9 dimension into p1=4 and p2=5.
    # Then p2=5 into p21=2 and p22=3.
    # So we have sub-problems: (4, 4, 4), (4, 4, 2), (4, 4, 3).
    
    # We need to generate U, V, W for each sub-problem and concatenate them.
    # The indices for U are (i, j), for V are (j, k), for W are (k, i).
    # The sub-problems have different k ranges.
    # For (4, 4, 4), k is in [0, 4).
    # For (4, 4, 2), k is in [4, 6).
    # For (4, 4, 3), k is in [6, 9).
    
    # We will use the schoolbook algorithm for simplicity, but try to use Strassen for (4, 4, 4).
    # Actually, for (4, 4, 4), we can use 49 multiplications.
    # For (4, 4, 2), we can use 14 multiplications.
    # For (4, 4, 3), we can use 22 multiplications.
    # Total rank = 49 + 14 + 22 = 85.
    
    # However, implementing Strassen's algorithm for (4, 4, 4) and the others is complex.
    # Let's start with the schoolbook algorithm for all parts, which gives rank 144.
    # Then we can try to optimize.
    
    # For now, let's return the schoolbook decomposition.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
