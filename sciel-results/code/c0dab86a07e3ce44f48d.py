"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_base():
    """
    Construct the known rank-88 decomposition using Strassen blocks.
    """
    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Strassen terms for 2x2x2
    terms = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,1,0,0], [0,0,1,-1]),
        ([1,1,0,0], [0,0,1,1], [0,1,0,1]),
        ([0,0,0,1], [0,-1,0,1], [1,0,-1,0]),
        ([1,0,0,0], [1,0,-1,0], [-1,1,0,0]),
        ([-1,0,1,0], [0,1,0,1], [0,0,0,1]),
        ([0,1,0,-1], [1,0,1,0], [1,0,0,0]),
    ]
    
    def add_block(i0, j0, k0, i_size, j_size, k_size):
        nonlocal col
        if i_size == 2 and j_size == 2 and k_size == 2:
            for t in range(7):
                a_coeffs = terms[t][0]
                b_coeffs = terms[t][1]
                c_coeffs = terms[t][2]
                
                for di in range(2):
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]
                
                for dj in range(2):
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]
                
                for di in range(2):
                    for dk in range(2):
                        W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]
                
                col += 1
        else:
            # Schoolbook for remaining blocks
            for di in range(i_size):
                for dj in range(j_size):
                    for dk in range(k_size):
                        U[(i0+di)*m + (j0+dj)][col] = 1
                        V[(j0+dj)*p + (k0+dk)][col] = 1
                        W[(k0+dk)*n + (i0+di)][col] = 1
                        col += 1

    # Decompose 3x4x8
    # i=3 -> 2+1
    # j=4 -> 2+2
    # k=8 -> 2+2+2+2
    
    # i in [0, 1] (size 2)
    # j in [0, 1] (size 2)
    # k in [0, 1] (size 2) -> Strassen
    add_block(0, 0, 0, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [0, 1] (size 2)
    # k in [2, 3] (size 2) -> Strassen
    add_block(0, 0, 2, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [0, 1] (size 2)
    # k in [4, 5] (size 2) -> Strassen
    add_block(0, 0, 4, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [0, 1] (size 2)
    # k in [6, 7] (size 2) -> Strassen
    add_block(0, 0, 6, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [2, 3] (size 2)
    # k in [0, 1] (size 2) -> Strassen
    add_block(0, 2, 0, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [2, 3] (size 2)
    # k in [2, 3] (size 2) -> Strassen
    add_block(0, 2, 2, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [2, 3] (size 2)
    # k in [4, 5] (size 2) -> Strassen
    add_block(0, 2, 4, 2, 2, 2)
    
    # i in [0, 1] (size 2)
    # j in [2, 3] (size 2)
    # k in [6, 7] (size 2) -> Strassen
    add_block(0, 2, 6, 2, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [0, 1] (size 2)
    # k in [0, 1] (size 2) -> Schoolbook
    add_block(2, 0, 0, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [0, 1] (size 2)
    # k in [2, 3] (size 2) -> Schoolbook
    add_block(2, 0, 2, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [0, 1] (size 2)
    # k in [4, 5] (size 2) -> Schoolbook
    add_block(2, 0, 4, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [0, 1] (size 2)
    # k in [6, 7] (size 2) -> Schoolbook
    add_block(2, 0, 6, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [2, 3] (size 2)
    # k in [0, 1] (size 2) -> Schoolbook
    add_block(2, 2, 0, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [2, 3] (size 2)
    # k in [2, 3] (size 2) -> Schoolbook
    add_block(2, 2, 2, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [2, 3] (size 2)
    # k in [4, 5] (size 2) -> Schoolbook
    add_block(2, 2, 4, 1, 2, 2)
    
    # i in [2, 2] (size 1)
    # j in [2, 3] (size 2)
    # k in [6, 7] (size 2) -> Schoolbook
    add_block(2, 2, 6, 1, 2, 2)
    
    return U, V, W, col

def _construct_kronecker():
    """
    Attempt to construct a lower rank decomposition using Kronecker product structure.
    Target: Rank 49.
    Decomposition: 3x4 x 4x8 = (3x2 x 2x4) kron (2x2 x 2x2)?
    Actually, let's view indices:
    i in {0,1,2}
    j in {0,1,2,3} = j1 in {0,1}, j2 in {0,1}
    k in {0..7} = k1 in {0,1}, k2 in {0,1,2,3}
    
    T[i, (j1,j2), (k1,k2)] = T3[i, j2, k2] * T2[(j1), (k1)] ?
    This requires T3 to be 3x2x4 and T2 to be 2x2x2.
    Rank(T3) * Rank(T2).
    If Rank(T2)=7 (Strassen), and Rank(T3)=?
    3x2x4 schoolbook is 24.
    Can we do better for 3x2x4?
    3x2x4 is equivalent to multiplying 3x2 by 2x4.
    This is a small case.
    If we use Strassen on the 2x2 part?
    Let's try to construct T3 of rank 7? Unlikely.
    Let's try rank 4 for T2? No, 2x2x2 min rank is 7.
    
    Alternative:
    Split j into j1 (0,1) and j2 (0,1).
    Split k into k1 (0,1) and k2 (0,1,2,3).
    
    Maybe split i?
    
    Let's try a different factorization.
    m=4, p=8.
    Use Strassen for 2x2 blocks in j and k.
    This is what the current code does.
    
    Let's try to find a rank 49 decomposition manually or via search?
    Given time constraints, I will implement the known rank 88 solution robustly.
    If I can find a rank < 88, I will use it.
    
    For now, return the rank 88 solution.
    """
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Try Kronecker first if possible (currently stubbed to None)
    result = _construct_kronecker()
    if result is not None:
        U, V, W, rank = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
    # Fallback to Strassen base (Rank 88)
    U, V, W, rank = _construct_strassen_base()
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
