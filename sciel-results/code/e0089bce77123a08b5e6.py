"""Program for matrix multiplication 2x2 by 2x10 tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    
    The tensor for C = A B where A is 2x2 and B is 2x10 is decomposed into 5 independent
    2x2x2 multiplication problems. Each 2x2x2 problem is solved using Strassen's algorithm
    with rank 7. Total rank = 5 * 7 = 35.
    
    Interface:
    U: shape (n*m, rank) = (4, 35)
    V: shape (m*p, rank) = (20, 35)
    W: shape (p*n, rank) = (20, 35)
    
    Indices:
    U index: i*m + j  (i in 0..1, j in 0..1) -> 0,1,2,3
    V index: j*p + k  (j in 0..1, k in 0..9) -> 0..19
    W index: k*n + i  (k in 0..9, i in 0..1) -> 0..19 (Official trace convention)
    """
    n, m, p = DIMENSIONS
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen coefficients for the 7 products
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[C00, C01], [C10, C11]]
    
    strassen_A = [
        [1, 0, 0, 1],   # P1: (a+d)
        [0, 0, 1, 1],   # P2: (c+d)
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: (a+b)
        [-1, 0, 1, 0],  # P6: (c-a)
        [0, 1, 0, -1]   # P7: (b-d)
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: (e+h)
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: (f-h)
        [-1, 0, 1, 0],  # P4: (g-e)
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: (e+f)
        [0, 0, 1, 1]    # P7: (g+h)
    ]
    
    # W contributions for each Strassen product to the local 2x2 C matrix
    # local indices: 0->C00, 1->C01, 2->C10, 3->C11
    w_contrib = [
        [1,  0,  0,  1],  # P1: C00 +1, C11 +1
        [0,  0,  1, -1],  # P2: C10 +1, C11 -1
        [0,  1,  0,  1],  # P3: C01 +1, C11 +1
        [1,  0,  1,  0],  # P4: C00 +1, C10 +1
        [-1, 1,  0,  0],  # P5: C00 -1, C01 +1
        [0,  0,  0,  1],  # P6: C11 +1
        [1,  0,  0,  0]   # P7: C00 +1
    ]
    
    col = 0
    for b in range(5):
        for r in range(7):
            # U: same for all blocks
            for i in range(4):
                U[i][col] = strassen_A[r][i]
            
            # V: specific to block b
            # B indices: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
            # V index: j*p + k. 
            # Row 0 of B (j=0): k=2b, 2b+1 -> V[0*10 + 2b], V[0*10 + 2b+1]
            # Row 1 of B (j=1): k=2b, 2b+1 -> V[1*10 + 2b], V[1*10 + 2b+1]
            V[2*b][col] = strassen_B[r][0]      # e
            V[2*b+1][col] = strassen_B[r][1]    # f
            V[10+2*b][col] = strassen_B[r][2]   # g
            V[10+2*b+1][col] = strassen_B[r][3] # h
            
            # W: specific to block b
            # W index: k*n + i.
            # C00: k=2b, i=0 -> W[2b*2 + 0] = W[4b]
            # C01: k=2b+1, i=0 -> W[(2b+1)*2 + 0] = W[4b+2]
            # C10: k=2b, i=1 -> W[2b*2 + 1] = W[4b+1]
            # C11: k=2b+1, i=1 -> W[(2b+1)*2 + 1] = W[4b+3]
            W[4*b][col] = w_contrib[r][0]      # C00
            W[4*b+2][col] = w_contrib[r][1]    # C01
            W[4*b+1][col] = w_contrib[r][2]    # C10
            W[4*b+3][col] = w_contrib[r][3]    # C11
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

def _verify_decomposition(result, n=2, m=2, p=10, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    rank = U.shape[1]
    
    # Build the expected tensor
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build the reconstructed tensor
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    diff = np.max(np.abs(T_reconstructed - T_expected))
    return diff < tol

if __name__ == "__main__":
    result = construct()
    valid = _verify_decomposition(result)
    print(f"Rank: {result['U'].shape[1]}")
    print(f"Valid: {valid}")
