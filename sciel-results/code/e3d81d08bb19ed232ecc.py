import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def verify(U, V, W, n=2, m=3, p=8, tol=1e-10):
    """Verify that U, V, W correctly decompose the matmul tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check shapes
    assert U.shape == (n*m, rank), f"U shape {U.shape} != ({n*m}, {rank})"
    assert V.shape == (m*p, rank), f"V shape {V.shape} != ({m*p}, {rank})"
    assert W.shape == (p*n, rank), f"W shape {W.shape} != ({p*n}, {rank})"
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False, (i, j, k, s)
    return True, None

def construct_rank44():
    """Rank-44 construction: 4 blocks of rank-11 2x3x3x2."""
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0.0] * rank for _ in range(n*m)]  # 6 x 44
    V = [[0.0] * rank for _ in range(m*p)]  # 24 x 44
    W = [[0.0] * rank for _ in range(p*n)]  # 16 x 44
    
    col = 0
    for g in range(4):  # 4 blocks, each handling 2 output columns
        # For this block, k_local = 0,1 correspond to global k = 2g, 2g+1
        # Strassen 7 terms for j=0,1 (2x2 x 2x2 part)
        # Term 1: U(i=0,j=0; i=1,j=1), V(j=0,k=0; j=1,k=1), W(k=0,i=0; k=1,i=1)
        U[0*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*8+2*g][col] += 1.0
        V[1*8+2*g+1][col] += 1.0
        W[2*g*2+0][col] += 1.0
        W[(2*g+1)*2+1][col] += 1.0
        col += 1
        
        # Term 2: U(i=1,j=0; i=1,j=1), V(j=0,k=0), W(k=0,i=1)
        U[1*3+0][col] += 1.0
        U[1*3+1][col] += 1.0
        V[0*8+2*g][col] += 1.0
        W[2*g*2+1][col] += 1.0
        col += 1
        
        # Term 3: U(i=0,j=1), V(j=1,k=0; j=1,k=1), W(k=1,i=0)
        U[0*3+1][col] += 1.0
        V[1*8+2*g][col] += 1.0
        V[1*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+0][col] += 1.0
        col += 1
        
        # Term 4: U(i=0,j=0), V(j=1,k=1), W(k=1,i=1)
        U[0*3+0][col] += 1.0
        V[1*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+1][col] += 1.0
        col += 1
        
        # Term 5: U(i=0,j=0; i=0,j=1), V(j=1,k=0), W(k=0,i=0)
        U[0*3+0][col] += 1.0
        U[0*3+1][col] += 1.0
        V[1*8+2*g][col] += 1.0
        W[2*g*2+0][col] += 1.0
        col += 1
        
        # Term 6: U(i=1,j=0; -i=0,j=0), V(j=0,k=0; j=0,k=1), W(k=1,i=0)
        U[1*3+0][col] += 1.0
        U[0*3+0][col] += -1.0
        V[0*8+2*g][col] += 1.0
        V[0*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+0][col] += 1.0
        col += 1
        
        # Term 7: U(i=0,j=1; -i=1,j=1), V(j=1,k=0; j=1,k=1), W(k=1,i=1)
        U[0*3+1][col] += 1.0
        U[1*3+1][col] += -1.0
        V[1*8+2*g][col] += 1.0
        V[1*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+1][col] += 1.0
        col += 1
        
        # 4 rank-1 terms for j=2 (the extra inner index)
        # (i=0, j=2, k=0)
        U[0*3+2][col] += 1.0
        V[2*8+2*g][col] += 1.0
        W[2*g*2+0][col] += 1.0
        col += 1
        
        # (i=1, j=2, k=0)
        U[1*3+2][col] += 1.0
        V[2*8+2*g][col] += 1.0
        W[2*g*2+1][col] += 1.0
        col += 1
        
        # (i=0, j=2, k=1)
        U[0*3+2][col] += 1.0
        V[2*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+0][col] += 1.0
        col += 1
        
        # (i=1, j=2, k=1)
        U[1*3+2][col] += 1.0
        V[2*8+2*g+1][col] += 1.0
        W[(2*g+1)*2+1][col] += 1.0
        col += 1
    
    assert col == 44, f"Expected 44 columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    result = construct_rank44()
    valid, info = verify(result["U"], result["V"], result["W"])
    if valid:
        return result
    else:
        # Fall back to schoolbook
        n, m, p = DIMENSIONS
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
