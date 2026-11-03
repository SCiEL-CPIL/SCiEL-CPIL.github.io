"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the full tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Check against the true multiplication tensor: T[i,j,k] = 1 if j==j (always), 
    # actually T[i,j,k] should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'} ... 
    # The tensor for matmul: for each (i,j,k), the coefficient is 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid (i,j,k) in the sense that 
    # the multiplication is: C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor entries are: T[i,j,k] = 1 (each term A[i,j]*B[j,k] contributes to C[i,k])
    # So we need: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # And for "wrong" combinations... actually the tensor is defined only on the 
    # multiplication pattern. Let me re-read the convention.
    # W[k*n+i, r] - this is the output factor indexed by (k,i) where k is the column of C, i is the row
    # U[i*m+j, r] - input A factor indexed by (i,j)
    # V[j*p+k, r] - input B factor indexed by (j,k)
    # The decomposition says: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is because the tensor T_{ijk} = 1 for all i,j,k (each product term A[i,j]*B[j,k] 
    # appears exactly once in C[i,k] = sum_j A[i,j]*B[j,k])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Compute T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Use einsum
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # T should be all 1s
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
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

def _try_block_decomp(n, m, p):
    """Try to use block structure: split p into p1+p2, use known decompositions."""
    # For 4x4x9, try splitting 9 into 4+5
    # Use schoolbook for both parts initially, then try to improve
    best = None
    best_rank = n * m * p
    
    # Try various splits of p
    for p1 in range(1, p):
        p2 = p - p1
        # Schoolbook ranks
        r1 = n * m * p1
        r2 = n * m * p2
        total_rank = r1 + r2
        if total_rank < best_rank:
            # Build the combined decomposition
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Part 1: columns 0..p1-1 of B, columns 0..p1-1 of C
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p1):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            # Part 2: columns p1..p-1 of B, columns p1..p-1 of C
            for i in range(n):
                for j in range(m):
                    for k in range(p1, p):
                        U[i*m+j][col] = 1
                        V[j*p+k][col] = 1
                        W[k*n+i][col] = 1
                        col += 1
            
            if _verify(U, V, W, n, m, p):
                best_rank = total_rank
                best = {"U": U, "V": V, "W": W}
    
    return best, best_rank

def _strassen_4x4x4():
    """Return a rank-49 decomposition for 4x4x4 matrix multiplication.
    This uses known results. For now, return schoolbook (rank 64) as fallback."""
    n = m = p = 4
    # For simplicity, use schoolbook - rank 64
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try block decomposition
    block_result, block_rank = _try_block_decomp(n, m, p)
    if block_result is not None and block_rank < best_rank:
        best_result = block_result
        best_rank = block_rank
    
    # Check deadline
    if time.time() > _DEADLINE - 10:
        return best_result
    
    # Try to improve the 4x4x5 part using a small search
    # For now, return the best we have
    return best_result
# EVOLVE-BLOCK-END
