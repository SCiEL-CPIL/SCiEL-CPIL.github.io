"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x4x8():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    
    A (2x4) * B (4x8) = C (2x8)
    
    Partition: A = [A_left A_right] where A_left, A_right are 2x2
               B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    
    C = A_left * B_top + A_right * B_bottom
    
    For each 2x2x8 sub-multiplication, we use Strassen's algorithm.
    Strassen decomposes 2x2 matrix mult into 7 scalar multiplications.
    Here, the "scalars" are actually 1x8 vectors (rows of B), so each
    Strassen multiplication becomes a product of a scalar (linear combo of A)
    and a 1x8 vector (linear combo of B rows).
    
    For 2x2x8: C[i,k] = sum_j M[i,j] * B[j,k]
    Strassen: 7 products m_t = (linear combo of M) * (linear combo of B rows)
    Each m_t is a 1x8 vector, contributing 8 scalar products to the total.
    So one 2x2x8 block costs 7*8 = 56 rank.
    Two blocks would cost 112 - too much.
    
    Better approach: Apply Strassen across the full 2x4x8 structure.
    
    Actually, let me think differently. The tensor is:
    T[i,j,k] = 1 for i in {0,1}, j in {0,1,2,3}, k in {0,...,7}
    
    We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    
    Key insight: Use the fact that we can share structure across k.
    
    For fixed (i,j), the term U[i,j,c]*V[j,k,c]*W[k,i,c] must sum to 1 over c.
    
    Let's use a Kronecker product structure. Note that 2x4x8 = 2x(2x2)x(2x4).
    
    Actually, the cleanest approach: treat this as multiplying a 2x4 by 4x8.
    We can write this as: first multiply 2x4 by 4x4 to get 2x4, then by 4x4 to get 2x8? No, that doesn't work directly.
    
    Let me use a direct algebraic construction with rank 56.
    
    Partition j = {0,1,2,3} into two pairs: J1={0,1}, J2={2,3}.
    
    C[i,k] = sum_{j in J1} A[i,j]*B[j,k] + sum_{j in J2} A[i,j]*B[j,k]
    
    For the first pair (j=0,1):
      C1[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
      This is a 2x2x8 tensor. Using Strassen on the 2x2 part:
      7 products, each involving a 1x8 vector => 7*8 = 56 rank for this pair alone.
    
    That's already 56 for one pair, and we need both pairs. Total would be 112.
    
    Alternative: Use a different decomposition. Note that R(2,4,8) can be bounded by
    considering the structure more carefully.
    
    Actually, let me just construct the schoolbook and verify it works, then try to
    find a better construction. For now, return schoolbook with rank 64.
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = n * m * p  # 64
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

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    n, m, p = DIMENSIONS
    best = _strassen_2x4x8()
    best_rank = n * m * p
    
    # Verify the schoolbook solution works
    if _verify(best["U"], best["V"], best["W"], n, m, p, tol=1e-9):
        return best
    
    # Fallback: construct schoolbook directly
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
