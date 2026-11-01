"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a minimal-rank tensor decomposition for 2x2 by 2x14 matrix multiplication.
    Tries rank-28 first, falls back to rank-49 Strassen, then rank-56 schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Try rank-28 construction first
    result = _construct_rank28(n, m, p)
    if result is not None:
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result

    # Attempt Strassen-based decomposition (rank 49)
    result = _construct_strassen_49(n, m, p)
    if result is not None:
        if _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result

    # Fallback: Schoolbook decomposition (rank 56)
    return _construct_schoolbook(n, m, p)

def _construct_rank28(n, m, p):
    """
    Attempt a rank-28 decomposition exploiting the structure where
    C[i,k] = A[0,k]*B[0,i] + A[1,k]*B[1,i].
    Uses 2 terms per k-value instead of 7 per 2-k block.
    """
    if time.time() > start_time_global + 140:
        return None

    rank = p * 2  # 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # For each k, we need:
    # C[0,k] = A[0,k]*B[0,0] + A[1,k]*B[1,0]
    # C[1,k] = A[0,k]*B[0,1] + A[1,k]*B[1,1]
    #
    # Term 2k: contributes A[0,k] * B[0,i] to C[i,k]
    # Term 2k+1: contributes A[1,k] * B[1,i] to C[i,k]
    #
    # But wait - this is just schoolbook with rank 4 (n*m*p = 4*14 = 56).
    # The rank-28 idea needs sharing.
    #
    # Alternative: use the fact that for fixed k, we have a 2x2 "matrix" B[:,k]
    # multiplied by a 1x2 vector A[:,k]^T... no, A is 2x2 so A[:,k] is a column.
    # Actually C[i,k] = sum_j A[j,k] * B[j,i]
    # This is: for each k, C[:,k] = B^T * A[:,k]
    # So we need rank-2 per k (since B^T is 2x2 and A[:,k] is 2x1)
    # That gives rank = 2 * 14 = 28!
    #
    # For each k:
    #   C[0,k] = B[0,0]*A[0,k] + B[1,0]*A[1,k]
    #   C[1,k] = B[0,1]*A[0,k] + B[1,1]*A[1,k]
    #
    # Term 2k: U picks A[0,k], V picks B[:,0] (column 0 of B), W picks identity for row 0
    #   Actually: U[0,k] = A[0,k] (coefficient for A[0,k])
    #   V[0,k] = B[0,0], V[1,k] = B[0,1] -- no, V is indexed by (j,k) where j is row of B
    #
    # Let me re-derive. U[i*m+j, c] where i=0..1, j=0..1 indexes A[i,j]
    # V[j*p+k, c] where j=0..1, k=0..13 indexes B[j,k]
    # W[k*n+i, c] where k=0..13, i=0..1 indexes C[k,i]
    #
    # We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for the "diagonal"
    # But actually the tensor T[i,j,k] = 1 if the multiplication gives C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for this specific case!)
    #
    # Wait no. The multiplication tensor for (n,m,p) = (2,2,14) has:
    # T[i, j, k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,13}
    # because C[i,k] = sum_j A[i,j] * B[j,k], so every triple (i,j,k) contributes.
    #
    # So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for ALL i,j,k.
    #
    # Rank-28: For each k, use 2 terms.
    # Term (k, 0): U[0,0,c]=1, U[1,0,c]=0 (picks A[0,0])
    #              V[0,k,c]=1, V[1,k,c]=0 (picks B[0,k])
    #              W[k,0,c]=1, W[k,1,c]=0 (contributes to C[0,k])
    #              This gives contribution 1 to T[0,0,k]
    #
    # But we need T[1,0,k]=1 too. So this single term only covers (i=0,j=0,k).
    #
    # The issue: with 2 terms per k, we can cover at most 2 of the 4 (i,j) pairs for each k.
    # We need all 4 pairs (i=0,1; j=0,1) for each k. So 2 terms per k is insufficient
    # unless we use non-trivial linear combinations.
    #
    # Let's try: for each k, use terms that cover multiple (i,j) pairs.
    # Term (k,0): U = [1, 0, 0, 0] (A[0,0]), V = [e_k for row 0, e_k for row 1]... 
    #
    # Actually V[j,k,c] is a single value per (j,k). So for fixed k and term c:
    # V[0,k,c] and V[1,k,c] are scalars.
    #
    # Let me try: for each k, 2 terms:
    #   c0 = 2k:   U = [1, 0, 1, 0] (A[0,0]+A[1,0]), V = [1, 1] at positions (0,k),(1,k)
    #              W = [1, 1] at positions (k,0),(k,1)
    #              Contribution to T[i,j,k]: U[i,j]*V[j,k]*W[k,i]
    #              T[0,0,k]: 1*1*1 = 1
    #              T[1,0,k]: 1*1*1 = 1  
    #              T[0,1,k]: 0*1*1 = 0
    #              T[1,1,k]: 0*1*1 = 0
    #
    #   c1 = 2k+1: U = [0, 0, 0, 1] (A[1,1]), V = [0, 1] at positions (0,k),(1,k)
    #              W = [0, 1] at positions (k,0),(k,1)
    #              T[0,0,k]: 0*0*0 = 0
    #              T[1,0,k]: 0*0*0 = 0
    #              T[0,1,k]: 0*1*0 = 0
    #              T[1,1,k]: 1*1*1 = 1
    #
    # Total: T[0,0,k]=1, T[1,0,k]=1, T[0,1,k]=0, T[1,1,k]=1
    # Missing T[0,1,k]=1!
    #
    # Let me try different forms:
    #   c0: U = [1, 0, 0, 0] (A[0,0]), V = [1, 0], W = [1, 0]
    #       T[0,0,k]=1, others 0
    #   c1: U = [0, 0, 1, 1] (A[1,0]+A[1,1]), V = [0, 1], W = [1, 1]
    #       T[0,0,k]=0, T[1,0,k]=1*0*1=0... no
    #
    # This is getting complex. Let me just verify that rank 28 is not achievable
    # with simple constructions and fall back to 49.
    return None

def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    if time.time() > start_time_global + 140:
        return None
        
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_forms = [
        [1, 0, 0, 1],   # s=0: A00+A11
        [0, 0, 1, 1],   # s=1: A10+A11
        [1, 0, 0, 0],   # s=2: A00
        [0, 0, 0, 1],   # s=3: A11
        [1, 1, 0, 0],   # s=4: A00+A01
        [-1, 0, 1, 0],  # s=5: A10-A00
        [0, 1, 0, -1]   # s=6: A01-A11
    ]
    
    for r in range(num_blocks):
        idx_e = 2 * r       # B00
        idx_f = 2 * r + 1   # B01
        idx_g = p + 2 * r   # B10
        idx_h = p + 2 * r + 1 # B11
        
        w_idx_c00 = 4 * r
        w_idx_c10 = 4 * r + 1
        w_idx_c01 = 4 * r + 2
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            if s == 0:
                V[idx_e][c] = 1
                V[idx_h][c] = 1
            elif s == 1:
                V[idx_e][c] = 1
            elif s == 2:
                V[idx_f][c] = 1
                V[idx_h][c] = -1
            elif s == 3:
                V[idx_e][c] = -1
                V[idx_g][c] = 1
            elif s == 4:
                V[idx_h][c] = 1
            elif s == 5:
                V[idx_e][c] = 1
                V[idx_f][c] = 1
            elif s == 6:
                V[idx_g][c] = 1
                V[idx_h][c] = 1
                
            # Fill W
            if s == 0:
                W[w_idx_c00][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 1:
                W[w_idx_c10][c] = 1
                W[w_idx_c11][c] = -1
            elif s == 2:
                W[w_idx_c01][c] = 1
                W[w_idx_c11][c] = 1
            elif s == 3:
                W[w_idx_c00][c] = 1
                W[w_idx_c10][c] = 1
            elif s == 4:
                W[w_idx_c00][c] = -1
                W[w_idx_c01][c] = 1
            elif s == 5:
                W[w_idx_c11][c] = 1
            elif s == 6:
                W[w_idx_c00][c] = 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_tests=2):
    """
    Verify that the decomposition correctly computes matrix multiplication.
    Checks the tensor condition: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    for all i, j, k.
    """
    rank = len(U[0])
    # Check all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _construct_schoolbook(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

start_time_global = time.time()
# EVOLVE-BLOCK-END
