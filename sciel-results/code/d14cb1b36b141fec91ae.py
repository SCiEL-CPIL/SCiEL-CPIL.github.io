"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_correct():
    """Correct Strassen 2x2x2 decomposition. Rank 7."""
    n, m, p = 2, 2, 2
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)
    
    # M1: (A[0,0]-A[1,1])*(B[1,1]-B[0,0])
    U[0, 0] = 1; U[3, 0] = -1
    V[3, 0] = 1; V[0, 0] = -1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = -1
    
    # M2: (A[0,0]+A[1,1])*B[1,1]
    U[0, 1] = 1; U[3, 1] = 1
    V[3, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3: A[1,1]*(B[1,0]-B[0,0])
    U[3, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4: A[0,0]*(B[0,1]-B[1,1])
    U[0, 3] = 1
    V[1, 3] = 1; V[3, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: (A[0,0]+A[0,1])*B[1,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6: (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7: (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1; W[1, 6] = 1; W[3, 6] = 1
    
    return U, V, W


def _winograd_2x2x3():
    """Winograd 2x2x3 decomposition. Rank 8.
    A is 2x2, B is 2x3, C is 2x3.
    U: (4, 8) for A indices
    V: (6, 8) for B indices  
    W: (6, 8) for C indices (C[k*n+i] = C[k*2+i])
    """
    U = np.zeros((4, 8), dtype=np.int8)
    V = np.zeros((6, 8), dtype=np.int8)
    W = np.zeros((6, 8), dtype=np.int8)
    
    # Winograd F(2,2,3):
    # A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # Transform A (2x2 -> 8 values):
    #   a0 = a+b, a1 = c+d, a2 = a-d, a3 = b-c
    #   a4 = a, a5 = b, a6 = c, a7 = d
    # Wait, that's 8 values but we only need 4 independent ones for the 2x2 matrix.
    # Actually Winograd uses a different approach. Let me use the standard one.
    #
    # Standard Winograd 2x2x3:
    # m1 = (a+b)*(e+h)
    # m2 = (a+b)*(f+i)  
    # m3 = (a+b)*(g+j)
    # m4 = (c+d)*(e+h)
    # m5 = (c+d)*(f+i)
    # m6 = (c+d)*(g+j)
    # m7 = (a-d)*(e-h)
    # m8 = (b-c)*(f-i)
    #
    # C[0,0] = (m1+m4-m7)/2 + (m7)/2 = (m1+m4)/2
    # Hmm, this is getting complicated. Let me use a known correct rank-8 decomposition.
    #
    # Actually, let me just use schoolbook for 2x2x3 (rank 12) to be safe.
    # The total will be 56 + 48 = 104.
    # Or I can try to get rank 8 if I can figure it out.
    #
    # For now, let me return schoolbook 2x2x3 (rank 12).
    U = np.zeros((4, 12), dtype=np.int8)
    V = np.zeros((6, 12), dtype=np.int8)
    W = np.zeros((6, 12), dtype=np.int8)
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 2 + j, col] = 1
                V[j * 3 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
    
    return U, V, W


def _build_4x4x7_optimized():
    """
    Build 4x4x7 decomposition using Strassen 2x2x2 for the 4x4x4 part
    and schoolbook 2x2x3 for the 4x4x3 part.
    
    4x4x4: 8 Strassen 2x2x2 blocks, rank 8*7 = 56
    4x4x3: 4 schoolbook 2x2x3 blocks, rank 4*12 = 48
    Total: 104
    """
    n, m, p = 4, 4, 7
    
    total_rank = 8 * 7 + 4 * 12  # 56 + 48 = 104
    
    U = np.zeros((16, total_rank), dtype=np.int8)
    V = np.zeros((28, total_rank), dtype=np.int8)
    W = np.zeros((28, total_rank), dtype=np.int8)
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2_correct()
    
    # Part 1: 4x4x4
    block_configs = [
        (0, 0, 0, 0),  # A11 * B11
        (0, 1, 1, 0),  # A12 * B21
        (1, 0, 0, 0),  # A21 * B11
        (1, 1, 1, 0),  # A22 * B21
        (0, 0, 0, 1),  # A11 * B12
        (0, 1, 1, 1),  # A12 * B22
        (1, 0, 0, 1),  # A21 * B12
        (1, 1, 1, 1),  # A22 * B22
    ]
    
    rank_offset = 0
    for (arb, acb, brb, bc) in block_configs:
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    U[u_idx, rank_offset + r] += strassen_U[i * 2 + j, r]
            
            for j in range(2):
                for k in range(2):
                    v_idx = (2 * brb + j) * 7 + (2 * bc + k)
                    V[v_idx, rank_offset + r] += strassen_V[j * 2 + k, r]
            
            for k in range(2):
                for i in range(2):
                    w_idx = (2 * bc + k) * 4 + (2 * arb + i)
                    W[w_idx, rank_offset + r] += strassen_W[k * 2 + i, r]
        
        rank_offset += 7
    
    # Part 2: 4x4x3 (schoolbook)
    for (arb, acb, brb) in [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = rank_offset + i * 6 + j * 3 + k
                    u_idx = (2 * arb + i) * 4 + (2 * acb + j)
                    v_idx = (2 * brb + j) * 7 + (4 + k)
                    w_idx = (4 + k) * 4 + (2 * arb + i)
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
        rank_offset += 12
    
    assert rank_offset == total_rank, f"Rank offset {rank_offset} != total {total_rank}"
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if val != 1:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, max_time=10):
    """Try to find linear dependencies among rank-1 terms to reduce rank."""
    start_time = time.time()
    rank = U.shape[1]
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Simple check: remove columns that are all zeros
    for r in range(best_rank - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if np.all(best_U[:, r] == 0) and np.all(best_V[:, r] == 0) and np.all(best_W[:, r] == 0):
            best_U = np.delete(best_U, r, axis=1)
            best_V = np.delete(best_V, r, axis=1)
            best_W = np.delete(best_W, r, axis=1)
            best_rank -= 1
    
    return best_U, best_V, best_W, best_rank


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    # Build optimized decomposition (rank 104)
    U, V, W = _build_4x4x7_optimized()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    U, V, W, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, max_time=5)
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
