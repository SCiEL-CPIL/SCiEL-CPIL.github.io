"""Matrix multiplication 3,4,7 - tensor decomposition with rank minimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

DEADLINE = 150.0
START_TIME = time.time()

def time_remaining():
    return DEADLINE - (time.time() - START_TIME) - 5.0

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Strassen 2x2x2 decomposition coefficients
    strassen_U = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [0, 0, 1, -1], # M6
        [0, 1, 0, -1], # M7
    ]
    strassen_V = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, -1, 0], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]
    strassen_W = [
        [1, 0, 1, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    
    # Strassen 2x2x1 decomposition coefficients
    # For 2x2x1, the tensor is (li, lj, lk=0) with 3 non-zero entries
    # Actually for 2x2x1, we need to think about it differently.
    # A 2x2x1 matrix mult: C[li,0] = sum_lj A[li,lj] * B[lj,0]
    # This is just a 2x2 vector-matrix product. The rank of this tensor is 2 (just schoolbook).
    # Wait, no. The tensor for 2x2x1 has components T[li, lj, 0] = delta_{li,lj} * ... 
    # Actually, for matrix multiplication C = A*B where A is 2x2, B is 2x1, C is 2x1.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # This is a bilinear map, not trilinear. In the tensor decomposition framework for matmul,
    # we have U[li*m+lj, r], V[lj*p+lk, r], W[lk*n+li, r].
    # For 2x2x1: li in {0,1}, lj in {0,1}, lk in {0}.
    # U has 4 entries, V has 2 entries, W has 2 entries.
    # The tensor T[li, lj, lk] = delta_{li,lk} (no wait, that's not right).
    # T[li, lj, lk] = 1 if the multiplication contributes, 0 otherwise.
    # For matmul: C[li, lk] = sum_lj A[li, lj] * B[lj, lk]
    # So T[li, lj, lk] = 1 for all li, lj, lk in their ranges.
    # For 2x2x1: T[li, lj, 0] = 1 for li in {0,1}, lj in {0,1}.
    # This is a 2x2x1 tensor with all ones.
    # Rank of all-ones 2x2x1 tensor: we can decompose as
    # M1: U=[1,0,0,0], V=[1,0], W=[1,0] -> contributes 1 to T[0,0,0]
    # M2: U=[0,0,1,0], V=[0,1], W=[0,1] -> contributes 1 to T[1,1,0]
    # But we need T[0,1,0]=1 and T[1,0,0]=1.
    # M1: U=[1,1,0,0], V=[1,0], W=[1,0] -> contributes to T[0,0,0] and T[0,1,0]
    # M2: U=[0,0,1,1], V=[0,1], W=[0,1] -> contributes to T[1,0,0] and T[1,1,0]
    # So rank 2 for 2x2x1. Same as schoolbook (2 terms). No savings.
    
    # Actually, let me reconsider. For 2x2x1, schoolbook gives 2 terms (one for each li).
    # Can we do better? The tensor is rank 2 over any field (it's a 2x2 matrix of ones, which has rank 2).
    # Wait, the tensor is T[li, lj, lk] where lk=0 only. So it's a 2x2 matrix.
    # T = [[1, 1], [1, 1]] which has rank 2. So minimum rank is 2. No savings.
    
    # Hmm, so 2x2x1 doesn't help. Let me think about what "2x2x1 Strassen-like" means.
    
    # Actually, I think the knowledge note might be referring to a different approach.
    # Let me just use the 4 Strassen 2x2x2 blocks as before, giving rank 80.
    # Wait, let me recount.
    # n=3, m=4, p=7. Total elements = 84.
    # Strassen blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    # That's 2*4*4 = 32 elements.
    # 4 blocks * 7 = 28 terms.
    # Remaining: 84 - 32 = 52 terms.
    # Total: 28 + 52 = 80.
    
    # The knowledge says a program achieved 78. How?
    # Maybe by using a 2x2x2 block for k in {4,5,6}? But k only goes up to 6, so k in {4,5,6} is 3 values, not 2.
    # Or maybe by splitting differently.
    
    # Let me try: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5} with 3 blocks of 2x2x2?
    # k in {0,1}, {2,3}, {4,5}: 3 blocks * 2 j-blocks = 6 blocks * 7 = 42 terms.
    # Covers i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5} = 2*4*6 = 48 elements.
    # Remaining: 84 - 48 = 36 terms.
    # Total: 42 + 36 = 78.
    
    # That's better! Let me use this approach.
    
    strassen_blocks = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4),
        (0, 2, 0), (0, 2, 2), (0, 2, 4),
    ]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    for i0, j0, k0 in strassen_blocks:
        for r in range(7):
            u_entries = []
            v_entries = []
            w_entries = []
            
            for li in range(2):
                for lj in range(2):
                    val = strassen_U[r][li*2+lj]
                    if val != 0:
                        u_entries.append(((i0+li)*m + (j0+lj), val))
            
            for lj in range(2):
                for lk in range(2):
                    val = strassen_V[r][lj*2+lk]
                    if val != 0:
                        v_entries.append(((j0+lj)*p + (k0+lk), val))
            
            for lk in range(2):
                for li in range(2):
                    val = strassen_W[r][lk*2+li]
                    if val != 0:
                        w_entries.append(((k0+lk)*n + (i0+li), val))
            
            U_terms.append(u_entries)
            V_terms.append(v_entries)
            W_terms.append(w_entries)
    
    strassen_covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    strassen_covered.add((i0+di, j0+dj, k0+dk))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_covered:
                    U_terms.append([(i*m + j, 1)])
                    V_terms.append([(j*p + k, 1)])
                    W_terms.append([(k*n + i, 1)])
    
    rank = len(U_terms)
    
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    for r in range(rank):
        for idx, val in U_terms[r]:
            U[idx, r] = val
        for idx, val in V_terms[r]:
            V[idx, r] = val
        for idx, val in W_terms[r]:
            W[idx, r] = val
    
    # Verify correctness
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    computed = 0
                    for r in range(rank):
                        computed += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if computed != expected:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        U = np.zeros((n*m, n*m*p), dtype=np.int64)
        V = np.zeros((m*p, n*m*p), dtype=np.int64)
        W = np.zeros((p*n, n*m*p), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
