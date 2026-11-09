import time
import random
import numpy as np
from scipy.optimize import minimize, linprog

DIMENSIONS = [2, 4, 5]

def _build_known_rank36():
    """Build the known rank-36 decomposition using Strassen + schoolbook blocks."""
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    S_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    S_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    S_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]
    
    SB_U = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    SB_V = [[1,0], [0,1], [1,0], [0,1]]
    SB_W = [[1,0], [1,0], [0,1], [0,1]]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base, col_start):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi][col_start + r] = S_W[r][lk * 2 + li]
        return col_start + 7

    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4

    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_schoolbook_2x2x1_block(0, 0, 4, col)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_schoolbook_2x2x1_block(0, 2, 4, col)
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that the decomposition correctly represents matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # should equal delta[i,i'] * delta[j,j'] * delta[k,k'] for the matmul tensor
    # Actually, for matmul: T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition gives: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k]) * W[k*n+i,r]
    # Wait, that's not right. Let me reconsider.
    
    # Standard tensor decomposition of matmul:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Decomposition: C[i,k] = sum_r U[i,j,r] A[i,j] V[j,k,r] B[j,k] W[i,k,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    
    # The tensor is: T[i,j,k] such that sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k]
    # Actually T is a 3-tensor where T[i,j,k] is the coefficient.
    # For matmul: C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T has T[i,j,k] = 1 if this contributes to C[i,k], else 0.
    # More precisely: T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} ... no.
    
    # The matmul tensor is: T[i,j,k] = 1 for all i,j,k in the appropriate ranges
    # No wait. Let me think again.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as: C[i,k] = sum_{i',j',k'} T[i',j',k'] A[i',j'] B[j',k'] delta_{i,i'} delta_{k,k'}
    # Hmm, that's getting complicated. Let me use the standard formulation.
    
    # The tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No, that's not right either.
    
    # Standard: The matrix multiplication tensor T has T[i,j,k] = 1 if the product A[i,j]*B[j,k]
    # contributes to C[i,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # Wait, that would make it a rank-1 tensor which is wrong.
    
    # Let me be precise. The bilinear form is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form in the sense that we can write:
    # C[i,k] = sum_{i,j,k} M[i,j,k] A[i,j] B[j,k]
    # where M[i,j,k] = 1 if i is the row index, j is the column/row index, k is the column index,
    # and the contribution goes to C[i,k]. But this isn't quite a standard tensor.
    
    # Actually, the standard way: the matmul tensor is defined such that
    # <M, A ⊗ B> = C, where C[i,k] = sum_j A[i,j] B[j,k]
    # In index notation: C[i,k] = sum_{j} M[i,j,k] A[i,j] B[j,k]
    # So M[i,j,k] = 1 for all i,j,k (since each term A[i,j]B[j,k] appears exactly once in C[i,k]).
    # But that makes M a tensor of all ones, which has rank 1! That can't be right.
    
    # I think the issue is that M depends on which output element we're computing.
    # The full tensor is: for each output (i,k), the computation is sum_j A[i,j] B[j,k].
    # So the tensor is: T[(i,k), i, j, k] ... no.
    
    # Let me use the standard formulation from tensor rank literature:
    # The matmul tensor M_{n,m,p} is an n*m*p dimensional tensor where
    # M_{i,j,k} = 1 for all i,j,k, but with the constraint that the decomposition
    # must satisfy: sum_r U[i,j,r] V[j,k,r] W[i,k,r] = delta_{i,i'} delta_{k,k'} 
    # Hmm, I'm confusing myself.
    
    # Let me just verify numerically with random matrices.
    n, m, p = 2, 4, 5
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Using the decomposition:
    # C[i,k] = sum_r W[i,k,r] * (sum_j U[i,j,r] * A[i,j]) * (sum_j V[j,k,r] * B[j,k])
    # Wait, that's not standard either. Let me check the convention.
    
    # The standard decomposition: C[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) (sum_j V[j*p+k, r] B[j,k]) W[k*n+i, r]
    
    C_decomp = np.zeros((n, p))
    for r in range(rank):
        u_vec = np.zeros((n, m))
        v_vec = np.zeros((m, p))
        for i in range(n):
            for j in range(m):
                u_vec[i, j] = U[i*m+j, r]
        for j in range(m):
            for k in range(p):
                v_vec[j, k] = V[j*p+k, r]
        w_val = W[k*n+i, r] if False else None  # This is wrong, W depends on i,k
        
        # Actually: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
        # But the sums over j are separate for U and V.
        # Let me compute it properly.
        pass
    
    # Let me redo this more carefully.
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_sum = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                v_sum = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                C_decomp[i, k] += W[k*n+i, r] * u_sum * v_sum
    
    if not np.allclose(C_decomp, C_expected, atol=tol):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s margin for 120s limit
    
    n, m, p = 2, 4, 5
    known_solution = _build_known_rank36()
    best_solution = known_solution
    best_rank = 36
    
    # We construct the decomposition using 2x2x2 Strassen blocks and 2x2x1 schoolbook blocks.
    # The problem 2x4x5 is split into two 2x2x5 sub-problems (rows of B split into two groups of 2).
    # Each 2x2x5 is split into one 2x2x2 and one 2x2x3.
    # 2x2x2 uses Strassen (rank 7).
    # 2x2x3 uses 2x2x2 (rank 7) + 2x2x1 (rank 4) = 11.
    # Total rank per 2x2x5 block: 7 + 11 = 18.
    # Total rank: 18 + 18 = 36.
    
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    # Strassen coefficients for 2x2x2
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    # r=0..6
    S_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1]  # M7
    ]
    S_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1]   # M7
    ]
    S_W = [
        [1,0,0,1],  # M1
        [0,1,0,-1], # M2
        [0,0,1,1],  # M3
        [1,1,0,0],  # M4
        [-1,0,1,0], # M5
        [0,0,0,1],  # M6
        [1,0,0,0]   # M7
    ]
    
    # Schoolbook coefficients for 2x2x1
    # U[i*2+j], V[j*2+0], W[0*2+i]
    # r=0..3
    # M1: A00 B00
    # M2: A01 B10
    # M3: A10 B00
    # M4: A11 B10
    SB_U = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]
    SB_V = [
        [1,0],
        [0,1],
        [1,0],
        [0,1]
    ]
    SB_W = [
        [1,0],
        [1,0],
        [0,1],
        [0,1]
    ]
    
    col = 0
    
    # Helper to add Strassen block
    def add_strassen_block(i_base, j_base, k_base, col_start):
        for r in range(7):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = S_U[r][li * 2 + lj]
            for lj in range(2):
                for lk in range(2):
                    gj = j_base + lj
                    gk = k_base + lk
                    V[gj * p + gk][col_start + r] = S_V[r][lj * 2 + lk]
            for lk in range(2):
                for li in range(2):
                    gk = k_base + lk
                    gi = i_base + li
                    W[gk * n + gi][col_start + r] = S_W[r][lk * 2 + li]
        return col_start + 7

    # Helper to add Schoolbook 2x2x1 block
    def add_schoolbook_2x2x1_block(i_base, j_base, k_base, col_start):
        for r in range(4):
            for li in range(2):
                for lj in range(2):
                    gi = i_base + li
                    gj = j_base + lj
                    U[gi * m + gj][col_start + r] = SB_U[r][li * 2 + lj]
            for lj in range(2):
                gj = j_base + lj
                gk = k_base
                V[gj * p + gk][col_start + r] = SB_V[r][lj]
            for li in range(2):
                gk = k_base
                gi = i_base + li
                W[gk * n + gi][col_start + r] = SB_W[r][li]
        return col_start + 4

    # We have two main blocks corresponding to j_base = 0 and j_base = 2.
    # For each j_block, we have 5 k columns.
    # We split k into k0=0,1 (Strassen) and k1=2,3,4 (Schoolbook 2x2x3).
    # Schoolbook 2x2x3 is split into 2x2x2 (Strassen) + 2x2x1 (Schoolbook).
    # So for each j_block:
    # 1. Strassen for k=0,1 (j_block, k_base=0)
    # 2. Strassen for k=2,3 (j_block, k_base=2)
    # 3. Schoolbook for k=4 (j_block, k_base=4)
    
    # Block 1: j_base = 0
    col = add_strassen_block(0, 0, 0, col) # k=0,1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_strassen_block(0, 0, 2, col) # k=2,3
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook_2x2x1_block(0, 0, 4, col) # k=4
    
    # Block 2: j_base = 2
    col = add_strassen_block(0, 2, 0, col) # k=0,1
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_strassen_block(0, 2, 2, col) # k=2,3
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    col = add_schoolbook_2x2x1_block(0, 2, 4, col) # k=4
    
    assert col == rank, f"Column mismatch: {col} vs {rank}"
    
    return {"U": U, "V": V, "W": W}
