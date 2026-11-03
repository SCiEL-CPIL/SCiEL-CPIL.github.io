"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor contraction via einsum."""
    rank = len(U[0]) if len(U) > 0 else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape: U is (n*m, rank) -> (n, m, rank)
    # V is (m*p, rank) -> (m, p, rank)
    # W is (p*n, rank) -> (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want T[i,j,k] == 1 for all i,j,k
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    return np.allclose(T, 1.0, atol=tol)

def _verify_sparse(U, V, W, n, m, p, tol=1e-9):
    """Verification that works with sparse-like matrices (lists of lists)."""
    rank = len(U[0]) if len(U) > 0 else 0
    if rank == 0:
        return False
    
    # Build tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    T = [[[0.0]*p for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                T[i][j][k] = s
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if abs(T[i][j][k] - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _strassen_2x2():
    """
    Return Strassen decomposition for <2,2,2> with rank 7.
    U (4x7), V (4x7), W (4x7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
    V = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
    W = [[0]*7 for _ in range(4)]  # 2x2 = 4 rows
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11=0, A22=3, B11=0, B22=3
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = -1  # C11 += M1, C22 -= M1... wait
    # Actually W[k*2+i, r] where k is col of C, i is row of C
    # C11: k=0,i=0 -> W[0][0]
    # C12: k=1,i=0 -> W[2][0]
    # C21: k=0,i=1 -> W[1][0]
    # C22: k=1,i=1 -> W[3][0]
    
    # Let me redo this more carefully.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # W index: k*n + i = k*2 + i
    # C[0,0]: W[0*2+0] = W[0]
    # C[0,1]: W[1*2+0] = W[2]
    # C[1,0]: W[0*2+1] = W[1]
    # C[1,1]: W[1*2+1] = W[3]
    
    # M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,1])
    # Contributes to: C[0,0] += M1, C[1,1] -= M1 (from C22 = M1 - M2 + M3 + M6, but M1 has +1)
    # Wait, C22 = M1 - M2 + M3 + M6, so M1 contributes +1 to C22.
    # C11 = M1 + M4 - M5 + M7, so M1 contributes +1 to C11.
    U[0][0] = 1; U[3][0] = 1  # A[0,0], A[1,1]
    V[0][0] = 1; V[3][0] = 1  # B[0,0], B[1,1]
    W[0][0] = 1; W[3][0] = 1  # C[0,0], C[1,1]
    
    # M2 = (A[1,0] + A[1,1]) B[0,0]
    U[2][1] = 1; U[3][1] = 1  # A[1,0], A[1,1]
    V[0][1] = 1  # B[0,0]
    W[1][1] = 1; W[3][1] = -1  # C[1,0] += M2, C[1,1] -= M2
    
    # M3 = A[0,0] (B[0,1] - B[1,1])
    U[0][2] = 1  # A[0,0]
    V[1][2] = 1; V[3][2] = -1  # B[0,1], -B[1,1]
    W[2][2] = 1; W[3][2] = 1  # C[0,1] += M3, C[1,1] += M3
    
    # M4 = A[1,1] (B[1,0] - B[0,0])
    U[3][3] = 1  # A[1,1]
    V[2][3] = 1; V[0][3] = -1  # B[1,0], -B[0,0]
    W[0][3] = 1; W[1][3] = 1  # C[0,0] += M4, C[1,0] += M4
    
    # M5 = (A[0,0] + A[0,1]) B[1,1]
    U[0][4] = 1; U[1][4] = 1  # A[0,0], A[0,1]
    V[3][4] = 1  # B[1,1]
    W[0][4] = -1; W[2][4] = 1  # C[0,0] -= M5, C[0,1] += M5
    
    # M6 = (A[1,0] - A[0,0]) (B[0,0] + B[0,1])
    U[2][5] = 1; U[0][5] = -1  # A[1,0], -A[0,0]
    V[0][5] = 1; V[1][5] = 1  # B[0,0], B[0,1]
    W[3][5] = 1  # C[1,1] += M6
    
    # M7 = (A[0,1] - A[1,1]) (B[1,0] + B[1,1])
    U[1][6] = 1; U[3][6] = -1  # A[0,1], -A[1,1]
    V[2][6] = 1; V[3][6] = 1  # B[1,0], B[1,1]
    W[0][6] = 1  # C[0,0] += M7
    
    return {"U": U, "V": V, "W": W}

def _construct_block_decomposition(n, m, p, block_n, block_m, block_p, block_decomp, time_limit=100):
    """
    Construct full decomposition by tiling with block decomposition.
    n, m, p: full dimensions
    block_n, block_m, block_p: block dimensions
    block_decomp: dict with U, V, W for <block_n, block_m, block_p>
    """
    if time.time() > time_limit:
        return None
    
    # Number of blocks in each dimension
    n_blocks = n // block_n
    m_blocks = m // block_m
    p_blocks = p // block_p
    
    # Check if dimensions divide evenly
    if n_blocks * block_n != n or m_blocks * block_m != m or p_blocks * block_p != p:
        return None
    
    total_blocks = n_blocks * m_blocks * p_blocks
    block_rank = len(block_decomp["U"][0])
    total_rank = total_blocks * block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    block_U = block_decomp["U"]
    block_V = block_decomp["V"]
    block_W = block_decomp["W"]
    
    bi = 0
    for ib in range(n_blocks):
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                offset = bi * block_rank
                
                # U block: rows (ib*block_n + i) * m + (jb*block_m + j)
                for i in range(block_n):
                    for j in range(block_m):
                        u_idx_local = i * block_m + j
                        u_idx_full = (ib * block_n + i) * m + (jb * block_m + j)
                        for c in range(block_rank):
                            val = block_U[u_idx_local][c]
                            if val != 0:
                                U[u_idx_full][offset + c] += val
                
                # V block: rows (jb*block_m + j) * p + (kb*block_p + k)
                for j in range(block_m):
                    for k in range(block_p):
                        v_idx_local = j * block_p + k
                        v_idx_full = (jb * block_m + j) * p + (kb * block_p + k)
                        for c in range(block_rank):
                            val = block_V[v_idx_local][c]
                            if val != 0:
                                V[v_idx_full][offset + c] += val
                
                # W block: rows (kb*block_p + k) * n + (ib*block_n + i)
                for k in range(block_p):
                    for i in range(block_n):
                        w_idx_local = k * block_n + i
                        w_idx_full = (kb * block_p + k) * n + (ib * block_n + i)
                        for c in range(block_rank):
                            val = block_W[w_idx_local][c]
                            if val != 0:
                                W[w_idx_full][offset + c] += val
                
                bi += 1
    
    return {"U": U, "V": V, "W": W}

def _search_rank_improved(n_sub, m_sub, p_sub, target_rank, num_trials=500, seed=0, time_limit=50):
    """
    Improved random search for decomposition of <n_sub, m_sub, p_sub> with target_rank.
    Uses better initialization strategies.
    """
    rng = random.Random(seed)
    U_size = n_sub * m_sub
    V_size = m_sub * p_sub
    W_size = p_sub * n_sub
    
    # Strategy 1: Random integers in [-1, 1]
    for trial in range(num_trials):
        if time.time() > time_limit:
            break
        
        U = [[rng.randint(-1, 1) for _ in range(target_rank)] for _ in range(U_size)]
        V = [[rng.randint(-1, 1) for _ in range(target_rank)] for _ in range(V_size)]
        W = [[rng.randint(-1, 1) for _ in range(target_rank)] for _ in range(W_size)]
        
        if _verify_fast(U, V, W, n_sub, m_sub, p_sub, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Sparse random - each row has 1-2 non-zero entries
    for trial in range(num_trials):
        if time.time() > time_limit:
            break
        
        def _sparse_row(size, rank, rng):
            row = [0] * rank
            num_nonzero = rng.randint(1, min(2, rank))
            positions = rng.sample(range(rank), num_nonzero)
            for pos in positions:
                row[pos] = rng.choice([-1, 1])
            return row
        
        U = [_sparse_row(U_size, target_rank, rng) for _ in range(U_size)]
        V = [_sparse_row(V_size, target_rank, rng) for _ in range(V_size)]
        W = [_sparse_row(W_size, target_rank, rng) for _ in range(W_size)]
        
        if _verify_fast(U, V, W, n_sub, m_sub, p_sub, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    
    # Strategy 3: Based on schoolbook with random perturbations
    for trial in range(num_trials // 2):
        if time.time() > time_limit:
            break
        
        # Start from schoolbook and try to merge columns
        U = [[0] * target_rank for _ in range(U_size)]
        V = [[0] * target_rank for _ in range(V_size)]
        W = [[0] * target_rank for _ in range(W_size)]
        
        # Assign random columns
        col_assignments = {}
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    col = rng.randint(0, target_rank - 1)
                    col_assignments[(i, j, k)] = col
        
        for (i, j, k), col in col_assignments.items():
            U[i * m_sub + j][col] += 1
            V[j * p_sub + k][col] += 1
            W[k * n_sub + i][col] += 1
        
        if _verify_fast(U, V, W, n_sub, m_sub, p_sub, tol=1e-9):
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)
    
    # Baseline: Schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Verify baseline
    if not _verify_fast(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    
    # Strategy 1: Try Strassen-based block decomposition
    # For 3x4x4, we can't directly use 2x2 blocks for all dimensions since 3 is not divisible by 2.
    # But we can try:
    # Option A: Treat as three 1x4x4 multiplications? No, that's not how it works.
    # Option B: Split the problem differently.
    
    # Let's try to find a decomposition for <3,2,2> first.
    # If R(3,2,2) = r, then R(3,4,4) <= 4 * r (by splitting m=4 into 2+2 and p=4 into 2+2).
    
    # First, try to use known results or search for <3,2,2>
    found_322 = None
    for r in range(11, 5, -1):
        if time.time() > deadline - 20:
            break
        result = _search_rank_improved(3, 2, 2, r, num_trials=200, seed=r*137, time_limit=10)
        if result:
            found_322 = result
            break
    
    if found_322:
        sub_rank = len(found_322["U"][0])
        total_rank = 4 * sub_rank
        if total_rank < best_rank:
            # Construct full decomposition
            # We need to map <3,2,2> to the full <3,4,4>
            # Split m=4 into two 2s, p=4 into two 2s
            # 4 blocks: (m_block, p_block) in {(0,0), (0,1), (1,0), (1,1)}
            
            U = [[0] * total_rank for _ in range(12)]
            V = [[0] * total_rank for _ in range(16)]
            W = [[0] * total_rank for _ in range(12)]
            
            for m_block in range(2):
                for p_block in range(2):
                    offset = (m_block * 2 + p_block) * sub_rank
                    a_col_start = m_block * 2
                    b_row_start = m_block * 2
                    c_col_start = p_block * 2
                    
                    for i in range(3):
                        for j in range(2):
                            u_idx_local = i * 2 + j
                            u_idx_full = i * 4 + (a_col_start + j)
                            for c in range(sub_rank):
                                val = found_322["U"][u_idx_local][c]
                                if val != 0:
                                    U[u_idx_full][offset + c] += val
                    
                    for j in range(2):
                        for k in range(2):
                            v_idx_local = j * 2 + k
                            v_idx_full = (b_row_start + j) * 4 + (c_col_start + k)
                            for c in range(sub_rank):
                                val = found_322["V"][v_idx_local][c]
                                if val != 0:
                                    V[v_idx_full][offset + c] += val
                    
                    for k in range(2):
                        for i in range(3):
                            w_idx_local = k * 3 + i
                            w_idx_full = (c_col_start + k) * 3 + i
                            for c in range(sub_rank):
                                val = found_322["W"][w_idx_local][c]
                                if val != 0:
                                    W[w_idx_full][offset + c] += val
            
            if _verify_fast(U, V, W, 3, 4, 4):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = total_rank
    
    # Strategy 2: Try direct random search for <3,4,4> with rank < 48
    # This is expensive but might find something
    if time.time() < deadline - 30:
        for r in range(47, 39, -1):
            if time.time() > deadline - 10:
                break
            result = _search_rank_improved(3, 4, 4, r, num_trials=100, seed=r*7, time_limit=8)
            if result:
                if _verify_fast(result["U"], result["V"], result["W"], 3, 4, 4):
                    best_result = result
                    best_rank = r
                    break
    
    # Strategy 3: Try to use Strassen for a 2x2 sub-block and schoolbook for the rest
    # This is complex, so we'll skip for now
    
    return best_result
# EVOLVE-BLOCK-END
