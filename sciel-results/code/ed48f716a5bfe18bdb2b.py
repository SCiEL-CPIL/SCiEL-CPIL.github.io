import time
import numpy as np
from fractions import Fraction

def _construct_schoolbook():
    """
    Construct the standard schoolbook decomposition for 2x3 * 3x10.
    Rank = 60.
    U shape: (6, 60)
    V shape: (30, 60)
    W shape: (20, 60)
    
    Indices:
    A (2x3): idx = i*3 + j
    B (3x10): idx = j*10 + k
    C (2x10): idx = i*10 + k (but W uses k*n+i convention)
    """
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i  # Convention: W[k*n+i, r]
                
                U[idx_a, t] = 1
                V[idx_b, t] = 1
                W[idx_c, t] = 1
                t += 1
                
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify if U, V, W (numpy arrays) form a valid decomposition.
    Returns True if exact.
    """
    n, m, p = 2, 3, 10
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = U.shape[1]
    
    # Check all 1-entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Sum_t U[a,t] V[b,t] W[c,t]
                # U[a] is vector of shape (rank,)
                # V[b] is vector of shape (rank,)
                # W[c] is vector of shape (rank,)
                # sum is dot(U[a], V[b] * W[c])
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                
                # Check if close to 1
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                    
    # Check a few 0-entries to ensure no spurious terms
    # The tensor is sparse.
    # Total entries 6*30*20 = 3600.
    # Non-zero entries: 60.
    # We can check random zeros or a systematic subset.
    # Let's check all entries where a=(i,j), b=(j',k), c=(k',i') 
    # such that j != j' OR k != k' (mismatch in j or k)
    # Actually, the support is determined by j_A == j_B and k_B == k_C and i_A == i_C.
    
    # Efficient check:
    # Compute T_recon and T_true?
    # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    # T_true is zeros except at (i*m+j, j*p+k, k*n+i) = 1.
    
    T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    
    # Check zeros
    # For each a, b, c
    # If (a, b, c) is not in support, T_recon[a,b,c] must be 0.
    
    # Support:
    # i in 0..1, j in 0..2, k in 0..9
    # a = i*3+j, b = j*10+k, c = k*2+i
    
    # Let's just check all a, b where j_A != j_B.
    # a = i*3+j => j_A = a % 3
    # b = j'*10+k => j_B = b // 10
    
    for a in range(dim_u):
        j_a = a % 3
        for b in range(dim_v):
            j_b = b // 10
            if j_a != j_b:
                # For any c, it should be 0
                # Check max abs in T_recon[a, b, :]
                if not np.allclose(T_recon[a, b, :], 0.0, atol=1e-9):
                    return False
    
    # Check i_A != i_C
    # c = k*2+i => i_C = c % 2
    # a = i*3+j => i_A = a // 3
    for a in range(dim_u):
        i_a = a // 3
        for c in range(dim_w):
            i_c = c % 2
            if i_a != i_c:
                if not np.allclose(T_recon[:, c, :][:, 0], 0.0, atol=1e-9): # This is wrong, T_recon is 3D
                    # T_recon[a, :, c] should be 0? No.
                    # If i_A != i_C, then for any b, T_recon[a, b, c] should be 0.
                    pass
    
    # Correct zero check:
    for a in range(dim_u):
        i_a = a // 3
        j_a = a % 3
        for b in range(dim_v):
            j_b = b // 10
            k_b = b % 10
            if j_a != j_b:
                continue # Already checked or skip
            for c in range(dim_w):
                i_c = c % 2
                k_c = c // 2
                if i_a != i_c or k_b != k_c:
                    if not np.isclose(T_recon[a, b, c], 0.0, atol=1e-9):
                        return False
                        
    return True

def _strassen_2x2_block(A1, A2, B1, B2):
    """
    Compute 2x2 matrix multiplication using Strassen.
    A1, A2, B1, B2 are 2x2 matrices.
    Returns C1, C2 (2x2 matrices).
    C = [[C1, C2]]
    C1 = A1*B1, C2 = A1*B2? No.
    Let A = [[A1, A2]], B = [[B1], [B2]].
    C = A*B = [[A1*B1 + A2*B2]]
    Strassen for 2x2 blocks:
    M1 = (A1+A2)(B1+B2)
    M2 = A2(B1-B2)
    C1 = M1 - M2
    C2 = A1(B1-B2) ... wait.
    
    Standard Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    C11 = M1 - M4
    C12 = M4 + M5
    C21 = M3 + M6
    C22 = M1 - M2
    
    Here A is 1x2, B is 2x1.
    A = [[A1, A2]], B = [[B1], [B2]]
    C = [[A1*B1 + A2*B2]]
    
    Let's embed in 2x2:
    A_full = [[A1, A2], [0, 0]]
    B_full = [[B1, 0], [B2, 0]]
    C_full = A_full * B_full = [[A1*B1 + A2*B2, 0], [0, 0]]
    
    Strassen M's:
    M1 = (A1+0)(B1+B2) = A1(B1+B2)
    M2 = (A2+0)(B1-B2) = A2(B1-B2)
    M3 = (A1+A2)(B1+0) = (A1+A2)B1
    M4 = (0+0)(B2+0) = 0
    M5 = (A1+0)(0+B2) = A1*B2
    M6 = (0-A2)(B1+0) = -A2*B1
    M7 = (A2+0)(B2-B1) = A2(B2-B1)
    
    C11 = M1 - M4 = A1(B1+B2)
    C12 = M4 + M5 = A1*B2
    C21 = M3 + M6 = (A1+A2)B1 - A2*B1 = A1*B1
    C22 = M1 - M2 = A1(B1+B2) - A2(B1-B2) = A1*B1 + A1*B2 - A2*B1 + A2*B2
    
    This doesn't seem to reduce operations for 1x2 * 2x1 directly in a simple way if we just want the scalar sum.
    Wait, the output is a 2x1 vector? No, A is 2x3, B is 3x10.
    
    Let's group columns of B.
    B is 3x10. Group into 5 pairs of columns.
    B = [B1, B2, B3, B4, B5] where each Bi is 3x2.
    A is 2x3.
    C = A * B = [A*B1, A*B2, A*B3, A*B4, A*B5].
    Each A*Bi is 2x2.
    
    We can use Strassen to compute each 2x2 product A * Bi?
    No, A is 2x3. Bi is 3x2.
    This is not square.
    
    However, we can split A rows and Bi columns?
    Let's look at the inspiration: "Exploit the linearity in the output dimension by grouping the 10 columns of B into five pairs and applying a 2x2 Strassen decomposition to each pair".
    
    If we treat the pair of columns as a 2x1 vector of 2x1 column vectors?
    Let B_pair = [b_1, b_2] where b_1, b_2 are column vectors of size 3.
    C_pair = [c_1, c_2] where c_1 = A*b_1, c_2 = A*b_2.
    c_1, c_2 are vectors of size 2.
    
    So we want to compute two linear combinations of the same matrix A.
    This is equivalent to multiplying A (2x3) by B_pair (3x2).
    
    Can we use Strassen-like decomposition for 2x3 * 3x2?
    Standard rank is 2*3*2 = 12.
    Strassen for 2x2 is 7.
    Is there a 10-rank or 9-rank decomposition for 2x3 x 3x2?
    
    If we can find a decomposition of rank R for 2x3 x 3x2, then total rank would be 5*R.
    If R=10, total rank 50.
    If R=9, total rank 45.
    If R=8, total rank 40.
    
    Let's try to find a low rank decomposition for 2x3 x 3x2.
    
    """
    pass

def _decompose_2x3x3x2():
    """
    Find a decomposition for 2x3 * 3x2.
    U shape (6, r), V shape (6, r), W shape (4, r).
    """
    n, m, p = 2, 3, 2
    # Try ranks from 12 down to 8
    # Use random initialization and gradient descent?
    # Or known decompositions?
    
    # Let's try a random search for integer/half-integer solutions.
    # Space is small.
    # U: 6*r, V: 6*r, W: 4*r.
    # r=10 => 100 vars.
    
    # We'll stick to the schoolbook for this block if we can't find better quickly.
    # Schoolbook rank 12.
    
    def _construct_block_schoolbook():
        rank = 12
        U = np.zeros((6, rank))
        V = np.zeros((6, rank))
        W = np.zeros((4, rank))
        
        t = 0
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    idx_a = i * 3 + j
                    idx_b = j * 2 + k
                    idx_c = k * 2 + i
                    U[idx_a, t] = 1
                    V[idx_b, t] = 1
                    W[idx_c, t] = 1
                    t += 1
        return U, V, W
    
    # Try to find rank 10 or 11?
    # Given time constraints, I will return the schoolbook for the block.
    # The total rank will be 5 * 12 = 60. No improvement.
    
    # To improve, I need a non-trivial decomposition.
    # I will attempt a simple random integer search for rank 10.
    
    def _try_find_block_decomp(rank, time_limit=2.0):
        start = time.time()
        np.random.seed(42)
        
        def verify(U, V, W):
            # Check all 1s
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        a = i*3+j
                        b = j*2+k
                        c = k*2+i
                        val = np.dot(U[a], V[b]*W[c])
                        if not np.isclose(val, 1.0, atol=1e-6):
                            return False
            # Check zeros
            # a=(i,j), b=(j',k), c=(k',i')
            # j!=j' or k!=k' or i!=i'
            for i in range(2):
                for j in range(3):
                    for jp in range(3):
                        if jp == j: continue
                        for k in range(2):
                            for kp in range(2):
                                if kp == k: continue
                                a = i*3+j
                                b = jp*2+k
                                c = kp*2+i
                                val = np.dot(U[a], V[b]*W[c])
                                if not np.isclose(val, 0.0, atol=1e-6):
                                    return False
                            for i2 in range(2):
                                if i2 == i: continue
                                for kp in range(2):
                                    if kp == k: continue
                                    a = i*3+j
                                    b = jp*2+k
                                    c = kp*2+i2
                                    val = np.dot(U[a], V[b]*W[c])
                                    if not np.isclose(val, 0.0, atol=1e-6):
                                        return False
            return True

        # Random restarts
        for _ in range(50):
            if time.time() - start > time_limit:
                break
            U = np.random.randint(-2, 3, size=(6, rank))
            V = np.random.randint(-2, 3, size=(6, rank))
            W = np.random.randint(-2, 3, size=(4, rank))
            if verify(U, V, W):
                return U, V, W
        return None
    
    # Try rank 10
    res = _try_find_block_decomp(10, time_limit=3.0)
    if res:
        return res
    
    # Try rank 11
    res = _try_find_block_decomp(11, time_limit=3.0)
    if res:
        return res
        
    # Fallback to schoolbook
    return _construct_block_schoolbook()

def _construct_strassen_like():
    """
    Construct the full 2x3x3x10 decomposition using 5 blocks of 2x3x3x2.
    """
    n, m, p = 2, 3, 10
    p_blocks = p // 2 # 5 blocks
    block_size = 2
    
    # Get one block decomposition
    U_block, V_block, W_block = _decompose_2x3x3x2()
    r_block = U_block.shape[1]
    
    # Total rank
    rank = r_block * p_blocks
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Fill in blocks
    # B is 3x10. Columns 0,1 are block 0. 2,3 block 1. etc.
    # A is 2x3. Same for all blocks.
    # C is 2x10. Columns 0,1 are block 0. etc.
    
    for b_idx in range(p_blocks):
        # Block index in V: columns 2*b_idx and 2*b_idx+1
        # Block index in W: columns 2*b_idx and 2*b_idx+1 (since C is 2x10, k is column)
        # W index: k*n + i.
        # For block b_idx, k is 2*b_idx and 2*b_idx+1.
        
        rank_offset = b_idx * r_block
        
        # U part: same for all blocks, but we need to replicate it for each block?
        # No, U is shared?
        # In the decomposition T = sum_t U_t V_t W_t.
        # If we use independent blocks, the sum is disjoint.
        # So we just place the block decomposition in the corresponding columns of U, V, W.
        # But U is only indexed by A (2x3). It doesn't depend on the block.
        # However, the tensor terms are (i, j, k).
        # If we decompose the block for k=2*b_idx, 2*b_idx+1, the terms are (i, j, k).
        # U[i,j, t] is the same for all k?
        # In the schoolbook, U[i,j,t] is 1 for specific t.
        # In the block decomposition, U_block[i,j,t_local] is defined.
        # We need to map t_local to global t.
        # t_global = rank_offset + t_local.
        # U[i,j, t_global] = U_block[i,j, t_local].
        
        for i in range(n):
            for j in range(m):
                idx_a = i*m + j
                for t_local in range(r_block):
                    idx_t = rank_offset + t_local
                    U[idx_a, idx_t] = U_block[idx_a, t_local]
                    
        for j in range(m):
            for k_local in range(block_size):
                k_global = 2*b_idx + k_local
                idx_b = j*p + k_global
                for t_local in range(r_block):
                    idx_t = rank_offset + t_local
                    # V_block index for (j, k_local)
                    idx_b_block = j*block_size + k_local
                    V[idx_b, idx_t] = V_block[idx_b_block, t_local]
                    
        for k_local in range(block_size):
            k_global = 2*b_idx + k_local
            for i in range(n):
                idx_c = k_global*n + i
                for t_local in range(r_block):
                    idx_t = rank_offset + t_local
                    # W_block index for (k_local, i)
                    idx_c_block = k_local*n + i
                    W[idx_c, idx_t] = W_block[idx_c_block, t_local]
                    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140
    
    # Fallback
    schoolbook = _construct_schoolbook()
    best_result = {
        "U": schoolbook["U"].tolist(),
        "V": schoolbook["V"].tolist(),
        "W": schoolbook["W"].tolist()
    }
    best_rank = 60
    
    # Try Strassen-like
    try:
        strassen_like = _construct_strassen_like()
        rank_sl = strassen_like["U"].shape[1]
        
        # Verify
        U_sl = np.array(strassen_like["U"])
        V_sl = np.array(strassen_like["V"])
        W_sl = np.array(strassen_like["W"])
        
        if _verify_decomposition(U_sl, V_sl, W_sl):
            if rank_sl < best_rank:
                best_result = {
                    "U": strassen_like["U"].tolist(),
                    "V": strassen_like["V"].tolist(),
                    "W": strassen_like["W"].tolist()
                }
                best_rank = rank_sl
    except Exception as e:
        pass
    
    # If we found 60, try to improve with global optimization?
    # Unlikely to beat 60 easily without specific structure.
    # The block approach gives 60 if block is 12.
    # If block is 10, it gives 50.
    
    return best_result
