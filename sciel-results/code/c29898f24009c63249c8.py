import numpy as np
import time

DIMENSIONS = [5, 5, 5]
_START_TIME = time.time()
_TIMEOUT = 145  # safety margin before 150s limit

def _get_2x2_factors():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # P2: (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    # P3: A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1
    # P4: A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=-1; W[1,3]=1
    # P5: (A00+A10)B11
    U[0,4]=1; U[2,4]=1
    V[3,4]=1
    W[2,4]=1; W[3,4]=1
    # P6: (A10-A00)(B00+B01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1
    # P7: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=-1
    
    return U, V, W

def _get_2x1_factors():
    """Return U, V, W for 2x1x1 multiplication (rank 2).
    A is 2x1, B is 1x1, C is 2x1.
    C_i = A_i * B_0 for i=0,1.
    """
    # U: shape (2*1, rank) = (2, 2)
    # V: shape (1*1, rank) = (1, 2)
    # W: shape (1*2, rank) = (2, 2)
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2))
    
    # r=0: C_0 = A_0 * B_0
    U[0, 0] = 1
    V[0, 0] = 1
    W[0, 0] = 1  # W index: k*n+i = 0*2+0 = 0
    
    # r=1: C_1 = A_1 * B_0
    U[1, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # W index: k*n+i = 0*2+1 = 1
    
    return U, V, W

def _get_1x2_factors():
    """Return U, V, W for 1x2x2 multiplication (rank 1).
    A is 1x2, B is 2x2, C is 1x2.
    C_j = sum_i A_0i * B_ij. This is rank 1?
    Actually, C_0 = A_00*B_00 + A_01*B_10
    C_1 = A_00*B_01 + A_01*B_11
    This is a matrix product of (1x2) * (2x2) = 1x2.
    The tensor has indices (i,j,k) -> (0, i, j) for A, (i,k) for B, (j) for C?
    Let's use the standard form: C_j = sum_i A_i B_ij.
    U shape: (1*2, rank) = (2, rank)
    V shape: (2*2, rank) = (4, rank)
    W shape: (2*1, rank) = (2, rank)
    
    Rank 1 decomposition:
    r=0:
    U: A_0, A_1 -> U[0,0]=1, U[1,0]=1? No.
    C_j = A_0 * B_0j + A_1 * B_1j.
    This is sum over i.
    Term 1: A_0 * B_00 for C_0. U[0,0]=1, V[0,0]=1, W[0,0]=1.
    Term 2: A_1 * B_10 for C_0. U[1,1]=1, V[2,1]=1, W[0,1]=1.
    Term 3: A_0 * B_01 for C_1. U[0,2]=1, V[1,2]=1, W[1,2]=1.
    Term 4: A_1 * B_11 for C_1. U[1,3]=1, V[3,3]=1, W[1,3]=1.
    Wait, can we do better?
    C = A B. A is 1x2, B is 2x2.
    C_0 = A_0 B_00 + A_1 B_10
    C_1 = A_0 B_01 + A_1 B_11
    
    Is there a rank 2 decomposition?
    r=0: A_0 * B_0j. C_j gets A_0 B_0j.
    r=1: A_1 * B_1j. C_j gets A_1 B_1j.
    Yes! Rank 2.
    
    U: (2, 2). V: (4, 2). W: (2, 2).
    r=0:
    U[0,0]=1 (A_0). U[1,0]=0.
    V[0,0]=1 (B_00), V[1,0]=1 (B_01). V[2,0]=0, V[3,0]=0.
    W[0,0]=1 (C_0), W[1,0]=1 (C_1).
    r=1:
    U[0,1]=0. U[1,1]=1 (A_1).
    V[0,1]=0, V[1,1]=0. V[2,1]=1 (B_10), V[3,1]=1 (B_11).
    W[0,1]=1 (C_0), W[1,1]=1 (C_1).
    """
    U = np.zeros((2, 2))
    V = np.zeros((4, 2))
    W = np.zeros((2, 2))
    
    U[0, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1
    
    U[1, 1] = 1
    V[2, 1] = 1; V[3, 1] = 1
    W[0, 1] = 1; W[1, 1] = 1
    
    return U, V, W

def _get_1x1_factors():
    """Return U, V, W for 1x1x1 multiplication (rank 1)."""
    U = np.array([[1.0]])
    V = np.array([[1.0]])
    W = np.array([[1.0]])
    return U, V, W

def _get_2x2x1_factors():
    """Return U, V, W for 2x2x1 multiplication (rank 2).
    A is 2x2, B is 2x1, C is 2x1.
    C_i = sum_j A_ij B_j.
    C_0 = A_00 B_0 + A_01 B_1
    C_1 = A_10 B_0 + A_11 B_1
    
    Rank 2:
    r=0: B_0 * A_i0.
    r=1: B_1 * A_i1.
    """
    U = np.zeros((4, 2))
    V = np.zeros((2, 2))
    W = np.zeros((2, 2))
    
    # r=0: B_0 * A_i0
    U[0, 0] = 1 # A_00
    U[2, 0] = 1 # A_10
    V[0, 0] = 1 # B_0
    W[0, 0] = 1 # C_0
    W[1, 0] = 1 # C_1
    
    # r=1: B_1 * A_i1
    U[1, 1] = 1 # A_01
    U[3, 1] = 1 # A_11
    V[1, 1] = 1 # B_1
    W[0, 1] = 1 # C_0
    W[1, 1] = 1 # C_1
    
    return U, V, W

def _get_1x1x2_factors():
    """Return U, V, W for 1x1x2 multiplication (rank 1).
    A is 1x1, B is 1x2, C is 1x2.
    C_j = A_0 B_0j.
    Rank 1: A_0 * B_0j.
    """
    U = np.array([[1.0]])
    V = np.zeros((2, 1))
    W = np.zeros((2, 1))
    
    V[0, 0] = 1 # B_00
    V[1, 0] = 1 # B_01
    W[0, 0] = 1 # C_0
    W[1, 0] = 1 # C_1
    
    return U, V, W

def _block_mult_factors(n, m, p):
    """
    Get factors for n x m x p multiplication using recursive block decomposition.
    Returns U, V, W arrays.
    """
    if n == 1 and m == 1 and p == 1:
        return _get_1x1_factors()
    if n == 2 and m == 2 and p == 2:
        return _get_2x2_factors()
    if n == 2 and m == 1 and p == 1:
        return _get_2x1_factors()
    if n == 1 and m == 2 and p == 2:
        return _get_1x2_factors()
    if n == 2 and m == 2 and p == 1:
        return _get_2x2x1_factors()
    if n == 1 and m == 1 and p == 2:
        return _get_1x1x2_factors()
    
    # Recursive case for 5x5x5
    # Partition: n=5=2+2+1, m=5=2+2+1, p=5=2+2+1
    # This creates a 3x3x3 grid of blocks.
    # Block sizes:
    # i=0,1: n_size=2
    # i=2: n_size=1
    # Same for m, p.
    
    # We will use a general recursive block multiplication.
    # However, for 5x5x5, it's easier to just hardcode the 3x3x3 block structure
    # and use the appropriate sub-factors.
    
    # Let's define the block structure for 5x5x5.
    # A is 5x5. Blocks A_ij are n_i x m_j.
    # B is 5x5. Blocks B_jk are m_j x p_k.
    # C is 5x5. Blocks C_ik are n_i x p_k.
    
    # Sizes:
    # n_sizes = [2, 2, 1]
    # m_sizes = [2, 2, 1]
    # p_sizes = [2, 2, 1]
    
    # For each block (i, j, k), we compute C_ik += A_ij B_jk.
    # We can accumulate the factors for each (i, j, k) term.
    
    n_sizes = [2, 2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 1]
    
    # Precompute factors for each unique block size combination
    # (n_i, m_j, p_k)
    # Unique sizes:
    # (2,2,2) -> Strassen 7
    # (2,2,1) -> 2x2x1 rank 2
    # (2,1,2) -> 2x1x2? Not defined yet.
    # (2,1,1) -> 2x1x1 rank 2
    # (1,2,2) -> 1x2x2 rank 2
    # (1,2,1) -> 1x2x1? Not defined yet.
    # (1,1,2) -> 1x1x2 rank 1
    # (1,1,1) -> 1x1x1 rank 1
    
    # We need to define the missing ones.
    
    # 2x1x2:
    # A is 2x1, B is 1x2, C is 2x2.
    # C_ij = A_i B_j.
    # Rank 1?
    # C_00 = A_0 B_0
    # C_01 = A_0 B_1
    # C_10 = A_1 B_0
    # C_11 = A_1 B_1
    # This is the outer product of A and B^T.
    # Rank 1 decomposition:
    # U: A. V: B^T. W: identity?
    # Let's check.
    # C_ij = U_i r V_j W_k? No.
    # C_ij = sum_r U_ir V_rj W_ki?
    # Indices: A(i,j), B(j,k), C(k,i)? No, C is 2x2.
    # Standard: C(i,k) = sum_j A(i,j) B(j,k).
    # Here A is 2x1 (i=0,1; j=0). B is 1x2 (j=0; k=0,1). C is 2x2 (i=0,1; k=0,1).
    # C(i,k) = A(i,0) B(0,k).
    # This is rank 1.
    # U: shape (2*1, 1) = (2, 1). U[i,0] = 1 for all i? No, A(i,0).
    # So U[i,0] corresponds to A(i,0).
    # V: shape (1*2, 1) = (2, 1). V[k,0] corresponds to B(0,k).
    # W: shape (2*2, 1) = (4, 1). W[k*2+i, 0] corresponds to C(i,k).
    # W[k*2+i, 0] = 1.
    
    def _get_2x1x2_factors():
        U = np.ones((2, 1))
        V = np.ones((2, 1))
        W = np.ones((4, 1))
        return U, V, W
    
    # 1x2x1:
    # A is 1x2, B is 2x1, C is 1x1.
    # C(0,0) = sum_j A(0,j) B(j,0).
    # This is a dot product.
    # Rank 2?
    # C = A_0 B_0 + A_1 B_1.
    # Rank 2.
    # U: shape (1*2, 2) = (2, 2).
    # V: shape (2*1, 2) = (2, 2).
    # W: shape (1*1, 2) = (1, 2).
    
    def _get_1x2x1_factors():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W
    
    # Now we have all base cases.
    # Let's build the 5x5x5 factors.
    
    # Offsets for blocks in A, B, C.
    # A: 5x5. Block (i,j) starts at row i_offset, col j_offset.
    # i_offset = sum(n_sizes[:i])
    # j_offset = sum(m_sizes[:j])
    
    # We will accumulate U, V, W columns.
    # Since the total rank is small (85), we can just build the full matrices.
    
    # First, calculate total rank.
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = n_sizes[i]
                mj = m_sizes[j]
                pk = p_sizes[k]
                if (ni, mj, pk) == (2, 2, 2):
                    total_rank += 7
                elif (ni, mj, pk) == (2, 2, 1):
                    total_rank += 2
                elif (ni, mj, pk) == (2, 1, 2):
                    total_rank += 1
                elif (ni, mj, pk) == (2, 1, 1):
                    total_rank += 2
                elif (ni, mj, pk) == (1, 2, 2):
                    total_rank += 2
                elif (ni, mj, pk) == (1, 2, 1):
                    total_rank += 2
                elif (ni, mj, pk) == (1, 1, 2):
                    total_rank += 1
                elif (ni, mj, pk) == (1, 1, 1):
                    total_rank += 1
    
    # total_rank should be 85.
    
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    
    col = 0
    for i in range(3):
        n_i = n_sizes[i]
        n_off = sum(n_sizes[:i])
        for j in range(3):
            m_j = m_sizes[j]
            m_off = sum(m_sizes[:j])
            for k in range(3):
                p_k = p_sizes[k]
                p_off = sum(p_sizes[:k])
                
                if (n_i, m_j, p_k) == (2, 2, 2):
                    u, v, w = _get_2x2_factors()
                    rank_block = 7
                elif (n_i, m_j, p_k) == (2, 2, 1):
                    u, v, w = _get_2x2x1_factors()
                    rank_block = 2
                elif (n_i, m_j, p_k) == (2, 1, 2):
                    u, v, w = _get_2x1x2_factors()
                    rank_block = 1
                elif (n_i, m_j, p_k) == (2, 1, 1):
                    u, v, w = _get_2x1_factors()
                    rank_block = 2
                elif (n_i, m_j, p_k) == (1, 2, 2):
                    u, v, w = _get_1x2_factors()
                    rank_block = 2
                elif (n_i, m_j, p_k) == (1, 2, 1):
                    u, v, w = _get_1x2x1_factors()
                    rank_block = 2
                elif (n_i, m_j, p_k) == (1, 1, 2):
                    u, v, w = _get_1x1x2_factors()
                    rank_block = 1
                elif (n_i, m_j, p_k) == (1, 1, 1):
                    u, v, w = _get_1x1_factors()
                    rank_block = 1
                else:
                    raise ValueError(f"Unexpected block size {n_i}x{m_j}x{p_k}")
                
                # Place u into U
                # u is (n_i*m_j, rank_block)
                # U is (25, total_rank)
                # Rows in U: i_idx * 5 + j_idx
                # i_idx in [n_off, n_off+n_i), j_idx in [m_off, m_off+m_j)
                for ru in range(u.shape[0]):
                    a_i = ru // m_j
                    a_j = ru % m_j
                    global_i = n_off + a_i
                    global_j = m_off + a_j
                    global_ru = global_i * 5 + global_j
                    U[global_ru, col:col+rank_block] = u[ru, :]
                
                # Place v into V
                # v is (m_j*p_k, rank_block)
                # V rows: j_idx * 5 + k_idx
                for rv in range(v.shape[0]):
                    b_j = rv // p_k
                    b_k = rv % p_k
                    global_j = m_off + b_j
                    global_k = p_off + b_k
                    global_rv = global_j * 5 + global_k
                    V[global_rv, col:col+rank_block] = v[rv, :]
                
                # Place w into W
                # w is (p_k*n_i, rank_block)
                # W rows: k_idx * 5 + i_idx
                for rw in range(w.shape[0]):
                    c_k = rw // n_i
                    c_i = rw % n_i
                    global_k = p_off + c_k
                    global_i = n_off + c_i
                    global_rw = global_k * 5 + global_i
                    W[global_rw, col:col+rank_block] = w[rw, :]
                
                col += rank_block
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _block_mult_factors(n, m, p)
    
    # Convert to lists of lists for the return interface
    U_list = [[int(round(x)) for x in row] for row in U.tolist()]
    V_list = [[int(round(x)) for x in row] for row in V.tolist()]
    W_list = [[int(round(x)) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
