"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Constructs a tensor decomposition for matrix multiplication C = A @ B
    where A is n x m and B is m x p.
    
    Decomposition form:
    C_k_i = sum_l W_{k,i,l} * ( sum_{i',j'} U_{i',j',l} A_{i',j'} ) * ( sum_{j'',k'} V_{j'',k',l} B_{j'',k'} )
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)  <-- Index is k*n + i
    """
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Keep well within 120s limit

    # Strategy: Block decomposition
    # A (2x4) = [A_L, A_R] where A_L, A_R are 2x2
    # B (4x5) = [B_1; B_2] where B_1, B_2 are 2x5
    # C = A_L B_1 + A_R B_2
    
    # We solve M_{2,2,5} twice.
    # M_{2,2,5} decomposition:
    # B_2x5 = [B_a(2x2), B_b(2x2), B_c(2x1)]
    # A B_a (rank 7, Strassen)
    # A B_b (rank 7, Strassen)
    # A B_c (rank 4, trivial)
    # Total per block: 18.
    # Total rank: 36.
    
    rank = 36
    # Shapes
    # U: (n*m, rank) = (8, 36)
    # V: (m*p, rank) = (20, 36)
    # W: (p*n, rank) = (10, 36)
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def u_idx(i, j): return i*m + j
    def v_idx(j, k): return j*p + k
    def w_idx(k, i): return k*n + i
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 decomposition
    def add_strassen_2x2x2(ai0, ai1, aj0, aj1, bj0, bj1, bk0, bk1):
        nonlocal col
        # Strassen constants
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b00 = v_idx(bj0, bk0)
        b01 = v_idx(bj0, bk1)
        b10 = v_idx(bj1, bk0)
        b11 = v_idx(bj1, bk1)
        
        c00 = w_idx(bk0, ai0)
        c01 = w_idx(bk1, ai0)
        c10 = w_idx(bk0, ai1)
        c11 = w_idx(bk1, ai1)
        
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        
        # M3
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        
        # M4
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1
        col += 1
        
    # Helper to add a 2x2 @ 2x1 decomposition (Rank 4)
    def add_vec_mat_2x2x1(ai0, ai1, aj0, aj1, bj0, bj1, bk):
        nonlocal col
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        
        a00 = u_idx(ai0, aj0)
        a01 = u_idx(ai0, aj1)
        a10 = u_idx(ai1, aj0)
        a11 = u_idx(ai1, aj1)
        
        b0 = v_idx(bj0, bk)
        b1 = v_idx(bj1, bk)
        
        c0 = w_idx(bk, ai0)
        c1 = w_idx(bk, ai1)
        
        # M1: A00 B0 -> C0
        U[a00, col] = 1; V[b0, col] = 1; W[c0, col] += 1; col += 1
        # M2: A01 B1 -> C0
        U[a01, col] = 1; V[b1, col] = 1; W[c0, col] += 1; col += 1
        # M3: A10 B0 -> C1
        U[a10, col] = 1; V[b0, col] = 1; W[c1, col] += 1; col += 1
        # M4: A11 B1 -> C1
        U[a11, col] = 1; V[b1, col] = 1; W[c1, col] += 1; col += 1
        
    # Block 1: A_L (rows 0,1 cols 0,1) @ B_1 (rows 0,1 cols 0,4)
    # B_1 is 2x5. Split into B_1a (cols 0,1), B_1b (cols 2,3), B_1c (col 4).
    
    # A_L @ B_1a (2x2 @ 2x2)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 0, 1)
    
    # A_L @ B_1b (2x2 @ 2x2)
    add_strassen_2x2x2(0, 1, 0, 1, 0, 1, 2, 3)
    
    # A_L @ B_1c (2x2 @ 2x1)
    add_vec_mat_2x2x1(0, 1, 0, 1, 0, 1, 4)
    
    # Block 2: A_R (rows 0,1 cols 2,3) @ B_2 (rows 2,3 cols 0,4)
    # B_2 is 2x5. Split into B_2a (cols 0,1), B_2b (cols 2,3), B_2c (col 4).
    
    # A_R @ B_2a (2x2 @ 2x2)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 0, 1)
    
    # A_R @ B_2b (2x2 @ 2x2)
    add_strassen_2x2x2(0, 1, 2, 3, 2, 3, 2, 3)
    
    # A_R @ B_2c (2x2 @ 2x1)
    add_vec_mat_2x2x1(0, 1, 2, 3, 2, 3, 4)
    
    # Check if we are running out of time
    if time.time() - start_time > timeout:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank by checking for linear dependencies
    # We can use SVD or QR decomposition to find the true rank
    # and potentially reduce the number of terms if some are linearly dependent
    
    # Check for zero columns
    new_rank = rank
    keep_cols = []
    
    for c in range(rank):
        if np.all(U[:, c] == 0) or np.all(V[:, c] == 0) or np.all(W[:, c] == 0):
            continue
        keep_cols.append(c)
    
    if len(keep_cols) < new_rank:
        new_rank = len(keep_cols)
        U = U[:, keep_cols]
        V = V[:, keep_cols]
        W = W[:, keep_cols]
        rank = new_rank

    # Try to further reduce rank by finding linear dependencies in the tensor
    # A term l is redundant if its Kronecker product vector is in the span of the others.
    # We do this greedily: iterate through terms, keep a term only if it's linearly independent
    # of all previously kept terms.
    
    def kron_vec(u, v, w):
        """Compute the Kronecker product vector for a single term."""
        # u has shape (n*m,), v has shape (m*p,), w has shape (p*n,)
        # The tensor element is u[i] * v[j] * w[k] for appropriate indices
        # Flatten: index = i * (m*p) * (p*n) + j * (p*n) + k
        # But we need to be careful about the indexing.
        # Actually, the tensor T has entries T[a,b,c] where a in [0,n*m), b in [0,m*p), c in [0,p*n)
        # T[a,b,c] = sum_l U[a,l] * V[b,l] * W[c,l]
        # So the vector for term l is the Kronecker product U[:,l] ⊗ V[:,l] ⊗ W[:,l]
        return np.kron(np.kron(u, v), w)
    
    # Build the list of term vectors
    term_vectors = []
    for l in range(rank):
        if np.all(U[:, l] == 0) or np.all(V[:, l] == 0) or np.all(W[:, l] == 0):
            term_vectors.append(None)  # Mark as zero
        else:
            term_vectors.append(kron_vec(U[:, l], V[:, l], W[:, l]))
    
    # Greedy linear independence check
    kept = []
    basis = []  # List of kept vectors
    
    for l in range(rank):
        if term_vectors[l] is None:
            continue
        vec = term_vectors[l]
        
        # Check if vec is in the span of basis
        if not basis:
            kept.append(l)
            basis.append(vec)
            continue
        
        # Form the matrix with basis vectors as columns and check rank
        # This is expensive but rank is at most 36
        B = np.column_stack(basis)
        # Check if rank([B, vec]) > rank(B)
        B_aug = np.column_stack([B, vec])
        r = np.linalg.matrix_rank(B)
        r_aug = np.linalg.matrix_rank(B_aug)
        
        if r_aug > r:
            kept.append(l)
            basis.append(vec)
    
    if len(kept) < rank:
        rank = len(kept)
        U = U[:, kept]
        V = V[:, kept]
        W = W[:, kept]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
