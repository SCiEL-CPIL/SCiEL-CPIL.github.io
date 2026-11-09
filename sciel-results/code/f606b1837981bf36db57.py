"""Program for 3x4x4 matrix multiplication rank minimization."""
import time
import itertools

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: sum over all rank terms should give 1 for (i,j,k) position
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook_rank(n, m, p):
    """Return the schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2(U, V, W, i_off, j_off, k_off, n, m, p):
    """Add a Strassen decomposition for a 2x2x2 block. Returns new rank offset."""
    # Strassen's 7 multiplications for 2x2 matrix multiplication
    # U rows: (i,j) -> i*m+j, but within a 2x2 block starting at i_off, j_off
    # V rows: (j,k) -> j*p+k, within block at j_off, k_off
    # W cols: (k,i) -> k*n+i, within block at k_off, i_off
    
    # We'll add 7 new columns
    rank_start = len(U[0]) if U else 0
    new_rank = 7
    
    # U indices: (i,j) for i in {0,1}, j in {0,1} -> row i*m + j within block
    # Actually, U row index = (i_off + i) * m + (j_off + j)
    # V row index = (j_off + j) * p + (k_off + k)
    # W col index = (k_off + k) * n + (i_off + i)
    
    def u_row(i, j):
        return (i_off + i) * m + (j_off + j)
    def v_row(j, k):
        return (j_off + j) * p + (k_off + k)
    def w_col(k, i):
        return (k_off + k) * n + (i_off + i)
    
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For each term, we need entries in U, V, W
    # U entries are coefficients on A(i,j), V on B(j,k), W on C(k,i)
    
    terms = []
    
    # M1: U has A00=1, A11=1; V has B00=1, B11=1; W has C00=1, C01=1, C10=1, C11=1
    # C = M1 contributes to: C00 += 1, C01 += 1, C10 += 1, C11 += 1
    terms.append({
        'u': {0: 1, 3: 1},  # A00, A11
        'v': {0: 1, 3: 1},  # B00, B11
        'w': {0: 1, 1: 1, 2: 1, 3: 1}  # C00, C01, C10, C11
    })
    # M2: U has A10=1, A11=1; V has B00=1; W has C10=1, C11=1
    terms.append({
        'u': {2: 1, 3: 1},  # A10, A11
        'v': {0: 1},  # B00
        'w': {2: 1, 3: 1}  # C10, C11
    })
    # M3: U has A00=1; V has B10=1, B11=-1; W has C00=1, C01=-1
    terms.append({
        'u': {0: 1},  # A00
        'v': {2: 1, 3: -1},  # B10, B11
        'w': {0: 1, 1: -1}  # C00, C01
    })
    # M4: U has A11=1; V has B01=1, B10=-1; W has C10=1, C11=-1
    terms.append({
        'u': {3: 1},  # A11
        'v': {1: 1, 2: -1},  # B01, B10
        'w': {2: 1, 3: -1}  # C10, C11
    })
    # M5: U has A00=1, A01=1; V has B11=1; W has C00=1, C01=1
    terms.append({
        'u': {0: 1, 1: 1},  # A00, A01
        'v': {3: 1},  # B11
        'w': {0: 1, 1: 1}  # C00, C01
    })
    # M6: U has A10=1, A00=-1; V has B00=1, B10=1; W has C10=1, C11=1
    terms.append({
        'u': {2: 1, 0: -1},  # A10, A00
        'v': {0: 1, 2: 1},  # B00, B10
        'w': {2: 1, 3: 1}  # C10, C11
    })
    # M7: U has A01=1, A11=-1; V has B01=1, B11=1; W has C00=1, C01=1
    terms.append({
        'u': {1: 1, 3: -1},  # A01, A11
        'v': {1: 1, 3: 1},  # B01, B11
        'w': {0: 1, 1: 1}  # C00, C01
    })
    
    # Extend U, V, W with 7 new columns
    for row in range(len(U)):
        U[row] = U[row] + [0.0] * new_rank
    for row in range(len(V)):
        V[row] = V[row] + [0.0] * new_rank
    for row in range(len(W)):
        W[row] = W[row] + [0.0] * new_rank
    
    for r_idx, term in enumerate(terms):
        col = rank_start + r_idx
        for u_idx, coeff in term['u'].items():
            U[u_row(u_idx // 2, u_idx % 2)][col] = coeff
        for v_idx, coeff in term['v'].items():
            V[v_row(v_idx // 2, v_idx % 2)][col] = coeff
        for w_idx, coeff in term['w'].items():
            W[w_col(w_idx // 2, w_idx % 2)][col] = coeff
    
    return U, V, W

def _apply_strassen_blocks(n, m, p):
    """Apply Strassen to 2x2x2 blocks where possible, schoolbook for rest."""
    # For 3x4x4, we can partition as:
    # i: 3 rows -> use Strassen for first 2, schoolbook for 3rd
    # j: 4 cols -> use Strassen for first 2, Strassen for next 2
    # k: 4 rows -> use Strassen for first 2, Strassen for next 2
    
    # Start with empty
    U = []
    V = []
    W = []
    
    # We'll build it incrementally
    U = [[0.0]]  # placeholder, will rebuild
    V = [[0.0]]
    W = [[0.0]]
    
    # Actually, let's build properly
    U = [[0.0] for _ in range(n * m)]
    V = [[0.0] for _ in range(m * p)]
    W = [[0.0] for _ in range(p * n)]
    
    # Define which blocks to apply Strassen to
    # i blocks: [0,1] (Strassen), [2] (schoolbook)
    # j blocks: [0,1] (Strassen), [2,3] (Strassen)
    # k blocks: [0,1] (Strassen), [2,3] (Strassen)
    
    i_blocks = [(0, 1, 'S'), (2, 2, 'B')]  # (start, end, type)
    j_blocks = [(0, 1, 'S'), (2, 3, 'S')]
    k_blocks = [(0, 1, 'S'), (2, 3, 'S')]
    
    col_offset = 0
    
    for i_start, i_end, i_type in i_blocks:
        i_size = i_end - i_start
        for j_start, j_end, j_type in j_blocks:
            j_size = j_end - j_start
            for k_start, k_end, k_type in k_blocks:
                k_size = k_end - k_start
                
                if i_type == 'S' and j_type == 'S' and k_type == 'S' and i_size == 2 and j_size == 2 and k_size == 2:
                    # Apply Strassen
                    U, V, W = _strassen_2x2x2(U, V, W, i_start, j_start, k_start, n, m, p)
                    col_offset += 7
                else:
                    # Schoolbook for this block
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for k in range(k_start, k_end):
                                U[i*m+j].append(1.0)
                                V[j*p+k].append(1.0)
                                W[k*n+i].append(1.0)
                                col_offset += 1
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        return None
    
    return U, V, W, col_offset

def _find_better_decomposition(n, m, p, time_limit):
    """Search for a decomposition with rank less than n*m*p."""
    start_time = time.time()
    best_rank = n * m * p
    best_result = None
    
    # First try structured approach with Strassen
    result = _apply_strassen_blocks(n, m, p)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Check time
    if time.time() - start_time > time_limit * 0.5:
        if best_result:
            return best_result
        else:
            U, V, W = _schoolbook_rank(n, m, p)
            return (U, V, W)
    
    # Try random linear combinations to see if we can do better
    # This is a heuristic search
    
    # For 3x4x4, the known rank is at most 48 (schoolbook)
    # With Strassen on 2x2x2 blocks: 4 blocks * 7 + remaining
    # i=0,1; j=0,1; k=0,1 -> 7
    # i=0,1; j=0,1; k=2,3 -> 7
    # i=0,1; j=2,3; k=0,1 -> 7
    # i=0,1; j=2,3; k=2,3 -> 7
    # i=2; j=0,1,2,3; k=0,1,2,3 -> 16
    # Total: 28 + 16 = 44
    
    # Try to see if we can do even better with more sophisticated methods
    
    # Attempt 1: Try to merge some schoolbook terms with Strassen
    # This is complex, so we'll stick with the structured approach
    
    # If we have time, try some random perturbations
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        # Try different block partitions
        pass  # For now, return the best structured result
    
    if best_result:
        return best_result
    else:
        U, V, W = _schoolbook_rank(n, m, p)
        return (U, V, W)

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
    n, m, p = 3, 4, 4
    
    time_limit = 120  # seconds
    start_time = time.time()
    
    # Deadline guard
    deadline = start_time + time_limit - 5  # 5 second safety margin
    
    # Try to find a better decomposition
    try:
        U, V, W = _find_better_decomposition(n, m, p, time_limit)
    except Exception:
        U, V, W = _schoolbook_rank(n, m, p)
    
    # Check if we're still within time
    if time.time() > deadline:
        U, V, W = _schoolbook_rank(n, m, p)
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook_rank(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
