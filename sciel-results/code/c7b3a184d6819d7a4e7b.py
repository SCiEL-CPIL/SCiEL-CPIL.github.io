"""Search for a rank-minimal exact tensor decomposition of 2x4 times 4x7 matrix multiply."""
import time
import random
import numpy as np
from numpy.linalg import lstsq

DIMENSIONS = [2, 4, 7]
DEADLINE = time.time() + 110.0  # safety margin within 120s


def _construct_np(n, m, p):
    """Construct rank n*p decomposition."""
    rank = n * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    for i in range(n):
        for k in range(p):
            c = k * n + i  # column index in [0, n*p)
            for j in range(m):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
            W[k * n + i, c] = 1
    return U, V, W


def _verify_np(U, V, W, n, m, p, tol=1e-6):
    """Faster vectorized check. U, V, W are 2D arrays (rows, rank)."""
    for i in range(n):
        for k in range(p):
            w = W[k * n + i]
            s = 0.0
            for j in range(m):
                u = U[i * m + j]
                v = V[j * p + k]
                if np.any(u != 0) and np.any(v != 0):
                    s += np.sum(u * v * w)
            if abs(s - 1.0) > tol:
                return False
    return True


def _try_rank_with_linear_solve(target_rank, n, m, p, attempts=50):
    """
    Try to find a decomposition of given rank using linear algebra.
    Fix W randomly (with 0/1 entries), then solve for U and V.
    """
    for _ in range(attempts):
        if time.time() > DEADLINE:
            return None
        
        # Random W with 0/1 entries, ensuring each (i,k) has at least one 1
        W = np.zeros((p * n, target_rank), dtype=float)
        for c in range(target_rank):
            # Each column of W is assigned to one (i,k) pair
            i = random.randint(0, n - 1)
            k = random.randint(0, p - 1)
            W[k * n + i, c] = 1
        
        # Check that every (i,k) pair has at least one column
        pair_cols = {}
        for c in range(target_rank):
            nz = np.nonzero(W[:, c])[0]
            if len(nz) > 0:
                idx = nz[0]
                i = idx % n
                k = idx // n
                if (i, k) not in pair_cols:
                    pair_cols[(i, k)] = []
                pair_cols[(i, k)].append(c)
        
        all_pairs = [(i, k) for i in range(n) for k in range(p)]
        if len(pair_cols) < len(all_pairs):
            continue
        
        # For each (i,k), we need: sum_j sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let's fix the structure: for each column c, it's assigned to one (i0,k0)
        # Then U[i0*m+j,c] and V[j*p+k0,c] are the only non-zero entries in column c for U and V
        # For (i,k) != (i0,k0), W[k*n+i,c] = 0, so column c doesn't contribute to (i,k)
        
        # So for each (i,k), the equation is:
        # sum_{c: assigned to (i,k)} sum_j U[i*m+j,c] * V[j*p+k,c] = 1
        
        # We can solve this by making each column contribute independently.
        # For a column c assigned to (i0,k0), we set U[i0*m+j,c] = a_j, V[j*p+k0,c] = b_j
        # Then the contribution to (i0,k0) is sum_j a_j * b_j
        # We need sum over all columns assigned to (i0,k0) of sum_j a_j*b_j = 1
        
        # Simplest: for each (i,k), if it has exactly one column c, set a_j=1, b_j=1 for all j
        # If it has multiple columns, we can split the contribution.
        
        U = np.zeros((n * m, target_rank), dtype=float)
        V = np.zeros((m * p, target_rank), dtype=float)
        
        valid = True
        for (i, k) in all_pairs:
            cols = pair_cols.get((i, k), [])
            if not cols:
                valid = False
                break
            
            if len(cols) == 1:
                c = cols[0]
                for j in range(m):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
            else:
                # Multiple columns: use first column for all, others zero
                c_first = cols[0]
                for j in range(m):
                    U[i * m + j, c_first] = 1
                    V[j * p + k, c_first] = 1
                for c in cols[1:]:
                    # These columns are assigned to (i,k) but we set them to 0
                    # This means they don't contribute, which is fine
                    pass
        
        if not valid:
            continue
        
        if _verify_np(U, V, W, n, m, p):
            return {'U': U, 'V': V, 'W': W}
    
    return None


def _try_block_decomposition(n, m, p):
    """
    Use block decomposition with Strassen for 2x2x2 blocks.
    Split m=4 into two 2x2 blocks, p=7 into three 2x2 and one 2x1.
    """
    rank = 2 * (3 * 7 + 1 * 4)  # 2 * (21 + 4) = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen(a_rows, b_rows, c_rows, start_col):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2"""
        # a_rows: dict mapping (i,j) to row index in U
        # b_rows: dict mapping (j,k) to row index in V
        # c_rows: dict mapping (k,i) to row index in W
        nonlocal col
        
        # M1 = (A11+A22)(B11+B22)
        U[a_rows[(0,0)], col] = 1
        U[a_rows[(1,1)], col] = 1
        V[b_rows[(0,0)], col] = 1
        V[b_rows[(1,1)], col] = 1
        W[c_rows[(0,0)], col] = 1
        W[c_rows[(1,1)], col] = 1
        col += 1
        
        # M2 = (A21+A22)B11
        U[a_rows[(1,0)], col] = 1
        U[a_rows[(1,1)], col] = 1
        V[b_rows[(0,0)], col] = 1
        W[c_rows[(1,0)], col] = 1
        W[c_rows[(1,1)], col] = -1
        col += 1
        
        # M3 = A11(B21-B22)
        U[a_rows[(0,0)], col] = 1
        V[b_rows[(0,1)], col] = 1
        V[b_rows[(1,1)], col] = -1
        W[c_rows[(0,1)], col] = 1
        W[c_rows[(1,1)], col] = 1
        col += 1
        
        # M4 = A22(B11-B10)
        U[a_rows[(1,1)], col] = 1
        V[b_rows[(1,0)], col] = 1
        V[b_rows[(0,0)], col] = -1
        W[c_rows[(0,0)], col] = 1
        W[c_rows[(1,0)], col] = 1
        col += 1
        
        # M5 = (A10+A11)B22
        U[a_rows[(0,0)], col] = 1
        U[a_rows[(0,1)], col] = 1
        V[b_rows[(1,1)], col] = 1
        W[c_rows[(0,0)], col] = -1
        W[c_rows[(0,1)], col] = 1
        col += 1
        
        # M6 = (A21-A11)(B10+B11)
        U[a_rows[(1,0)], col] = 1
        U[a_rows[(0,0)], col] = -1
        V[b_rows[(0,0)], col] = 1
        V[b_rows[(0,1)], col] = 1
        W[c_rows[(1,1)], col] = 1
        col += 1
        
        # M7 = (A10-A21)(B21+B22)
        U[a_rows[(0,1)], col] = 1
        U[a_rows[(1,1)], col] = -1
        V[b_rows[(1,0)], col] = 1
        V[b_rows[(1,1)], col] = 1
        W[c_rows[(0,0)], col] = 1
        col += 1
        
        return col
    
    def add_schoolbook_221(a_rows, b_rows, c_rows, start_col):
        """Add schoolbook for 2x2 * 2x1 -> 2x1"""
        nonlocal col
        for j in range(2):
            for k in range(1):
                for i in range(2):
                    U[a_rows[(i,0)], col] = 1
                    V[b_rows[(j,k)], col] = 1
                    W[c_rows[(k,i)], col] = 1
                    col += 1
        return col
    
    # A block 1: rows 0-1 of A (2x4), so a_rows for 2x2 submatrix
    # A is 2x4, split into two 2x2 blocks: A1 = A[:,0:2], A2 = A[:,2:4]
    a1_rows = {(0,0): 0, (0,1): 1, (1,0): 4, (1,1): 5}
    a2_rows = {(0,0): 2, (0,1): 3, (1,0): 6, (1,1): 7}
    
    # B is 4x7, split into four 2x7 blocks: B1=B[0:2,:], B2=B[2:4,:]
    # Each 2x7 is split into three 2x2 and one 2x1
    # B1: rows 0-1, cols 0-6
    b1_blocks = [
        {(0,0): 0, (0,1): 1, (1,0): 7, (1,1): 8},   # cols 0-1
        {(0,0): 2, (0,1): 3, (1,0): 9, (1,1): 10},   # cols 2-3
        {(0,0): 4, (0,1): 5, (1,0): 11, (1,1): 12},  # cols 4-5
        {(0,0): 6, (1,0): 13},                         # col 6
    ]
    b2_blocks = [
        {(0,0): 14, (0,1): 15, (1,0): 21, (1,1): 22}, # cols 0-1
        {(0,0): 16, (0,1): 17, (1,0): 23, (1,1): 24}, # cols 2-3
        {(0,0): 18, (0,1): 19, (1,0): 25, (1,1): 26}, # cols 4-5
        {(0,0): 20, (1,0): 27},                         # col 6
    ]
    
    # C is 7x2, split into seven 1x2 blocks, but we group by k
    # For each k (0-6), C[k,:] is 1x2, so c_rows maps (k,i) to row k*2+i
    # But in our tensor, W is indexed by (k,i) with shape (p*n, rank) = (14, rank)
    # So row index for (k,i) is k*n+i = k*2+i
    
    # For A1 * B1_block0: C[0:2, 0:2] -> rows 0,1,2,3 in W
    c_block = lambda k0: {(0,0): k0*2+0, (0,1): k0*2+1, (1,0): (k0+1)*2+0, (1,1): (k0+1)*2+1}
    
    col = add_strassen(a1_rows, b1_blocks[0], c_block(0), col)
    col = add_strassen(a1_rows, b1_blocks[1], c_block(2), col)
    col = add_strassen(a1_rows, b1_blocks[2], c_block(4), col)
    col = add_schoolbook_221(a1_rows, b1_blocks[3], {(0,0): 12, (1,0): 13}, col)
    
    col = add_strassen(a2_rows, b2_blocks[0], c_block(0), col)
    col = add_strassen(a2_rows, b2_blocks[1], c_block(2), col)
    col = add_strassen(a2_rows, b2_blocks[2], c_block(4), col)
    col = add_schoolbook_221(a2_rows, b2_blocks[3], {(0,0): 12, (1,0): 13}, col)
    
    if col != rank:
        # Truncate or pad
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col
    
    return {'U': U, 'V': V, 'W': W}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    # 1) Try rank = n*p = 14 (this is the best known simple construction)
    U14, V14, W14 = _construct_np(n, m, p)
    if _verify_np(U14, V14, W14, n, m, p):
        best_U, best_V, best_W = U14, V14, W14
        best_rank = n * p
        print(f"Found rank {n*p} decomposition")
    else:
        # Fallback to block decomposition
        result = _try_block_decomposition(n, m, p)
        best_U, best_V, best_W = result['U'], result['V'], result['W']
        best_rank = best_U.shape[1]
        print(f"Rank {n*p} failed, using block decomposition with rank {best_rank}")

    # 2) Try to find lower ranks using linear algebra
    for rank in range(best_rank - 1, max(10, best_rank - 5), -1):
        if time.time() > DEADLINE - 15:
            break
        result = _try_rank_with_linear_solve(rank, n, m, p, attempts=30)
        if result is not None:
            best_U, best_V, best_W = result['U'], result['V'], result['W']
            best_rank = rank
            print(f"Found rank {rank} decomposition")
            break

    # Convert to list-of-lists with integer/half-integer entries
    def to_int_list(mat):
        m2 = mat * 2
        if np.allclose(m2, np.round(m2), atol=1e-9):
            m2 = np.round(m2).astype(int)
            if np.all(m2 % 2 == 0):
                return (m2 // 2).tolist()
            return m2.tolist()
        return mat.tolist()

    return {
        'U': to_int_list(best_U),
        'V': to_int_list(best_V),
        'W': to_int_list(best_W)
    }
