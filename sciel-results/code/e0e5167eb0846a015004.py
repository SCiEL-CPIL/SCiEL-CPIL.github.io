"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify exact decomposition with tolerance."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _is_half_integer(x):
    return abs(2 * x - round(2 * x)) < 1e-9

def _verify_entries(U, V, W):
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not _is_half_integer(val):
                    return False
    return True

def _round_halfint(x):
    return round(2 * x) / 2.0

def _verify_and_round(U, V, W, n, m, p):
    """Round to half-integers and verify."""
    U_r = [[_round_halfint(x) for x in row] for row in U]
    V_r = [[_round_halfint(x) for x in row] for row in V]
    W_r = [[_round_halfint(x) for x in row] for row in W]
    if _verify_decomposition(U_r, V_r, W_r, n, m, p) and _verify_entries(U_r, V_r, W_r):
        return U_r, V_r, W_r
    return None

def _try_rank_als(n, m, p, rank, deadline, seed=42):
    """
    Alternating least squares to find a rank-r decomposition.
    For fixed V and W, the problem is linear in U (and similarly for V, W).
    """
    if time.time() > deadline:
        return None
    
    import random
    rng = random.Random(seed)
    
    def _rand_val():
        # Use a wider range of half-integers
        choices = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        return rng.choice(choices)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Try multiple random initializations
    n_trials = max(3, min(20, int((deadline - time.time()) / 5)))
    
    for trial in range(n_trials):
        if time.time() > deadline - 3:
            break
        
        # Random initialization
        U = [[_rand_val() for _ in range(rank)] for _ in range(n * m)]
        V = [[_rand_val() for _ in range(rank)] for _ in range(m * p)]
        W = [[_rand_val() for _ in range(rank)] for _ in range(p * n)]
        
        # Alternating minimization
        n_iters = max(20, min(200, int((deadline - time.time()) / 2)))
        
        for it in range(n_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U: for each (i,j), solve linear system
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j * p + k][r] * W[k * n + i][r]
                    b_vec = np.ones(p)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            U[i * m + j][r] = sol[r]
                    except:
                        pass
            
            # Update V: for each (j,k), solve linear system
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i * m + j][r] * W[k * n + i][r]
                    b_vec = np.ones(n)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            V[j * p + k][r] = sol[r]
                    except:
                        pass
            
            # Update W: for each (k,i), solve linear system
            for k in range(p):
                for i in range(n):
                    A_mat = np.zeros((m, rank))
                    for j in range(m):
                        for r in range(rank):
                            A_mat[j, r] = U[i * m + j][r] * V[j * p + k][r]
                    b_vec = np.ones(m)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            W[k * n + i][r] = sol[r]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
            
            if err < 1e-7:
                break
        
        if best_err < 1e-7:
            break
    
    if best_err < 1e-7 and best_U is not None:
        result = _verify_and_round(best_U, best_V, best_W, n, m, p)
        if result is not None:
            return result
    
    return None

def _try_block_decomposition(n, m, p, deadline):
    """
    Try to decompose [n,m,p] into blocks [n,m1,p] and [n,m2,p] where m1+m2=m.
    Each block can use its own optimal rank.
    """
    if time.time() > deadline:
        return None
    
    n, m, p = 2, 5, 6
    
    # Try splitting m=5 into m1+m2
    best_result = None
    best_rank = float('inf')
    
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() > deadline - 2:
            break
        
        # Get optimal ranks for [2, m1, p] and [2, m2, p]
        # For now, use schoolbook ranks as upper bounds
        rank1 = n * m1 * p
        rank2 = n * m2 * p
        
        # Try to find better ranks for sub-problems
        # This is a simplified version; in practice we'd recurse
        # For now, just use schoolbook for sub-problems
        
        total_rank = rank1 + rank2
        if total_rank >= best_rank:
            continue
        
        # Construct the combined decomposition
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Fill in block 1: [n, m1, p]
        col = 0
        for i in range(n):
            for j in range(m1):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Fill in block 2: [n, m2, p]
        for i in range(n):
            for j in range(m1, m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W):
            best_result = (U, V, W)
            best_rank = total_rank
    
    return best_result

def _try_strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    """
    # Strassen's 7 products
    # M1 = (A11) * (B11 - B22)
    # M2 = (A11 + A22) * B22
    # M3 = A22 * (B11 - B12)
    # M4 = A11 * (B22 + B12)
    # M5 = (A21 + A11) * B11
    # M6 = (A22 - A11) * (B12 + B22)
    # M7 = (A11 - A22) * (B11 + B21)
    
    # For 2x2 matrices:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = A*B = [[C11, C12], [C21, C22]]
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M4
    # C21 = M1 + M2
    # C22 = M2 - M1 + M3 + M6
    
    # U matrix: rows are (i,j) pairs, columns are the 7 products
    # U[i,j,prod] = coefficient of A[i,j] in product prod
    U = [[0]*7 for _ in range(4)]
    # Row 0: (0,0) = A11
    # Row 1: (0,1) = A12
    # Row 2: (1,0) = A21
    # Row 3: (1,1) = A22
    
    # M1: A11
    U[0][0] = 1
    # M2: A11 + A22
    U[0][1] = 1
    U[3][1] = 1
    # M3: A22
    U[3][2] = 1
    # M4: A11
    U[0][3] = 1
    # M5: A21 + A11
    U[2][4] = 1
    U[0][4] = 1
    # M6: A22 - A11
    U[3][5] = 1
    U[0][5] = -1
    # M7: A11 - A22
    U[0][6] = 1
    U[3][6] = -1
    
    # V matrix: rows are (j,k) pairs, columns are the 7 products
    V = [[0]*7 for _ in range(4)]
    # Row 0: (0,0) = B11
    # Row 1: (0,1) = B12
    # Row 2: (1,0) = B21
    # Row 3: (1,1) = B22
    
    # M1: B11 - B22
    V[0][0] = 1
    V[3][0] = -1
    # M2: B22
    V[3][1] = 1
    # M3: B11 - B12
    V[0][2] = 1
    V[1][2] = -1
    # M4: B22 + B12
    V[3][3] = 1
    V[1][3] = 1
    # M5: B11
    V[0][4] = 1
    # M6: B12 + B22
    V[1][5] = 1
    V[3][5] = 1
    # M7: B11 + B21
    V[0][6] = 1
    V[2][6] = 1
    
    # W matrix: rows are (k,i) pairs, columns are the 7 products
    W = [[0]*7 for _ in range(4)]
    # Row 0: (0,0) = C11
    # Row 1: (0,1) = C12
    # Row 2: (1,0) = C21
    # Row 3: (1,1) = C22
    
    # C11 = M1 + M4 - M5 + M7
    W[0][0] = 1
    W[0][3] = 1
    W[0][4] = -1
    W[0][6] = 1
    # C12 = M3 + M4
    W[1][2] = 1
    W[1][3] = 1
    # C21 = M1 + M2
    W[2][0] = 1
    W[2][1] = 1
    # C22 = M2 - M1 + M3 + M6
    W[3][1] = 1
    W[3][0] = -1
    W[3][2] = 1
    W[3][5] = 1
    
    return U, V, W

def _try_kronecker_structure(n, m, p, rank, deadline):
    """
    Try a Kronecker-structured decomposition where W has rank-1 structure per (k,i).
    """
    if time.time() > deadline:
        return None
    
    import random
    rng = random.Random(rank * 1234 + 56)
    
    best_result = None
    best_err = float('inf')
    
    n_trials = max(2, min(10, int((deadline - time.time()) / 10)))
    
    for trial in range(n_trials):
        if time.time() > deadline - 3:
            break
        
        # Initialize W with Kronecker structure: W[k*n+i, r] = alpha[i][r] * beta[k][r]
        alpha = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(n)]
        beta = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(p)]
        
        W = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    W[k * n + i][r] = alpha[i][r] * beta[k][r]
        
        # Initialize U and V
        U = [[0.5 for _ in range(rank)] for _ in range(n * m)]
        V = [[0.5 for _ in range(rank)] for _ in range(m * p)]
        
        # Alternating minimization
        n_iters = max(10, min(100, int((deadline - time.time()) / 5)))
        
        for it in range(n_iters):
            if time.time() > deadline - 2:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A_mat = np.zeros((p, rank))
                    for k in range(p):
                        for r in range(rank):
                            A_mat[k, r] = V[j * p + k][r] * W[k * n + i][r]
                    b_vec = np.ones(p)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            U[i * m + j][r] = sol[r]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A_mat = np.zeros((n, rank))
                    for i in range(n):
                        for r in range(rank):
                            A_mat[i, r] = U[i * m + j][r] * W[k * n + i][r]
                    b_vec = np.ones(n)
                    try:
                        sol = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        for r in range(rank):
                            V[j * p + k][r] = sol[r]
                    except:
                        pass
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                        err += (s - 1.0) ** 2
            err = np.sqrt(err / (n * m * p))
            
            if err < best_err:
                best_err = err
                if err < 1e-7:
                    result = _verify_and_round(U, V, W, n, m, p)
                    if result is not None:
                        return result
                    break
            
            if err < 1e-7:
                break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with schoolbook
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not (_verify_decomposition(U, V, W, n, m, p) and _verify_entries(U, V, W)):
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb
    
    # Try various ranks from low to high (lower rank is better)
    # Lower bound for [2,5,6] is max(2*5, 5*6, 6*2) = max(10,30,12) = 30
    # Try ranks from 30 upward
    
    candidate_ranks = list(range(30, 60))
    
    for r in candidate_ranks:
        if time.time() > deadline - 5:
            break
        
        # Try ALS approach
        result = _try_rank_als(n, m, p, r, deadline, seed=r * 100 + 42)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
        
        # Try Kronecker structure
        result = _try_kronecker_structure(n, m, p, r, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
