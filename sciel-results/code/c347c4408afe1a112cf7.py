"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _check_decomposition(U, V, W, n, m, p, rank, tol=1e-8):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor T[i,j,k] which should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # The matrix multiplication tensor: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 for all (i,j,k) in the "product" sense
    # Actually the matrix multiplication tensor M has M[i,j,k] = 1 if we think of it as
    # the structure constants. Let me reconsider.
    
    # For matrix multiplication A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has indices (i, j, k) and T[i,j,k] = 1 for all i,j,k
    # Wait no. The tensor is defined such that the multiplication is:
    # <T, A ⊗ B> = C where A is flattened to (n*m), B to (m*p), C to (n*p)
    # T is a 3-tensor with dimensions (n*m, m*p, n*p)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = i*p+k
    # T[a,b,c] = 1 if the indices are consistent (same j for a,b and same i,k for a,c)
    
    # Let me use the convention from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] - so W is indexed by (k, i) with k in [0,p), i in [0,n)
    # The decomposition: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    # T[a, b, c] where a = i*m + j, b = j'*p + k, c = k'*n + i'
    # T[a,b,c] = 1 if j == j' and i == i' and k == k', else 0
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i*m+j, j*p+k, k*n+i] = 1 (using the W convention W[k*n+i])
    
    # Let's build T and check
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the decomposition
    # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a tensor product: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # Efficient check: for each r, outer product
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    
    return np.allclose(T_decomp, T, atol=tol)

def _rank_min_lower_bound(n, m, p):
    """Known lower bounds for matrix multiplication rank."""
    # Strassen: for 2x2x2, rank 7
    # For general cases, use known bounds
    # The schoolbook rank is n*m*p
    # Strassen's algorithm gives better for 2x2x2
    # For 3x3x3, best known is 23 (or lower?)
    # Actually for 3x3x3, the best known upper bound is 23, lower bound is 19
    # For 3x3x5, we need to think about it
    return n * m * p  # fallback to schoolbook

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition as baseline
    U_best = [[0] * schoolbook_rank for _ in range(n * m)]
    V_best = [[0] * schoolbook_rank for _ in range(m * p)]
    W_best = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][column] = 1
                V_best[j * p + k][column] = 1
                W_best[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    U_best = [row[:] for row in U_best]
    V_best = [row[:] for row in V_best]
    W_best = [row[:] for row in W_best]
    
    # Try to find lower rank decompositions
    # Use random search with verification
    # Start from lower ranks and try to find valid decompositions
    
    # Known results:
    # 3x3x3: best known upper bound 23, lower bound 19
    # 3x3x5: we can try various ranks
    
    # Strategy: try ranks from lower bound up to schoolbook
    # For each rank, do random initialization and try to refine
    
    # Lower bound: for n x m x p, rank >= max(n*m, m*p, p*n, ceil(n*m*p / max(n,m,p)))
    # Actually the simple lower bound is max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)
    
    # For 3x3x5: lb = max(9, 15, 15) = 15
    # Schoolbook: 45
    
    # Let's try to find decompositions of rank < schoolbook
    # Use a combination of:
    # 1. Random search with integer/half-integer values
    # 2. Local optimization
    
    # For efficiency, work with numpy arrays during search, convert at end
    
    U_arr = np.array(U_best, dtype=float)
    V_arr = np.array(V_best, dtype=float)
    W_arr = np.array(W_best, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try ranks from lb up to schoolbook_rank - 1
    # For each rank, do random restarts with local search
    
    def try_rank(rank, max_iterations=500, max_restarts=10):
        """Try to find a decomposition of given rank."""
        nonlocal best_rank, U_best, V_best, W_best
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization with small values
            # Use values from {-1, -0.5, 0, 0.5, 1}
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Local search: try to reduce the error
            # Error = ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||_F
            def compute_error(U, V, W, rank):
                T_decomp = np.zeros((n*m, m*p, p*n))
                for r in range(rank):
                    T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                return np.linalg.norm(T_decomp - T)
            
            err = compute_error(U, V, W, rank)
            if err < 1e-8:
                # Found a valid decomposition
                best_rank = rank
                U_best = [[float(x) for x in row] for row in U]
                V_best = [[float(x) for x in row] for row in V]
                W_best = [[float(x) for x in row] for row in W]
                return True
            
            # Local search: perturb one entry at a time
            for it in range(max_iterations):
                if time.time() > deadline:
                    return False
                
                # Pick a random matrix and entry
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    arr = U
                    rows, cols = n*m, rank
                elif mat_choice == 'V':
                    arr = V
                    rows, cols = m*p, rank
                else:
                    arr = W
                    rows, cols = p*n, rank
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
                
                # Try changing the value
                old_val = arr[row, col]
                new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                
                if new_val == old_val:
                    continue
                
                arr[row, col] = new_val
                new_err = compute_error(U, V, W, rank)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-8:
                        best_rank = rank
                        U_best = [[float(x) for x in row] for row in U]
                        V_best = [[float(x) for x in row] for row in V]
                        W_best = [[float(x) for x in row] for row in W]
                        return True
                else:
                    arr[row, col] = old_val
            
            # If we made progress, keep going with another restart
            if err < 1.0:
                # Promising, do more iterations
                for it in range(max_iterations):
                    if time.time() > deadline:
                        return False
                    mat_choice = random.choice(['U', 'V', 'W'])
                    if mat_choice == 'U':
                        arr = U
                        rows, cols = n*m, rank
                    elif mat_choice == 'V':
                        arr = V
                        rows, cols = m*p, rank
                    else:
                        arr = W
                        rows, cols = p*n, rank
                    
                    row = random.randint(0, rows - 1)
                    col = random.randint(0, cols - 1)
                    
                    old_val = arr[row, col]
                    new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                    
                    if new_val == old_val:
                        continue
                    
                    arr[row, col] = new_val
                    new_err = compute_error(U, V, W, rank)
                    
                    if new_err < err:
                        err = new_err
                        if err < 1e-8:
                            best_rank = rank
                            U_best = [[float(x) for x in row] for row in U]
                            V_best = [[float(x) for x in row] for row in V]
                            W_best = [[float(x) for x in row] for row in W]
                            return True
                    else:
                        arr[row, col] = old_val
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    # Start with a few promising ranks
    # For 3x3x5, try ranks around 15-30
    
    # Be strategic: try a range of ranks
    ranks_to_try = list(range(max(lb, 15), min(schoolbook_rank, 35)))
    random.shuffle(ranks_to_try)
    
    # Try a subset of ranks with limited budget
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Limit work per rank
        if rank < 20:
            success = try_rank(rank, max_iterations=200, max_restarts=5)
        elif rank < 30:
            success = try_rank(rank, max_iterations=300, max_restarts=8)
        else:
            success = try_rank(rank, max_iterations=400, max_restarts=10)
        
        if success:
            # Found a valid decomposition, try to go lower
            # But first, let's try a few more lower ranks
            lower_ranks = [r for r in range(lb, rank) if r not in ranks_to_try[:ranks_to_try.index(rank)]]
            random.shuffle(lower_ranks)
            for lr in lower_ranks[:5]:
                if time.time() > deadline:
                    break
                try_rank(lr, max_iterations=300, max_restarts=10)
            break
    
    # Final verification
    if not _check_decomposition(U_best, V_best, W_best, n, m, p, best_rank):
        # Fall back to schoolbook
        best_rank = schoolbook_rank
        U_best = [[0] * schoolbook_rank for _ in range(n * m)]
        V_best = [[0] * schoolbook_rank for _ in range(m * p)]
        W_best = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_best[i * m + j][column] = 1
                    V_best[j * p + k][column] = 1
                    W_best[k * n + i][column] = 1
                    column += 1
    
    return {"U": U_best, "V": V_best, "W": W_best}

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook decomposition as baseline
    U_best = [[0] * schoolbook_rank for _ in range(n * m)]
    V_best = [[0] * schoolbook_rank for _ in range(m * p)]
    W_best = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_best[i * m + j][column] = 1
                V_best[j * p + k][column] = 1
                W_best[k * n + i][column] = 1
                column += 1
    
    best_rank = schoolbook_rank
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def compute_error(U, V, W, rank):
        T_decomp = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_decomp += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.linalg.norm(T_decomp - T)
    
    def try_rank(rank, max_iterations=500, max_restarts=10):
        nonlocal best_rank, U_best, V_best, W_best
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            err = compute_error(U, V, W, rank)
            if err < 1e-8:
                best_rank = rank
                U_best = [[float(x) for x in row] for row in U]
                V_best = [[float(x) for x in row] for row in V]
                W_best = [[float(x) for x in row] for row in W]
                return True
            
            # Local search
            for it in range(max_iterations):
                if time.time() > deadline:
                    return False
                
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    arr = U
                    rows, cols = n*m, rank
                elif mat_choice == 'V':
                    arr = V
                    rows, cols = m*p, rank
                else:
                    arr = W
                    rows, cols = p*n, rank
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, cols - 1)
                
                old_val = arr[row, col]
                new_val = random.choice([-1, -0.5, 0, 0.5, 1])
                
                if new_val == old_val:
                    continue
                
                arr[row, col] = new_val
                new_err = compute_error(U, V, W, rank)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-8:
                        best_rank = rank
                        U_best = [[float(x) for x in row] for row in U]
                        V_best = [[float(x) for x in row] for row in V]
                        W_best = [[float(x) for x in row] for row in W]
                        return True
                else:
                    arr[row, col] = old_val
    
    # Lower bound
    lb = max(n*m, m*p, p*n)
    
    # Try ranks from lower bound up to schoolbook_rank - 1
    # For 3x3x5: lb=15, schoolbook=45
    # Try a strategic subset
    
    # First try some key ranks
    key_ranks = [lb, lb+1, lb+2, lb+3, lb+5, lb+10, lb+15, lb+20]
    key_ranks = [r for r in key_ranks if r < schoolbook_rank]
    random.shuffle(key_ranks)
    
    for rank in key_ranks:
        if time.time() > deadline:
            break
        try_rank(rank, max_iterations=300, max_restarts=5)
        if best_rank < schoolbook_rank:
            break
    
    # If we found something, try to improve
    if best_rank < schoolbook_rank and time.time() < deadline:
        # Try ranks below current best
        for rank in range(lb, best_rank):
            if time.time() > deadline:
                break
            try_rank(rank, max_iterations=400, max_restarts=10)
    
    # Final verification
    if not _check_decomposition(U_best, V_best, W_best, n, m, p, best_rank):
        best_rank = schoolbook_rank
        U_best = [[0] * schoolbook_rank for _ in range(n * m)]
        V_best = [[0] * schoolbook_rank for _ in range(m * p)]
        W_best = [[0] * schoolbook_rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_best[i * m + j][column] = 1
                    V_best[j * p + k][column] = 1
                    W_best[k * n + i][column] = 1
                    column += 1
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
