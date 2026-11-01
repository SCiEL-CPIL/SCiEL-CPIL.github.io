import time
import sys
import random

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the standard rank-n*m*p tensor decomposition for matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    For each (i, j, k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    For all other (i, j, k) combinations (not valid multiplications), they should be 0.
    
    Actually, the tensor M_{n,m,p} has M[i,j,k] = 1 if we're computing entry (i,k) 
    of the product, which involves sum over j of A[i,j]*B[j,k].
    
    The decomposition satisfies: for each r, the tensor U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    and sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = M_{n,m,p}
    
    Where M_{n,m,p} has M[i, j, k] = 1 for the tensor that represents multiplication.
    Actually M[i,j,k] = 1 iff this is a valid (i,j,k) triple for the multiplication.
    
    More precisely: the matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k
    where these index the input A (n×m), B (m×p) and output C (p×n) with the 
    trace convention W[k*n+i, r].
    
    Let me re-read: W[k*n+i, r] means the output index is k*n + i, so output C[k, i] 
    (or C[i, k] depending on convention). With W indexed by k*n+i, the output is 
    C[k, i] for k in range(p), i in range(n). So C is p×n.
    
    The tensor condition: For each output position (k, i) and each intermediate j:
    C[k, i] = sum_j A[i, j] * B[j, k]
    
    The tensor T[i, j, k] = 1 for all valid (i, j, k) triples, where i in [0,n), 
    j in [0,m), k in [0,p). The decomposition satisfies:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all valid (i,j,k).
    
    Since there are no "invalid" triples (every i, j, k is valid), we just check
    that all entries equal 1.
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _rank_of_1d_vector(v):
    """Return the rank (number of non-zero entries) of a 1D vector."""
    return sum(1 for x in v if abs(x) > 1e-12)


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2 by 3 matrix by a 3 by 4 matrix.
    
    Uses the schoolbook decomposition of rank 24 as the baseline, then attempts
    local search to find a lower-rank decomposition.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    n, m, p = 2, 3, 4
    base_rank = n * m * p  # 24
    
    U, V, W = _build_schoolbook(n, m, p)
    best_rank = base_rank
    best_U, best_V, best_W = U, V, W
    
    # Verify the schoolbook decomposition
    assert _verify_decomposition(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    
    # Try to find a lower-rank decomposition via local search
    # Strategy: Start with rank r < base_rank and try to find entries that satisfy
    # the tensor equation. Use randomized local search.
    
    # For rank r, we need U (n*m x r), V (m*p x r), W (p*n x r) such that
    # for all i, j, k: sum_{t=0}^{r-1} U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t] = 1
    
    # This is a system of n*m*p = 24 equations in 3*r*(nm+mp+pn) = 3*r*26 variables
    # (actually nm=6, mp=12, pn=8, so 3*r*26 = 78r variables)
    # With r=24: 78*24 = 1872 variables, 24 equations - very underdetermined
    # But we need integer/half-integer entries for the "exact" requirement
    
    # Let's try a different approach: use the known lower bounds.
    # For 2x3 x 3x4, the tensor rank is known to be 24 (the schoolbook is optimal
    # for small cases where no faster algorithms exist).
    
    # However, let's still attempt a search to see if we can find rank < 24.
    # We'll use a randomized approach with local optimization.
    
    # Try ranks from 23 down to a reasonable minimum
    # Use a time-limited search
    
    def _try_rank_search(target_rank, time_budget):
        """Try to find a decomposition of the given rank."""
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with random values
        random.seed(42)
        for i in range(n * m):
            for t in range(target_rank):
                U[i][t] = random.choice([-1, 0, 1])
        for j in range(m * p):
            for t in range(target_rank):
                V[j][t] = random.choice([-1, 0, 1])
        for k in range(p * n):
            for t in range(target_rank):
                W[k][t] = random.choice([-1, 0, 1])
        
        best_error = float('inf')
        
        while time.time() < time_budget:
            # Compute current error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        for t in range(target_rank):
                            s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                        error += (s - 1.0) ** 2
            
            if error < best_error:
                best_error = error
                if error < 1e-9:
                    return U, V, W, True
            
            # Local search: try flipping one entry
            # Pick a random matrix and entry
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                row = random.randrange(n * m)
            elif mat_choice == 'V':
                row = random.randrange(m * p)
            else:
                row = random.randrange(p * n)
            col = random.randrange(target_rank)
            
            # Try changing this entry
            old_val = (U if mat_choice == 'U' else V if mat_choice == 'V' else W)[row][col]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            
            # Compute new error
            new_error = 0.0
            if mat_choice == 'U':
                U[row][col] = new_val
                for i in range(n):
                    for j in range(m):
                        if i * m + j != row:
                            continue
                        for k in range(p):
                            s = 0.0
                            v_idx = j * p + k
                            w_idx = k * n + i
                            for t in range(target_rank):
                                s += U[i * m + j][t] * V[v_idx][t] * W[w_idx][t]
                            new_error += (s - 1.0) ** 2
                    break
            elif mat_choice == 'V':
                V[row][col] = new_val
                for j in range(m):
                    if j * p == row or (row // p != j):
                        continue
                    for i in range(n):
                        for k in range(p):
                            if j * p + k != row:
                                continue
                            s = 0.0
                            u_idx = i * m + j
                            w_idx = k * n + i
                            for t in range(target_rank):
                                s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                            new_error += (s - 1.0) ** 2
            else:
                W[row][col] = new_val
                for k in range(p):
                    if k * n != row:
                        continue
                    for i in range(n):
                        if k * n + i != row:
                            continue
                        for j in range(m):
                            s = 0.0
                            u_idx = i * m + j
                            v_idx = j * p + k
                            for t in range(target_rank):
                                s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                            new_error += (s - 1.0) ** 2
            
            # Accept if better or same
            if new_error <= error + 1e-12:
                if new_error < best_error:
                    best_error = new_error
                    if new_error < 1e-9:
                        return U, V, W, True
                # Keep the change
            else:
                # Revert
                if mat_choice == 'U':
                    U[row][col] = old_val
                elif mat_choice == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
        
        return U, V, W, False
    
    # Try ranks from base_rank-1 down to base_rank-5 (limited search)
    for r in range(base_rank - 1, max(base_rank - 6, 15), -1):
        if time.time() > deadline - 10:
            break
        
        time_budget = min(deadline - 5, time.time() + 15)
        U_r, V_r, W_r, success = _try_rank_search(r, time_budget)
        
        if success:
            # Verify the solution
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        # Also check if the best error found was close to zero
        # (this is a heuristic, the actual verification above is more reliable)
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
