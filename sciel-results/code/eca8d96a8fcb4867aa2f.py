"""Matrix multiplication 3x4x6 rank minimization via combinatorial construction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 115.0  # safety margin

    n, m, p = DIMENSIONS  # 3, 4, 6
    # Schoolbook: rank = n*m*p = 72
    # We try to find a better rank via known results and construction.
    
    # Known bounds for R(3,4,6):
    # The standard bound is n*m*p = 72.
    # There's a known result: R(m,n,p) <= m*n + n*p + m*p - something...
    # Actually for small matrices, the best known is often just schoolbook
    # unless there's a special structure.
    
    # Let's try a constructive approach using bilinear complexity.
    # For 3x4x6, we can try to use the fact that one dimension is larger.
    
    # Strategy: Try to find a decomposition of rank < 72.
    # We'll use a simple randomized approach with verification.
    
    def verify_decomposition(U, V, W, n, m, p):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        rank = len(U[0])
        # Build the tensor T[i,j,k] = 1 if the product entry (i,k) in A@B uses A[i,j]*B[j,k]
        # Actually, the multiplication tensor has T[i,j,k] = 1 where i in [0,n), j in [0,m), k in [0,p)
        # The decomposition says: for each output (i,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Wait, the convention is W[k*n+i, r].
        
        # Let me re-read: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
        # The condition is: for all (i,j,k) with j in [0,m), the coefficient of A[i,j]*B[j,k] in C[i,k]
        # should be 1 if we're computing standard matrix multiplication.
        
        # Actually, the tensor for matrix multiplication is:
        # T_{i,j,k} = 1 for each triple (i,j,k) where i < n, j < m, k < p
        # and the decomposition must satisfy:
        # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
        
        # Let's verify this.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1  # Each term A[i,j]*B[j,k] appears exactly once in C[i,k]
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to improve using random initialization and gradient-free search
    # This is a non-convex problem, so we'll use random restarts with local search.
    
    import random
    random.seed(42)
    
    def random_decomposition(rank, scale=1.0):
        """Generate a random decomposition with integer/half-integer entries."""
        U = [[random.randint(-2, 2) * scale for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) * scale for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) * scale for _ in range(rank)] for _ in range(p * n)]
        return U, V, W
    
    def count_errors(U, V, W, n, m, p):
        """Count how many (i,j,k) triples are incorrectly decomposed."""
        rank = len(U[0])
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for c in range(rank):
                        actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(actual - 1.0) > 1e-9:
                        errors += 1
        return errors
    
    # Try ranks from best possible upward
    # Lower bound: at least n*p = 18 (since each output needs at least one term)
    # Actually the lower bound is max(n*m, m*p, n*p) but that's not tight.
    # For now, try ranks from 36 to 71.
    
    min_possible = 18  # n*p, very loose
    max_try = best_rank - 1
    
    for target_rank in range(min_possible, max_try + 1):
        if time.time() > deadline - 5.0:
            break
            
        found = False
        # Try a few random restarts for this rank
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            U, V, W = random_decomposition(target_rank)
            
            # Simple local search: flip individual entries to reduce errors
            errors = count_errors(U, V, W, n, m, p)
            
            max_steps = 100
            for step in range(max_steps):
                if time.time() > deadline - 5.0:
                    break
                if errors == 0:
                    found = True
                    break
                    
                # Pick a random entry to modify
                choice = random.choice(['U', 'V', 'W'])
                if choice == 'U':
                    row = random.randint(0, n*m - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = U[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    U[row][col_idx] = new_val
                elif choice == 'V':
                    row = random.randint(0, m*p - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = V[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    V[row][col_idx] = new_val
                else:
                    row = random.randint(0, p*n - 1)
                    col_idx = random.randint(0, target_rank - 1)
                    old_val = W[row][col_idx]
                    new_val = old_val + random.choice([-1, 1])
                    W[row][col_idx] = new_val
                    
                new_errors = count_errors(U, V, W, n, m, p)
                if new_errors > errors:
                    # Revert
                    if choice == 'U':
                        U[row][col_idx] = old_val
                    elif choice == 'V':
                        V[row][col_idx] = old_val
                    else:
                        W[row][col_idx] = old_val
                else:
                    errors = new_errors
            
            if found:
                # Verify
                if verify_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
        
        if found:
            break
    
    if time.time() > deadline:
        pass  # Just use best found so far
    
    return best_result
# EVOLVE-BLOCK-END
