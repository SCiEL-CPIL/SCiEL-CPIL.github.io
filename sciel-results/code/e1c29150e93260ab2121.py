"""Improved search for rank-minimal exact tensor decomposition of 3x3 * 3x4 matrix multiplication."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Schoolbook (rank 36) as fallback.
    2. Fast verification using precomputed triple arrays and vectorized NumPy.
    3. Structured random search with multiple initialization strategies.
    4. Local search with entry-level mutations and occasional column permutations.
    5. Try ranks from 18 upward, stopping at first valid decomposition.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s budget, leaving 10s margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    # Active triples: (a, b, c) where a=i*m+j, b=j*p+k, c=k*n+i
    active_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_triples.add((a, b, c))
    
    all_triples = list(itertools.product(range(n * m), range(m * p), range(p * n)))
    active_set = set(active_triples)
    active_mask = np.array([1 if (a, b, c) in active_set else 0 for a, b, c in all_triples], dtype=np.int64)
    all_triples_arr = np.array(all_triples, dtype=np.int32)
    num_triples = len(all_triples)
    
    def tensor_from_matrices(U, V, W, r):
        """
        Compute the full tensor value for all triples using vectorized approach.
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        Returns: tensor of shape (num_triples,)
        """
        U_np = np.asarray(U, dtype=np.int64)
        V_np = np.asarray(V, dtype=np.int64)
        W_np = np.asarray(W, dtype=np.int64)
        
        # For each column ci, compute outer product contributions
        # tensor[idx] += U_np[a_idx, ci] * V_np[b_idx, ci] * W_np[c_idx, ci]
        # where idx corresponds to (a, b, c) = all_triples_arr[idx]
        
        # Vectorize over columns using broadcasting
        # This creates an array of shape (r, n*m, m*p, p*n) which is 18*9*12*12 = 23328 elements
        # That's fine.
        tensor = np.zeros(num_triples, dtype=np.int64)
        for ci in range(r):
            u_col = U_np[:, ci]  # (n*m,)
            v_col = V_np[:, ci]  # (m*p,)
            w_col = W_np[:, ci]  # (p*n,)
            # Compute u[a]*v[b]*w[c] for all (a,b,c) triples
            # Use advanced indexing: for each idx, a=all_triples_arr[idx,0], etc.
            a_vals = all_triples_arr[:, 0]
            b_vals = all_triples_arr[:, 1]
            c_vals = all_triples_arr[:, 2]
            contributions = u_col[a_vals] * v_col[b_vals] * w_col[c_vals]
            tensor += contributions
        return tensor
    
    def verify_full(U, V, W, r):
        """Full verification: check all triples."""
        tensor = tensor_from_matrices(U, V, W, r)
        return bool(np.all(tensor == active_mask))
    
    def error_count(U, V, W, r):
        """Count number of incorrect triples."""
        tensor = tensor_from_matrices(U, V, W, r)
        return int(np.sum(tensor != active_mask))
    
    def active_errors_only(U, V, W, r):
        """Check only active triples (faster pre-filter)."""
        U_np = np.asarray(U, dtype=np.int64)
        V_np = np.asarray(V, dtype=np.int64)
        W_np = np.asarray(W, dtype=np.int64)
        
        errors = 0
        for (a, b, c) in active_triples:
            s = 0
            for ci in range(r):
                s += U_np[a, ci] * V_np[b, ci] * W_np[c, ci]
            if s != 1:
                errors += 1
        return errors
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def try_rank(r, max_attempts=200):
        """Try to find a rank-r decomposition using structured random search."""
        if time.time() > deadline:
            return None
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            strategy = attempt % 6
            
            if strategy == 0:
                # Very sparse: each row has 1-2 nonzeros
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nz = random.randint(1, 2)
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
                        
            elif strategy == 1:
                # Sparse with ±1, density ~r/3
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    num_nz = random.randint(1, max(1, r // 3))
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    num_nz = random.randint(1, max(1, r // 3))
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nz = random.randint(1, max(1, r // 3))
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            elif strategy == 2:
                # Random ±1 with some zeros (density ~0.5)
                U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
                V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
                W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(p * n)]
                
            elif strategy == 3:
                # Column-structured: each column gets a random pattern
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for ci in range(r):
                    u_pat = [random.choice([-1, 0, 1]) for _ in range(n * m)]
                    v_pat = [random.choice([-1, 0, 1]) for _ in range(m * p)]
                    w_pat = [random.choice([-1, 0, 1]) for _ in range(p * n)]
                    for a in range(n * m):
                        U[a][ci] = u_pat[a]
                    for b in range(m * p):
                        V[b][ci] = v_pat[b]
                    for c in range(p * n):
                        W[c][ci] = w_pat[c]
            
            elif strategy == 4:
                # Bipartite sparse: split rows into groups
                U = [[0] * r for _ in range(n * m)]
                V = [[0] * r for _ in range(m * p)]
                W = [[0] * r for _ in range(p * n)]
                for a in range(n * m):
                    half = r // 2
                    side = a % 2
                    if side == 0:
                        cols = list(range(0, half))
                    else:
                        cols = list(range(half, r))
                    num_nz = random.randint(1, max(1, len(cols) // 2))
                    chosen = random.sample(cols, min(num_nz, len(cols)))
                    for ci in chosen:
                        U[a][ci] = random.choice([-1, 1])
                for b in range(m * p):
                    half = r // 2
                    side = b % 2
                    if side == 0:
                        cols = list(range(0, half))
                    else:
                        cols = list(range(half, r))
                    num_nz = random.randint(1, max(1, len(cols) // 2))
                    chosen = random.sample(cols, min(num_nz, len(cols)))
                    for ci in chosen:
                        V[b][ci] = random.choice([-1, 1])
                for c in range(p * n):
                    num_nz = random.randint(1, max(1, r // 4))
                    cols = random.sample(range(r), num_nz)
                    for ci in cols:
                        W[c][ci] = random.choice([-1, 1])
            
            else:
                # Dense with small integers
                scale = 2
                U = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(n * m)]
                V = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(m * p)]
                W = [[random.randint(-scale, scale) for _ in range(r)] for _ in range(p * n)]
            
            # Quick pre-filter: check active triples only
            if active_errors_only(U, V, W, r) > 0:
                continue
            
            # Full verification
            if verify_full(U, V, W, r):
                return {"U": U, "V": V, "W": W}
        
        return None
    
    def local_search(U, V, W, r, max_steps=80):
        """
        Local search: mutate single entries, accept if error count decreases.
        Also try column reordering and sign flips of entire columns.
        """
        if time.time() > deadline:
            return None
        
        current_errors = error_count(U, V, W, r)
        if current_errors == 0:
            return {"U": U, "V": V, "W": W}
        
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        best_W = [row[:] for row in W]
        best_errors = current_errors
        no_improve = 0
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if best_errors == 0:
                break
            
            improved = False
            # Try several random mutations
            for _ in range(80):
                if time.time() > deadline:
                    break
                
                mat_choice = random.choice(['U', 'V', 'W'])
                
                if mat_choice == 'U':
                    idx = random.randint(0, n * m - 1)
                    ci = random.randint(0, r - 1)
                    old_val = U[idx][ci]
                    # Try changing to a different value
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    U[idx][ci] = new_val
                    errors = error_count(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        U[idx][ci] = old_val
                elif mat_choice == 'V':
                    idx = random.randint(0, m * p - 1)
                    ci = random.randint(0, r - 1)
                    old_val = V[idx][ci]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    V[idx][ci] = new_val
                    errors = error_count(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        V[idx][ci] = old_val
                else:
                    idx = random.randint(0, p * n - 1)
                    ci = random.randint(0, r - 1)
                    old_val = W[idx][ci]
                    new_val = random.choice([-2, -1, 0, 1, 2])
                    if new_val == old_val:
                        continue
                    W[idx][ci] = new_val
                    errors = error_count(U, V, W, r)
                    if errors < best_errors:
                        best_errors = errors
                        best_U = [row[:] for row in U]
                        best_V = [row[:] for row in V]
                        best_W = [row[:] for row in W]
                        improved = True
                        no_improve = 0
                    else:
                        W[idx][ci] = old_val
            
            if improved:
                # Continue from improved state
                U = [row[:] for row in best_U]
                V = [row[:] for row in best_V]
                W = [row[:] for row in best_W]
            else:
                no_improve += 1
                if no_improve > 3:
                    # Reset to best and perturb
                    U = [row[:] for row in best_U]
                    V = [row[:] for row in best_V]
                    W = [row[:] for row in best_W]
                    no_improve = 0
            
            if best_errors == 0:
                break
        
        if best_errors == 0:
            return {"U": best_U, "V": best_V, "W": best_W}
        return None
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*m, m*p, p*n) = max(9, 12, 12) = 12
    # For 3x3x3, Strassen gives 7. For 3x3x4, lower bound is at least 12.
    # Try from 12 upward, but be more aggressive with lower ranks.
    min_rank = max(n * m, m * p, p * n)  # 12
    
    # Use time budget to decide how many attempts per rank
    time_budget = deadline - start_time
    ranks_to_try = list(range(min_rank, best_rank + 1))
    
    # Prioritize lower ranks more
    if len(ranks_to_try) > 0:
        # Allocate more time to lower ranks
        weights = [max(1, best_rank - r + 1) for r in ranks_to_try]
        total_weight = sum(weights)
        
        for r in ranks_to_try:
            if time.time() > deadline:
                break
            
            time_remaining = deadline - time.time()
            if time_remaining < 3:
                break
            
            # Allocate attempts based on remaining time and rank priority
            weight = best_rank - r + 1
            max_attempts = min(500, max(5, int(time_remaining * weight / (total_weight * 0.8))))
            
            result = try_rank(r, max_attempts)
            if result is not None:
                best_result = result
                best_rank = r
                break
    # If no random solution found, try local search starting from random init at lowest feasible rank
    if best_rank == n * m * p:
        # Try local search at a few ranks
        for r in [min_rank, min_rank + 2, min_rank + 4]:
            if time.time() > deadline - 5:
                break
            if r >= best_rank:
                break
            
            # Initialize with sparse random
            U_start = [[0] * r for _ in range(n * m)]
            V_start = [[0] * r for _ in range(m * p)]
            W_start = [[0] * r for _ in range(p * n)]
            for a in range(n * m):
                num_nz = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nz)
                for ci in cols:
                    U_start[a][ci] = random.choice([-1, 1])
            for b in range(m * p):
                num_nz = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nz)
                for ci in cols:
                    V_start[b][ci] = random.choice([-1, 1])
            for c in range(p * n):
                num_nz = random.randint(1, min(3, r))
                cols = random.sample(range(r), num_nz)
                for ci in cols:
                    W_start[c][ci] = random.choice([-1, 1])
            
            ls_result = local_search(U_start, V_start, W_start, r, max_steps=60)
            if ls_result is not None:
                best_result = ls_result
                best_rank = r
                break
    
    return best_result
# EVOLVE-BLOCK-END
