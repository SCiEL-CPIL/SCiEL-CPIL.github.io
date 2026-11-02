"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    # Build schoolbook decomposition as fallback
    U = np.zeros((n * m, schoolbook_rank))
    V = np.zeros((m * p, schoolbook_rank))
    W = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify_decomposition(U_mat, V_mat, W_mat, rank, tol=1e-6):
        """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j, :rank] * V_mat[j * p + k, :rank] * W_mat[k * n + i, :rank])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def check_nice_entries(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val - round(val)) > tol and abs(val * 2 - round(val * 2)) > tol:
                    return False
        return True
    
    def to_nice_list(arr):
        """Convert array to list with integer/half-integer entries."""
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                if abs(val - round(val)) < 1e-8:
                    row.append(int(round(val)))
                elif abs(val * 2 - round(val * 2)) < 1e-8:
                    row.append(round(val * 2) / 2.0)
                else:
                    # Round to nearest half-integer as best effort
                    row.append(round(val * 2) / 2.0)
            result.append(row)
        return result
    
    def compute_error(U_mat, V_mat, W_mat, rank):
        """Compute max error across all tensor entries."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_mat[i * m + j, :rank] * V_mat[j * p + k, :rank] * W_mat[k * n + i, :rank])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def try_local_search(U0, V0, W0, rank, max_iters=5000, tol=1e-8):
        """Try to reduce error via coordinate descent on half-integer lattice."""
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        current_err = compute_error(U, V, W, rank)
        if current_err < tol:
            return U, V, W, True
        
        # Precompute indices for faster access
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i * m + j, j * p + k, k * n + i))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            for idx in range(len(indices)):
                if time.time() > deadline - 10:
                    break
                
                ui, vi, wi = indices[idx]
                
                # Try adjusting U[ui, r] for each r
                for r in range(rank):
                    old_u = U[ui, r]
                    v_val = V[vi, r]
                    w_val = W[wi, r]
                    
                    # Try nearby half-integers
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_u = old_u + delta
                        if abs(new_u) > 3.0:  # Keep values bounded
                            continue
                        U[ui, r] = new_u
                        new_err = compute_error(U, V, W, rank)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[ui, r] = old_u
                    
                    if current_err < tol:
                        return U, V, W, True
                
                # Try adjusting V[vi, r]
                for r in range(rank):
                    old_v = V[vi, r]
                    u_val = U[ui, r]
                    w_val = W[wi, r]
                    
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_v = old_v + delta
                        if abs(new_v) > 3.0:
                            continue
                        V[vi, r] = new_v
                        new_err = compute_error(U, V, W, rank)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[vi, r] = old_v
                    
                    if current_err < tol:
                        return U, V, W, True
                
                # Try adjusting W[wi, r]
                for r in range(rank):
                    old_w = W[wi, r]
                    u_val = U[ui, r]
                    v_val = V[vi, r]
                    
                    for delta in [-1.0, -0.5, 0.5, 1.0]:
                        new_w = old_w + delta
                        if abs(new_w) > 3.0:
                            continue
                        W[wi, r] = new_w
                        new_err = compute_error(U, V, W, rank)
                        if new_err < current_err - 1e-12:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[wi, r] = old_w
                    
                    if current_err < tol:
                        return U, V, W, True
            
            if not improved:
                break
        
        return U, V, W, current_err < tol
    
    # Try to find lower rank decompositions
    for target_rank in range(schoolbook_rank - 1, max(60, schoolbook_rank - 15), -1):
        if time.time() > deadline - 20:
            break
        
        max_attempts = 20 if target_rank >= 75 else 10
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            U_try = np.zeros((n * m, target_rank))
            V_try = np.zeros((m * p, target_rank))
            W_try = np.zeros((p * n, target_rank))
            
            # Different initialization strategies
            if attempt % 3 == 0:
                # Strategy 1: First target_rank schoolbook terms
                for r in range(target_rank):
                    k = r % p
                    j = (r // p) % m
                    i = (r // (p * m)) % n
                    U_try[i * m + j, r] = 1.0
                    V_try[j * p + k, r] = 1.0
                    W_try[k * n + i, r] = 1.0
            elif attempt % 3 == 1:
                # Strategy 2: Random subset of schoolbook terms
                all_terms = list(range(schoolbook_rank))
                np.random.seed(attempt * 7919)
                chosen = np.random.choice(all_terms, size=target_rank, replace=False)
                for r, orig_idx in enumerate(chosen):
                    k = orig_idx % p
                    j = (orig_idx // p) % m
                    i = (orig_idx // (p * m)) % n
                    U_try[i * m + j, r] = 1.0
                    V_try[j * p + k, r] = 1.0
                    W_try[k * n + i, r] = 1.0
            else:
                # Strategy 3: Schoolbook with some random half-integer perturbations
                for r in range(target_rank):
                    k = r % p
                    j = (r // p) % m
                    i = (r // (p * m)) % n
                    U_try[i * m + j, r] = 1.0
                    V_try[j * p + k, r] = 1.0
                    W_try[k * n + i, r] = 1.0
                
                # Add small random half-integer perturbations
                np.random.seed(attempt * 104729)
                for _ in range(target_rank // 4):
                    r = np.random.randint(target_rank)
                    ui = np.random.randint(n * m)
                    U_try[ui, r] += np.random.choice([-0.5, 0.5])
                    vi = np.random.randint(m * p)
                    V_try[vi, r] += np.random.choice([-0.5, 0.5])
                    wi = np.random.randint(p * n)
                    W_try[wi, r] += np.random.choice([-0.5, 0.5])
            
            # Run local search to try to fix the decomposition
            U_ls, V_ls, W_ls, success = try_local_search(U_try, V_try, W_try, target_rank)
            
            if success:
                if check_nice_entries(U_ls) and check_nice_entries(V_ls) and check_nice_entries(W_ls):
                    best_U, best_V, best_W = U_ls.copy(), V_ls.copy(), W_ls.copy()
                    best_rank = target_rank
                    break  # Found valid decomposition at this rank
        
        if best_rank == target_rank:
            continue
    
    # Final verification and conversion
    if not verify_decomposition(best_U, best_V, best_W, best_rank):
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = schoolbook_rank
    
    U_list = to_nice_list(best_U[:, :best_rank])
    V_list = to_nice_list(best_V[:, :best_rank])
    W_list = to_nice_list(best_W[:, :best_rank])
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
