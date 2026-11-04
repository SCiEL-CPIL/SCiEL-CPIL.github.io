import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    target = np.zeros((nm, mp, pn), dtype=float)
    for c_r in range(n):
        for b in range(m):
            for c_c in range(p):
                i = c_r * m + b
                k = b * p + c_c
                l = c_c * n + c_r
                target[i, k, l] = 1.0
    
    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)
    
    def get_strassen_66():
        rank = 66
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            U[A00, col] += 1; U[A11, col] += 1
            V[B00, col] += 1; V[B11, col] += 1
            W[C00, col] += 1; W[C11, col] += 1
            col += 1
            U[A10, col] += 1; U[A11, col] += 1
            V[B00, col] += 1
            W[C10, col] += 1; W[C11, col] -= 1
            col += 1
            U[A00, col] += 1
            V[B01, col] += 1; V[B11, col] -= 1
            W[C01, col] += 1; W[C11, col] += 1
            col += 1
            U[A11, col] += 1
            V[B10, col] += 1; V[B00, col] -= 1
            W[C00, col] += 1; W[C10, col] += 1
            col += 1
            U[A00, col] += 1; U[A01, col] += 1
            V[B11, col] += 1
            W[C00, col] -= 1; W[C01, col] += 1
            col += 1
            U[A10, col] += 1; U[A00, col] -= 1
            V[B00, col] += 1; V[B01, col] += 1
            W[C11, col] += 1
            col += 1
            U[A01, col] += 1; U[A11, col] -= 1
            V[B10, col] += 1; V[B11, col] += 1
            W[C00, col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            U[A0, col] += 1; V[B00, col] += 1; W[C0, col] += 1
            col += 1
            U[A1, col] += 1; V[B10, col] += 1; W[C0, col] += 1
            col += 1
            U[A0, col] += 1; V[B01, col] += 1; W[C1, col] += 1
            col += 1
            U[A1, col] += 1; V[B11, col] += 1; W[C1, col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U[:, :col], V[:, :col], W[:, :col]
    
    def compute_column_importance(U, V, W):
        """Compute how 'important' each column is by measuring its unique contribution."""
        rank = U.shape[1]
        importance = np.zeros(rank)
        
        # For each column c, compute the norm of u_c ⊗ v_c ⊗ w_c
        # and also how much it overlaps with other columns
        for c in range(rank):
            # Norm of the rank-1 tensor
            norm = np.linalg.norm(U[:, c]) * np.linalg.norm(V[:, c]) * np.linalg.norm(W[:, c])
            importance[c] = norm
        
        # Also compute pairwise overlaps to identify redundant columns
        # If two columns have very similar (u⊗v⊗w) directions, one may be removable
        return importance
    
    def greedy_prune(U, V, W, max_iter=500, seed=None, importance_order=None):
        """Aggressive greedy pruning with optional importance-based ordering."""
        if seed is not None:
            random.seed(seed)
            
        current_U = np.array(U, dtype=float).copy()
        current_V = np.array(V, dtype=float).copy()
        current_W = np.array(W, dtype=float).copy()
        current_rank = current_U.shape[1]
        
        # Build initial ordering
        if importance_order is not None:
            order = importance_order
        else:
            order = list(range(current_rank))
            random.shuffle(order)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
            
            removed_any = False
            
            for col_idx in order:
                if time.time() > deadline - 5.0:
                    break
                if col_idx >= current_rank:
                    continue
                
                # Create new arrays without this column
                mask = [i for i in range(current_rank) if i != col_idx]
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]
                
                # Quick check using numpy
                R = np.einsum('ic,kc,lc->ikl', U_new, V_new, W_new)
                
                if np.allclose(R, target, atol=1e-9):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    # Update order: remove the deleted column and shuffle remaining
                    order = [i for i in order if i != col_idx]
                    # Also add columns that were shifted
                    for i in range(current_rank):
                        if i not in order:
                            order.append(i)
                    random.shuffle(order)
                    break  # Restart scanning with new order
            
            if not removed_any:
                # Try a fresh random order
                order = list(range(current_rank))
                random.shuffle(order)
                if iteration > 50:  # Give up after many failed restarts
                    break
        
        return current_U, current_V, current_W, current_rank
    
    # Get the rank-66 solution
    U66, V66, W66 = get_strassen_66()
    
    # Verify it's correct
    assert verify(U66, V66, W66), "Rank-66 decomposition failed verification"
    
    best_U, best_V, best_W, best_rank = U66, V66, W66, 66
    
    # Strategy 1: Simple greedy pruning with multiple random restarts
    for restart in range(30):
        if time.time() > deadline - 15.0:
            break
        
        seed_val = restart * 7919 + 12345
        U_p, V_p, W_p, r_p = greedy_prune(U66, V66, W66, max_iter=200, seed=seed_val)
        
        if verify(U_p, V_p, W_p):
            if r_p < best_rank:
                best_rank = r_p
                best_U, best_V, best_W = U_p, V_p, W_p
    
    # Strategy 2: Importance-based ordering
    if time.time() < deadline - 20.0:
        importance = compute_column_importance(U66, V66, W66)
        # Try removing columns in order of lowest importance first
        order = np.argsort(importance).tolist()
        
        for restart in range(10):
            if time.time() > deadline - 10.0:
                break
            
            # Perturb the importance order slightly
            perturbed_order = order[:]
            for _ in range(5):
                i, j = random.randint(0, len(perturbed_order)-1), random.randint(0, len(perturbed_order)-1)
                perturbed_order[i], perturbed_order[j] = perturbed_order[j], perturbed_order[i]
            
            U_p, V_p, W_p, r_p = greedy_prune(U66, V66, W66, max_iter=200, seed=restart*31+7, importance_order=perturbed_order)
            
            if verify(U_p, V_p, W_p):
                if r_p < best_rank:
                    best_rank = r_p
                    best_U, best_V, best_W = U_p, V_p, W_p
    
    # Strategy 3: If we got to 65, try to go further
    if best_rank <= 65 and time.time() < deadline - 10.0:
        for restart in range(20):
            if time.time() > deadline - 5.0:
                break
            
            seed_val = restart * 104729 + 99999
            U_p, V_p, W_p, r_p = greedy_prune(best_U, best_V, best_W, max_iter=100, seed=seed_val)
            
            if verify(U_p, V_p, W_p):
                if r_p < best_rank:
                    best_rank = r_p
                    best_U, best_V, best_W = U_p, V_p, W_p
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
