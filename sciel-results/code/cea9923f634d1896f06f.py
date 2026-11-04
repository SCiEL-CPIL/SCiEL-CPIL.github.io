import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    # Lower bound is max(n*m, m*p, p*n) = 15
    lower_bound = max(n*m, m*p, p*n)
    
    def lattice_search(rank, max_iters=5000, restarts=10):
        """
        Perform random walk on integer/half-integer lattice to find exact decomposition.
        """
        for _ in range(restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with random half-integers or integers
            # Using small range to start near 0
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Compute initial T_approx
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T - T_approx
            curr_error = np.sum(diff**2)
            
            if curr_error < 1e-10:
                return U, V, W
            
            temperature = 1.0
            cooling = 0.999
            best_error = curr_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random non-zero element in diff
                # To be efficient, we can just pick random a,b,c and if diff is 0, skip
                # Or maintain a list of non-zero indices. Given small size, random pick is fine.
                a = np.random.randint(n*m)
                b = np.random.randint(m*p)
                c = np.random.randint(p*n)
                
                if abs(diff[a,b,c]) < 1e-10:
                    continue
                
                # Pick a random rank component r to modify
                r = np.random.randint(rank)
                
                # Decide which matrix to modify
                mod_choice = np.random.randint(3)
                
                old_val = 0.0
                new_val = 0.0
                delta = 0.0
                
                if mod_choice == 0:
                    old_val = U[a, r]
                    # Try +/- 0.5 or +/- 1
                    step = np.random.choice([0.5, 1.0]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                elif mod_choice == 1:
                    old_val = V[b, r]
                    step = np.random.choice([0.5, 1.0]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                else:
                    old_val = W[c, r]
                    step = np.random.choice([0.5, 1.0]) * np.random.choice([-1, 1])
                    new_val = old_val + step
                    delta = step
                    
                # Calculate change in error
                # We only need to update the error for the affected slice
                # If we change U[a,r] by delta, T_approx[a,b',c'] changes by delta * V[b',r] * W[c',r]
                # New error for a,b',c' is (T[a,b',c'] - T_approx[a,b',c'] - delta*V[b',r]*W[c',r])^2
                # Old error is (T[a,b',c'] - T_approx[a,b',c'])^2
                
                if mod_choice == 0:
                    # Affected indices: all b', c'
                    # We can compute the change in sum of squares
                    # Let d = T - T_approx
                    # new_d = d - delta * (V[:,r] * W[:,r])  (broadcast over b,c)
                    # change = sum((new_d)^2 - d^2)
                    # To be fast, we just check if the specific (a,b,c) improved? 
                    # No, global check is better for SA.
                    # But calculating full sum is O(mp).
                    # Let's do a cheap check: does it improve the current element (a,b,c)?
                    # If yes, accept. If no, maybe accept with probability exp(-delta_E/T)
                    
                    # Calculate delta_E for the whole tensor? 
                    # It's 225 multiplications. 5000 iters * 225 = 1.1M ops. Very fast.
                    # Let's just compute the full error change.
                    
                    # Update T_approx in place? No, we need to revert if rejected.
                    # Let's compute new_error directly.
                    # new_T_approx = T_approx + delta * np.outer(V[:,r], W[:,r])[:,:,a] ? 
                    # No, U is first dim.
                    # T_approx[a, b', c'] += delta * V[b', r] * W[c', r]
                    
                    # We can compute the change in error for the slice a
                    slice_old = diff[a] # shape (m*p, p*n)
                    update_vec = V[:, r] * W[:, r] # shape (m*p, p*n)
                    slice_new = slice_old - delta * update_vec
                    
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)
                    
                elif mod_choice == 1:
                    # Change V[b,r]. Affects all a', c'.
                    # T_approx[a', b, c'] += delta * U[a', r] * W[c', r]
                    slice_old = diff[:, b, :] # shape (n*m, p*n)
                    update_vec = U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :] # (n*m, p*n)
                    slice_new = slice_old - delta * update_vec
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)
                    
                else:
                    # Change W[c,r]. Affects all a', b'.
                    # T_approx[a', b', c] += delta * U[a', r] * V[b', r]
                    slice_old = diff[:, :, c] # shape (n*m, m*p)
                    update_vec = U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :] # (n*m, m*p)
                    slice_new = slice_old - delta * update_vec
                    delta_err = np.sum(slice_new**2) - np.sum(slice_old**2)

                # Accept if delta_err < 0 or with probability exp(-delta_err / temperature)
                if delta_err < 0:
                    # Accept
                    if mod_choice == 0:
                        U[a, r] = new_val
                        diff[a] -= delta * (V[:, r] * W[:, r])
                    elif mod_choice == 1:
                        V[b, r] = new_val
                        diff[:, b, :] -= delta * (U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :])
                    else:
                        W[c, r] = new_val
                        diff[:, :, c] -= delta * (U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :])
                    
                    curr_error += delta_err
                    if curr_error < best_error:
                        best_error = curr_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        
                    if curr_error < 1e-10:
                        return U, V, W
                        
                else:
                    # Reject or accept with SA probability
                    if temperature > 1e-6:
                        prob = np.exp(-delta_err / temperature)
                        if np.random.random() < prob:
                            if mod_choice == 0:
                                U[a, r] = new_val
                                diff[a] -= delta * (V[:, r] * W[:, r])
                            elif mod_choice == 1:
                                V[b, r] = new_val
                                diff[:, b, :] -= delta * (U[:, r][:, np.newaxis] * W[:, r][np.newaxis, :])
                            else:
                                W[c, r] = new_val
                                diff[:, :, c] -= delta * (U[:, r][:, np.newaxis] * V[:, r][np.newaxis, :])
                            curr_error += delta_err
                            
                temperature *= cooling
                
            # End of iters for this restart
            # Check best found
            if best_error < 1e-10:
                return best_U, best_V, best_W
                
        return None

    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Try to find a solution for this rank
        # Increase restarts for lower ranks? Or just fixed?
        # Let's try 20 restarts per rank.
        result = lattice_search(target_rank, max_iters=2000, restarts=20)
        
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return best_result
# EVOLVE-BLOCK-END
