import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Precompute target tensor
    target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        R = np.einsum('ac,bc,dc->abd', U, V, W)
        if np.max(np.abs(R - target)) > tol:
            return False
        # Check integer/half-integer
        for mat in [U, V, W]:
            # Check if val * 2 is close to an integer
            if not np.allclose(mat * 2, np.round(mat * 2), atol=tol):
                return False
        return True

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def compute_R(U, V, W):
        return np.einsum('ac,bc,dc->abd', U, V, W)

    def compute_error(R):
        return np.sum((R - target) ** 2)

    def local_search(rank, max_iters=50000, T_init=1.0, T_min=0.01):
        # Initialize with random half-integers
        # Values in {-1, 0, 1} or {-0.5, 0, 0.5}
        # Let's start with small values
        U = np.random.choice([-0.5, 0, 0.5], size=(nm, rank))
        V = np.random.choice([-0.5, 0, 0.5], size=(mp, rank))
        W = np.random.choice([-0.5, 0, 0.5], size=(pn, rank))
        
        R = compute_R(U, V, W)
        current_error = compute_error(R)
        
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_R = R.copy()
        
        T = T_init
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Choose random matrix and index
            mat_idx = np.random.randint(3) # 0: U, 1: V, 2: W
            if mat_idx == 0:
                i = np.random.randint(nm)
                c = np.random.randint(rank)
                # Change U[i, c]
                # Delta is +/- 0.5
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                
                # Update R
                # R[i, :, :] -= delta * outer(V[:, c], W[:, c])
                update = delta * np.outer(V[:, c], W[:, c])
                R[i, :, :] -= update
                
                # Calculate new error for slice i
                # E_old_slice = sum((target[i] - R_old[i])^2) -- we don't have R_old
                # We can calculate delta_error
                # E_new = E_old - sum((target[i] - R_old[i])^2) + sum((target[i] - R_new[i])^2)
                # R_old = R_new + update
                # This is tricky without storing R_old.
                # Let's just recalculate error for the slice.
                # But we need global E.
                # Let's maintain E.
                # E_old_slice = sum((target[i] - (R[i] + update))^2)
                # E_new_slice = sum((target[i] - R[i])^2)
                
                # Actually, simpler:
                # Store R before update? No, we updated in place.
                # Let's calculate delta_E directly.
                # Let x = target[i], r_new = R[i], r_old = r_new + update
                # term_new = (x - r_new)^2
                # term_old = (x - r_old)^2
                # delta_E = sum(term_new - term_old)
                
                # We have R[i] as r_new.
                # r_old = R[i] + update
                r_old = R[i, :, :] + update
                r_new = R[i, :, :]
                
                term_new = (target[i, :, :] - r_new) ** 2
                term_old = (target[i, :, :] - r_old) ** 2
                delta_E = np.sum(term_new - term_old)
                
                # Acceptance
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    current_error += delta_E
                    U[i, c] += delta
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_R = R.copy()
                else:
                    # Revert R
                    R[i, :, :] += update
                    # U was not changed
                    
            elif mat_idx == 1:
                j = np.random.randint(mp)
                c = np.random.randint(rank)
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                
                # R[:, j, :] -= delta * outer(U[:, c], W[:, c])
                update = delta * np.outer(U[:, c], W[:, c])
                R[:, j, :] -= update
                
                r_old = R[:, j, :] + update
                r_new = R[:, j, :]
                
                term_new = (target[:, j, :] - r_new) ** 2
                term_old = (target[:, j, :] - r_old) ** 2
                delta_E = np.sum(term_new - term_old)
                
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    current_error += delta_E
                    V[j, c] += delta
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_R = R.copy()
                else:
                    R[:, j, :] += update
                    
            else:
                k = np.random.randint(pn)
                c = np.random.randint(rank)
                delta = 0.5 if np.random.random() < 0.5 else -0.5
                
                # R[:, :, k] -= delta * outer(U[:, c], V[:, c])
                update = delta * np.outer(U[:, c], V[:, c])
                R[:, :, k] -= update
                
                r_old = R[:, :, k] + update
                r_new = R[:, :, k]
                
                term_new = (target[:, :, k] - r_new) ** 2
                term_old = (target[:, :, k] - r_old) ** 2
                delta_E = np.sum(term_new - term_old)
                
                if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                    current_error += delta_E
                    W[k, c] += delta
                    if current_error < best_error:
                        best_error = current_error
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_R = R.copy()
                else:
                    R[:, :, k] += update

            T *= cooling
            if T < T_min:
                T = T_min

        return best_U, best_V, best_W, best_error

    best_result = None
    best_rank = 42

    # Try ranks from low to high
    # We will try a few attempts for each rank
    for rank in range(25, 42):
        if time.time() > deadline - 20:
            break
            
        # Try a few random restarts
        attempts = 5 if rank < 35 else 10
        for attempt in range(attempts):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(hash((rank, attempt)) % 2**32)
            U, V, W, error = local_search(rank, max_iters=20000)
            
            if error < 1e-9:
                # Verify
                if verify(U, V, W):
                    best_result = (U, V, W)
                    best_rank = rank
                    break
        
        if best_result is not None:
            break

    if best_result is None:
        U, V, W = schoolbook()
        best_result = (U, V, W)
        best_rank = 42

    U, V, W = best_result
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
