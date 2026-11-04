import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def compute_reconstruction(U, V, W, rank):
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return Rec

    def is_valid(U, V, W, rank, atol=1e-9):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.allclose(Rec, 1.0, atol=atol)

    def compute_error(U, V, W, rank):
        Rec = compute_reconstruction(U, V, W, rank)
        return np.max(np.abs(Rec - 1.0))

    def local_optimize(U, V, W, rank, max_iters=500, atol=1e-9):
        """Local optimization with structured moves and half-integer constraint."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        valid_vals = np.array([x / 2.0 for x in range(-20, 21)])
        
        Rec = compute_reconstruction(U, V, W, rank)
        best_err = np.max(np.abs(Rec - 1.0))
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_err < atol:
                break
            
            improved = False
            
            # Try adjusting U entries
            for r in range(rank):
                v_col = V[:, r].reshape(m, p)
                w_col = W[:, r].reshape(p, n)
                
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        old_val = U[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 1.5]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            # Only (i, j, k') positions change: delta * v[j, k'] * w[k', i]
                            new_rec_affected = Rec[i, j, :] + delta * v_col[j, :] * w_col[:, i]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                U[idx, r] = cand
                                Rec = compute_reconstruction(U, V, W, rank)
                                err = np.max(np.abs(Rec - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    U[idx, r] = old_val
                                    Rec = compute_reconstruction(U, V, W, rank)
                        
                        if best_candidate != old_val:
                            U[idx, r] = best_candidate
                            Rec = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting V entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                w_col = W[:, r].reshape(p, n)
                
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        old_val = V[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 1.5]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            # Only (i, j, k) positions: delta * u[i, j] * w[k, i]
                            new_rec_affected = Rec[:, j, k] + delta * u_col[:, j] * w_col[k, :]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                V[idx, r] = cand
                                Rec = compute_reconstruction(U, V, W, rank)
                                err = np.max(np.abs(Rec - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    V[idx, r] = old_val
                                    Rec = compute_reconstruction(U, V, W, rank)
                        
                        if best_candidate != old_val:
                            V[idx, r] = best_candidate
                            Rec = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            # Try adjusting W entries
            for r in range(rank):
                u_col = U[:, r].reshape(n, m)
                v_col = V[:, r].reshape(m, p)
                
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        old_val = W[idx, r]
                        
                        candidates = valid_vals[np.abs(valid_vals - old_val) < 1.5]
                        if len(candidates) == 0:
                            candidates = np.array([old_val])
                        
                        best_candidate = old_val
                        best_candidate_err = best_err
                        
                        for cand in candidates:
                            if abs(cand - old_val) < 1e-12:
                                continue
                            
                            delta = cand - old_val
                            # Only (i, j', k) positions: delta * u[i, j'] * v[j', k]
                            new_rec_affected = Rec[i, :, k] + delta * u_col[i, :] * v_col[:, k]
                            err_affected = np.max(np.abs(new_rec_affected - 1.0))
                            
                            if err_affected < best_err - 1e-10:
                                W[idx, r] = cand
                                Rec = compute_reconstruction(U, V, W, rank)
                                err = np.max(np.abs(Rec - 1.0))
                                if err < best_candidate_err - 1e-10:
                                    best_candidate_err = err
                                    best_candidate = cand
                                else:
                                    W[idx, r] = old_val
                                    Rec = compute_reconstruction(U, V, W, rank)
                        
                        if best_candidate != old_val:
                            W[idx, r] = best_candidate
                            Rec = compute_reconstruction(U, V, W, rank)
                            best_err = best_candidate_err
                            improved = True
            
            if not improved:
                break
        
        if best_err < atol:
            return U, V, W, True
        return U, V, W, False

    def structured_init(rank, seed):
        """Structured initialization with algebraic patterns."""
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        for r in range(rank):
            pattern = r % 7
            
            if pattern == 0:
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            elif pattern == 1:
                i1 = r % n
                i2 = (r + 1) % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i1 * m + j, r] = 1
                U[i2 * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i1, r] = 1
                W[k * n + i2, r] = -1
            elif pattern == 2:
                for _ in range(np.random.randint(1, 4)):
                    idx = np.random.randint(0, n * m)
                    U[idx, r] = np.random.choice([-1, 1])
                for _ in range(np.random.randint(1, 4)):
                    idx = np.random.randint(0, m * p)
                    V[idx, r] = np.random.choice([-1, 1])
                for _ in range(np.random.randint(1, 4)):
                    idx = np.random.randint(0, p * n)
                    W[idx, r] = np.random.choice([-1, 1])
            elif pattern == 3:
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([0.5, 1.0, 1.5])
                V[j * p + k, r] = np.random.choice([0.5, 1.0, 1.5])
                W[k * n + i, r] = np.random.choice([0.5, 1.0, 1.5])
            elif pattern == 4:
                U[:, r] = np.random.choice([-1, -0.5, 0.5, 1], n * m)
                V[:, r] = np.random.choice([-1, -0.5, 0.5, 1], m * p)
                W[:, r] = np.random.choice([-1, -0.5, 0.5, 1], p * n)
            elif pattern == 5:
                i = r % n
                j = r % m
                k = r % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                if np.random.random() < 0.3:
                    U[(i + 1) % n * m + j, r] = 0.5
            else:
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                U[i * m + j, r] = np.random.choice([-1, 1])
                V[j * p + k, r] = np.random.choice([-1, 1])
                W[k * n + i, r] = np.random.choice([-1, 1])
                if np.random.random() < 0.5:
                    U[(i + 1) % n * m + (j + 1) % m, r] = 0.5
        
        return U, V, W

    def random_init(rank, seed):
        """Random initialization with half-integer snapping."""
        np.random.seed(seed)
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))
        return U, V, W

    def search_for_solution(max_rank, max_search_time=100.0):
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        for rank in range(min(max_rank, best_rank - 1), 15, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            n_attempts = 15
            for attempt in range(n_attempts):
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                seed = 42 + rank * 1000 + attempt * 13
                U, V, W = structured_init(rank, seed)
                
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                U_opt, V_opt, W_opt, success = local_optimize(U, V, W, rank, max_iters=300)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
                    best_rank = rank
                    return True
                
                # Also try random init
                U_r, V_r, W_r = random_init(rank, seed + 100)
                if is_valid(U_r, V_r, W_r, rank):
                    best_solution = {"U": U_r, "V": V_r, "W": W_r}
                    best_rank = rank
                    return True
                
                U_opt2, V_opt2, W_opt2, success2 = local_optimize(U_r, V_r, W_r, rank, max_iters=300)
                if success2:
                    best_solution = {"U": U_opt2, "V": V_opt2, "W": W_opt2}
                    best_rank = rank
                    return True
        
        return False

    search_for_solution(best_rank - 1)

    def to_lists(mat):
        return mat.tolist()

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
