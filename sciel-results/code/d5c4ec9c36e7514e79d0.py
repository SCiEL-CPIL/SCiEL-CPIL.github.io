import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

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

    def local_optimize(U, V, W, rank, max_iters=2000, atol=1e-9):
        """
        Local optimization with multi-entry moves and structured perturbations.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        valid_vals = np.array([x / 2.0 for x in range(-12, 13)])
        
        Rec = compute_reconstruction(U, V, W, rank)
        current_err = np.max(np.abs(Rec - 1.0))
        
        if current_err < atol:
            return U, V, W, True

        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if current_err < atol:
                break
            
            improved = False
            move_type = np.random.randint(4)  # 0: single U, 1: single V, 2: single W, 3: pair move
            
            if move_type == 3:
                # Pair move: change two entries in different matrices
                mat1 = np.random.randint(3)
                mat2 = np.random.randint(3)
                if mat1 == mat2:
                    mat2 = (mat1 + 1) % 3
                
                r = np.random.randint(rank)
                
                # Get indices for mat1
                if mat1 == 0:
                    idx1 = np.random.randint(n * m)
                    i1, j1 = idx1 // m, idx1 % m
                    old1 = U[idx1, r]
                    candidates1 = [c for c in valid_vals if abs(c - old1) <= 1.0]
                elif mat1 == 1:
                    idx1 = np.random.randint(m * p)
                    j1, k1 = idx1 // p, idx1 % p
                    old1 = V[idx1, r]
                    candidates1 = [c for c in valid_vals if abs(c - old1) <= 1.0]
                else:
                    idx1 = np.random.randint(p * n)
                    k1, i1 = idx1 // n, idx1 % n
                    old1 = W[idx1, r]
                    candidates1 = [c for c in valid_vals if abs(c - old1) <= 1.0]
                
                # Get indices for mat2
                if mat2 == 0:
                    idx2 = np.random.randint(n * m)
                    i2, j2 = idx2 // m, idx2 % m
                    old2 = U[idx2, r]
                    candidates2 = [c for c in valid_vals if abs(c - old2) <= 1.0]
                elif mat2 == 1:
                    idx2 = np.random.randint(m * p)
                    j2, k2 = idx2 // p, idx2 % p
                    old2 = V[idx2, r]
                    candidates2 = [c for c in valid_vals if abs(c - old2) <= 1.0]
                else:
                    idx2 = np.random.randint(p * n)
                    k2, i2 = idx2 // n, idx2 % n
                    old2 = W[idx2, r]
                    candidates2 = [c for c in valid_vals if abs(c - old2) <= 1.0]
                
                # Try a few candidate pairs
                best_err = current_err
                best_c1, best_c2 = old1, old2
                best_idx1, best_idx2 = idx1, idx2
                
                for c1 in candidates1[:5]:
                    for c2 in candidates2[:5]:
                        if c1 == old1 and c2 == old2:
                            continue
                        
                        # Compute new error
                        Rec_new = Rec.copy()
                        if mat1 == 0:
                            delta1 = c1 - old1
                            V_mat = V[:, r].reshape(m, p)
                            W_mat = W[:, r].reshape(p, n)
                            Rec_new[i1, :, :] += delta1 * (V_mat * W_mat[:, i1])
                        elif mat1 == 1:
                            delta1 = c1 - old1
                            U_mat = U[:, r].reshape(n, m)
                            W_mat = W[:, r].reshape(p, n)
                            Rec_new[:, j1, k1] += delta1 * (U_mat[:, j1] * W_mat[k1, :])
                        else:
                            delta1 = c1 - old1
                            U_mat = U[:, r].reshape(n, m)
                            V_mat = V[:, r].reshape(m, p)
                            Rec_new[i1, :, k1] += delta1 * (U_mat[i1, :] * V_mat[:, k1])
                        
                        if mat2 == 0:
                            delta2 = c2 - old2
                            V_mat = V[:, r].reshape(m, p)
                            W_mat = W[:, r].reshape(p, n)
                            Rec_new[i2, :, :] += delta2 * (V_mat * W_mat[:, i2])
                        elif mat2 == 1:
                            delta2 = c2 - old2
                            U_mat = U[:, r].reshape(n, m)
                            W_mat = W[:, r].reshape(p, n)
                            Rec_new[:, j2, k2] += delta2 * (U_mat[:, j2] * W_mat[k2, :])
                        else:
                            delta2 = c2 - old2
                            U_mat = U[:, r].reshape(n, m)
                            V_mat = V[:, r].reshape(m, p)
                            Rec_new[i2, :, k2] += delta2 * (U_mat[i2, :] * V_mat[:, k2])
                        
                        new_err = np.max(np.abs(Rec_new - 1.0))
                        if new_err < best_err - 1e-10:
                            best_err = new_err
                            best_c1, best_c2 = c1, c2
                            best_idx1, best_idx2 = idx1, idx2
                            best_rec = Rec_new
                
                if best_err < current_err - 1e-10:
                    if mat1 == 0:
                        U[best_idx1, r] = best_c1
                    elif mat1 == 1:
                        V[best_idx1, r] = best_c1
                    else:
                        W[best_idx1, r] = best_c1
                    if mat2 == 0:
                        U[best_idx2, r] = best_c2
                    elif mat2 == 1:
                        V[best_idx2, r] = best_c2
                    else:
                        W[best_idx2, r] = best_c2
                    Rec = best_rec
                    current_err = best_err
                    improved = True
            else:
                # Single entry move (original approach)
                mat_type = move_type  # 0, 1, or 2
                r = np.random.randint(rank)
                
                if mat_type == 0:
                    idx = np.random.randint(n * m)
                    i = idx // m
                    j = idx % m
                    old_val = U[idx, r]
                    candidates = [c for c in valid_vals if abs(c - old_val) <= 1.5]
                    
                    best_val = old_val
                    best_err = current_err
                    best_rec = Rec
                    
                    for cand in candidates[:6]:
                        if cand == old_val:
                            continue
                        delta = cand - old_val
                        V_mat = V[:, r].reshape(m, p)
                        W_mat = W[:, r].reshape(p, n)
                        Rec_new = Rec.copy()
                        Rec_new[i, :, :] += delta * (V_mat * W_mat[:, i])
                        new_err = np.max(np.abs(Rec_new - 1.0))
                        if new_err < best_err - 1e-10:
                            best_err = new_err
                            best_val = cand
                            best_rec = Rec_new
                    
                    if best_val != old_val:
                        U[idx, r] = best_val
                        Rec = best_rec
                        current_err = best_err
                        improved = True
                elif mat_type == 1:
                    idx = np.random.randint(m * p)
                    j = idx // p
                    k = idx % p
                    old_val = V[idx, r]
                    candidates = [c for c in valid_vals if abs(c - old_val) <= 1.5]
                    
                    best_val = old_val
                    best_err = current_err
                    best_rec = Rec
                    
                    for cand in candidates[:6]:
                        if cand == old_val:
                            continue
                        delta = cand - old_val
                        U_mat = U[:, r].reshape(n, m)
                        W_mat = W[:, r].reshape(p, n)
                        Rec_new = Rec.copy()
                        Rec_new[:, j, k] += delta * (U_mat[:, j] * W_mat[k, :])
                        new_err = np.max(np.abs(Rec_new - 1.0))
                        if new_err < best_err - 1e-10:
                            best_err = new_err
                            best_val = cand
                            best_rec = Rec_new
                    
                    if best_val != old_val:
                        V[idx, r] = best_val
                        Rec = best_rec
                        current_err = best_err
                        improved = True
                else:
                    idx = np.random.randint(p * n)
                    k = idx // n
                    i = idx % n
                    old_val = W[idx, r]
                    candidates = [c for c in valid_vals if abs(c - old_val) <= 1.5]
                    
                    best_val = old_val
                    best_err = current_err
                    best_rec = Rec
                    
                    for cand in candidates[:6]:
                        if cand == old_val:
                            continue
                        delta = cand - old_val
                        U_mat = U[:, r].reshape(n, m)
                        V_mat = V[:, r].reshape(m, p)
                        Rec_new = Rec.copy()
                        Rec_new[i, :, k] += delta * (U_mat[i, :] * V_mat[:, k])
                        new_err = np.max(np.abs(Rec_new - 1.0))
                        if new_err < best_err - 1e-10:
                            best_err = new_err
                            best_val = cand
                            best_rec = Rec_new
                    
                    if best_val != old_val:
                        W[idx, r] = best_val
                        Rec = best_rec
                        current_err = best_err
                        improved = True
            
            if not improved and iteration % 100 == 0:
                # Random restart perturbation
                if np.random.rand() < 0.3:
                    mat_type = np.random.randint(3)
                    r = np.random.randint(rank)
                    if mat_type == 0:
                        idx = np.random.randint(n * m)
                        U[idx, r] = np.random.choice(valid_vals)
                    elif mat_type == 1:
                        idx = np.random.randint(m * p)
                        V[idx, r] = np.random.choice(valid_vals)
                    else:
                        idx = np.random.randint(p * n)
                        W[idx, r] = np.random.choice(valid_vals)
                    Rec = compute_reconstruction(U, V, W, rank)
                    current_err = np.max(np.abs(Rec - 1.0))

        if current_err < atol:
            return U, V, W, True
        return U, V, W, False

    def structured_init(rank):
        """Create a structured initial guess based on algebraic patterns."""
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Try to use a cyclic pattern
        for r in range(rank):
            # Assign values based on a pattern
            for i in range(n):
                for j in range(m):
                    U[i * m + j, r] = np.random.choice([-1, 0, 1])
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = np.random.choice([-1, 0, 1])
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r] = np.random.choice([-1, 0, 1])
        
        return U, V, W

    def search_for_solution(max_rank, max_search_time=140.0):
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
                
                np.random.seed(42 + rank * 1000 + attempt * 7)
                
                if attempt % 3 == 0:
                    U, V, W = structured_init(rank)
                else:
                    U = np.random.choice([-1, -0.5, 0, 0.5, 1], (n * m, rank))
                    V = np.random.choice([-1, -0.5, 0, 0.5, 1], (m * p, rank))
                    W = np.random.choice([-1, -0.5, 0, 0.5, 1], (p * n, rank))
                
                if is_valid(U, V, W, rank):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = rank
                    return True
                
                U_opt, V_opt, W_opt, success = local_optimize(U, V, W, rank, max_iters=800)
                if success:
                    best_solution = {"U": U_opt, "V": V_opt, "W": W_opt}
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
