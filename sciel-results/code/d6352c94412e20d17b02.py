import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Return a rank-minimal (or near-minimal) exact tensor decomposition for
    multiplying a 3x4 matrix by a 4x6 matrix.
    
    The decomposition satisfies:
        A * B = sum_{t=0}^{rank-1} W[:,t] * outer(U[:,t], V[:,t])
    where:
        U is (n*m, rank) = (12, rank)
        V is (m*p, rank) = (24, rank)
        W is (p*n, rank) = (18, rank)
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n * m, m * p, p * n  # 12, 24, 18
    
    start_time = time.time()
    deadline = 130.0  # Leave 20s margin before 150s limit
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        T_rec = np.einsum('ijt,jkt,kit->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def try_lattice_points(U, V, W, rank):
        """Try rounding to various nearby lattice points (integers and half-integers)"""
        # Try standard rounding
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        if check_exact(U_s, V_s, W_s, rank):
            return U_s, V_s, W_s
        
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if check_exact(U_i, V_i, W_i, rank):
            return U_i, V_i, W_i
        
        # Try floor and ceil variants
        for fu in [np.floor, np.ceil]:
            for fv in [np.floor, np.ceil]:
                for fw in [np.floor, np.ceil]:
                    U_t = fu(U * 2) / 2.0
                    V_t = fv(V * 2) / 2.0
                    W_t = fw(W * 2) / 2.0
                    if check_exact(U_t, V_t, W_t, rank):
                        return U_t, V_t, W_t
        
        # Try mixed rounding: round some to int, some to half-int
        for shift_u in [0, 0.5]:
            for shift_v in [0, 0.5]:
                for shift_w in [0, 0.5]:
                    U_t = np.round((U - shift_u) * 2) / 2.0 + shift_u
                    V_t = np.round((V - shift_v) * 2) / 2.0 + shift_v
                    W_t = np.round((W - shift_w) * 2) / 2.0 + shift_w
                    if check_exact(U_t, V_t, W_t, rank):
                        return U_t, V_t, W_t
        
        return None

    def build_schoolbook(rank=None):
        """Build the schoolbook decomposition"""
        if rank is None:
            rank = rank_schoolbook
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if column < rank:
                        U[i * m + j, column] = 1
                        V[j * p + k, column] = 1
                        W[k * n + i, column] = 1
                    column += 1
        
        return U, V, W, rank

    def build_structured_decomposition(rank):
        """
        Build a structured decomposition using algebraic patterns.
        For 3x4x6, we can exploit symmetries to reduce rank.
        """
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        # Pattern 1: Group by i (rows of A)
        # For each i, we need to compute row i of A times B
        # This can be done with m*p = 24 terms, but we can share
        # across i using linearity
        
        # Try a structured approach:
        # Use the fact that T[i,j,k] = 1 for all i,j,k
        # We can write 1 = sum of products that telescope
        
        # Simple structured pattern: use rank = n*m*p but with sharing
        # Actually, let's try a different approach:
        # For 3x4x6, known bounds suggest rank <= 36 or so
        
        # Try: for each pair (i,k), we need sum_j A[i,j]*B[j,k]
        # We can write this using rank terms where each term
        # contributes to multiple (i,k) pairs
        
        # Pattern: use Kronecker-like structure
        # Term t corresponds to (a, b) where a in range(n), b in range(m*p)
        # But that's just schoolbook again
        
        # Let's try: for each i, use m terms that sum over j with B contributions
        # U[i*m+j, t] = 1 for t = i*m + j (schoolbook for A)
        # But share B terms across i
        
        # Alternative: exploit that T is all ones
        # T[i,j,k] = 1 = 1*1*1
        # We can use rank = n + m + p - 2 via a known identity?
        # No, that's for specific tensors
        
        # For now, return schoolbook if rank == rank_schoolbook
        if rank == rank_schoolbook:
            return build_schoolbook(rank)
        
        # Try a heuristic structured fill
        np.random.seed(42)
        
        # Initialize with zeros
        U = np.zeros((NM, rank))
        V = np.zeros((MP, rank))
        W = np.zeros((PN, rank))
        
        # Fill with structured random values from {-1, -0.5, 0, 0.5, 1}
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        for t in range(rank):
            # Sparse: only a few non-zero entries per column
            nz_u = np.random.choice(NM, size=min(3, NM), replace=False)
            nz_v = np.random.choice(MP, size=min(5, MP), replace=False)
            nz_w = np.random.choice(PN, size=min(3, PN), replace=False)
            
            for idx in nz_u:
                U[idx, t] = values[np.random.randint(len(values))]
            for idx in nz_v:
                V[idx, t] = values[np.random.randint(len(values))]
            for idx in nz_w:
                W[idx, t] = values[np.random.randint(len(values))]
        
        return U, V, W, rank

    def als_decompose(rank, max_iters=300, seed=0, init=None):
        """Alternating Least Squares for 3rd order tensor decomposition with better convergence"""
        np.random.seed(seed)
        
        if init is not None:
            U, V, W = init
        else:
            # Structured initialization: start from schoolbook-like structure
            U = np.zeros((n, m, rank))
            V = np.zeros((m, p, rank))
            W = np.zeros((p, n, rank))
            
            # Initialize with small random values plus structured components
            for t in range(rank):
                # Deterministic pattern based on t
                i = t % n
                j = (t // n) % m
                k = (t // (n*m)) % p
                U[i, j, t] = 1.0
                V[j, k, t] = 1.0
                W[k, i, t] = 1.0
            
            # Add small random perturbation to break symmetry
            U += np.random.randn(n, m, rank) * 0.01
            V += np.random.randn(m, p, rank) * 0.01
            W += np.random.randn(p, n, rank) * 0.01
        
        prev_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = np.inf
        
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if current_err < best_err:
                best_err = current_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if abs(prev_err - current_err) < 1e-16:
                break
            if current_err < 1e-12:
                break
            prev_err = current_err
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    M = V[j, :, :] * W[:, i, :]  # shape (p, rank)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, j, :], rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    M = U[:, j, :] * W[k, :, :]  # shape (n, rank)
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(M.T, T[:, j, k], rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    M = U[i, :, :] * V[:, k, :]  # shape (m, rank)
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(M.T, T[i, :, k], rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Use the best solution found
        U, V, W = best_U, best_V, best_W
        
        # Final check
        T_rec = np.einsum('ijt,jkt,kit->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-6:
            snapped = try_lattice_points(U.reshape(NM, rank), V.reshape(MP, rank), W.reshape(PN, rank), rank)
            if snapped:
                return snapped
            
            # Try local polish with scipy
            if final_err < 1.0:
                try:
                    from scipy.optimize import minimize
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijt,jkt,kit->ijk', U_x, V_x, W_x)
                        return np.sum((T_rec - T)**2)
                    
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-18})
                    
                    if res.fun < 1e-10:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        snapped = try_lattice_points(U_opt.reshape(NM, rank), V_opt.reshape(MP, rank), W_opt.reshape(PN, rank), rank)
                        if snapped:
                            return snapped
                except Exception:
                    pass
        
        return None

    def direct_search(rank, num_trials=100):
        """Direct random search with structured perturbations of schoolbook"""
        base_U, base_V, base_W, _ = build_schoolbook(rank)
        
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Create a perturbation: modify some columns
            U_try = base_U.copy()
            V_try = base_V.copy()
            W_try = base_W.copy()
            
            # Randomly modify a few columns
            num_mods = min(5, rank)
            cols = np.random.choice(rank, num_mods, replace=False)
            for c in cols:
                # Set to random small values
                U_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=NM)
                V_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=MP)
                W_try[:, c] = np.random.choice([-1, 0, 1, 0.5, -0.5], size=PN)
            
            if check_exact(U_try, V_try, W_try, rank):
                return U_try, V_try, W_try
        
        return None

    def try_rank(rank, num_restarts=5):
        """Try to find decomposition of given rank"""
        # First try direct search (fast)
        result = direct_search(rank, num_trials=30)
        if result is not None:
            return result
        
        # Try structured initialization
        U_s, V_s, W_s, _ = build_structured_decomposition(rank)
        if check_exact(U_s, V_s, W_s, rank):
            return U_s, V_s, W_s
        
        # Then try ALS with multiple seeds
        for seed in range(num_restarts):
            if time.time() > deadline - 10:
                return None
            result = als_decompose(rank, max_iters=150, seed=seed)
            if result is not None:
                return result
        
        return None

    best_result = None
    best_rank = rank_schoolbook
    
    # Try a focused set of ranks with more effort on each
    # Based on known results, try ranks that are likely achievable
    candidate_ranks = [71, 70, 68, 66, 64, 60, 56, 52, 48, 44, 40, 36, 32, 28]
    
    for rank in candidate_ranks:
        if rank >= rank_schoolbook:
            continue
        if time.time() > deadline - 20:
            break
        
        result = try_rank(rank, num_restarts=3)
        if result is not None:
            U, V, W = result
            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            break
    
    # Fallback to schoolbook
    if best_result is None:
        U, V, W, rank = build_schoolbook()
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
