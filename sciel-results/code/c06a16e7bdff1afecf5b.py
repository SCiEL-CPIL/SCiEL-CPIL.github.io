import numpy as np
from fractions import Fraction
import time
import random
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 7
    
    # Precompute target tensor indices
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a in 0..5, b in 0..20, c in 0..13
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.append((a, b, c))
    
    # Create a dense target tensor for fast numpy operations
    # Size: 6 x 21 x 14 = 1764 elements. Very small.
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for a, b, c in valid_indices:
        T_target[a, b, c] = 1.0

    def verify_float(U, V, W, rank, tol=1e-5):
        """
        Fast check using NumPy.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        """
        # T_reconstructed[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # We can compute this efficiently.
        # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        # This creates a 6x21x14 array.
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check valid indices
        for a, b, c in valid_indices:
            if abs(T_recon[a, b, c] - 1.0) > tol:
                return False
        
        # Check invalid indices
        # We can create a mask
        mask = np.zeros_like(T_recon, dtype=bool)
        for a, b, c in valid_indices:
            mask[a, b, c] = True
        
        # If any invalid entry is significantly non-zero, fail
        if np.any(np.abs(T_recon[~mask]) > tol):
            return False
            
        return True

    def verify_exact(U, V, W, rank):
        """
        Exact check using Fractions.
        """
        Uf = [[Fraction(U[i][r]) for r in range(rank)] for i in range(n * m)]
        Vf = [[Fraction(V[i][r]) for r in range(rank)] for i in range(m * p)]
        Wf = [[Fraction(W[i][r]) for r in range(rank)] for i in range(p * n)]
        
        # Check valid
        for a, b, c in valid_indices:
            s = Fraction(0)
            for r in range(rank):
                s += Uf[a][r] * Vf[b][r] * Wf[c][r]
            if s != 1:
                return False
        
        # Check invalid
        # To speed this up, we can skip if we are confident, but for correctness we must check.
        # However, 1764 * rank operations with Fractions is slow if rank is large.
        # We only call this if float check passed, so it should be rare.
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) in valid_indices:
                        continue
                    s = Fraction(0)
                    for r in range(rank):
                        s += Uf[a][r] * Vf[b][r] * Wf[c][r]
                    if s != 0:
                        return False
        return True

    def snap_to_half_integers(arr, tol=0.1):
        """
        Snap values to nearest 0.5 if close enough.
        Returns True if all values were close enough.
        """
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest = round(val * 2) / 2.0
                if abs(val - nearest) < tol:
                    snapped[i, j] = nearest
                else:
                    return None # Not close enough
        return snapped

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    start_time = time.time()
    deadline = start_time + 145.0
    
    best_result = schoolbook()
    
    # Optimization Strategy
    # We want to find U, V, W with rank r < 42.
    # Objective: min || T(U,V,W) - T_target ||_F^2 + lambda * || snap(U,V,W) - (U,V,W) ||^2
    # Actually, just minimize data fidelity and then snap.
    
    def objective(params, rank):
        """
        params: flattened array of U, V, W
        """
        size_u = n * m * rank
        size_v = m * p * rank
        size_w = p * n * rank
        
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:size_u + size_v + size_w]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Compute T_recon
        # T_recon[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
        # Using einsum
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Data fidelity
        error = np.sum((T_recon - T_target) ** 2)
        
        # Regularization: encourage values to be close to 0, 0.5, 1, -0.5, -1
        # Penalty for not being half-integer
        # dist_to_half = min_{k in Z} (x - k/2)^2
        # We can approximate this by summing (x - round(2x)/2)^2
        # This is non-smooth, but we are using Powell/Nelder-Mead which can handle it roughly,
        # or we can just rely on the data fidelity and snap at the end.
        # Let's add a soft penalty to keep values bounded and near half-integers.
        # penalty = sum((U - snap(U))**2) ...
        # For simplicity in the objective, we just minimize error.
        # The snapping is done post-optimization.
        
        return error

    # Try ranks 41 down to 35
    # Use multiple random restarts for each rank
    for rank in range(41, 34, -1):
        if time.time() > deadline - 10.0:
            break
            
        found = False
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize with small random values
            # Scale: entries should be around 1.
            # If we start with 0, it's a saddle point.
            # Start with random values in [-1, 1]
            np.random.seed(restart)
            
            size_u = n * m * rank
            size_v = m * p * rank
            size_w = p * n * rank
            total_size = size_u + size_v + size_w
            
            params0 = np.random.uniform(-1, 1, total_size)
            
            # Use Powell method, which is derivative-free and good for this
            # Max iterations per restart
            max_iter = 200
            
            try:
                res = minimize(
                    objective, 
                    params0, 
                    args=(rank,), 
                    method='Powell', 
                    options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                
                if res.success or res.fun < 1e-5:
                    # Extract U, V, W
                    U_flat = res.x[:size_u]
                    V_flat = res.x[size_u:size_u + size_v]
                    W_flat = res.x[size_u + size_v:]
                    
                    U = U_flat.reshape(n * m, rank)
                    V = V_flat.reshape(m * p, rank)
                    W = W_flat.reshape(p * n, rank)
                    
                    # Snap to half-integers
                    U_snap = snap_to_half_integers(U, tol=0.2)
                    V_snap = snap_to_half_integers(V, tol=0.2)
                    W_snap = snap_to_half_integers(W, tol=0.2)
                    
                    if U_snap is not None and V_snap is not None and W_snap is not None:
                        # Convert to list of lists for verify
                        U_list = U_snap.tolist()
                        V_list = V_snap.tolist()
                        W_list = W_snap.tolist()
                        
                        # Float check first
                        if verify_float(U_list, V_list, W_list, rank):
                            # Exact check
                            if verify_exact(U_list, V_list, W_list, rank):
                                best_result = {"U": U_list, "V": V_list, "W": W_list}
                                found = True
                                break
            except Exception as e:
                pass
        
        if found:
            # Found a lower rank, continue to try even lower?
            # Usually, if we find 41, 40 might be harder.
            # But let's try.
            continue
        else:
            # If we failed to find rank 41, it's likely 40 is also hard.
            # However, sometimes 40 is easier due to different structure? Unlikely.
            # Break to save time for fallback or lower ranks if time permits.
            # Given the difficulty, breaking is safer.
            break

    return best_result
