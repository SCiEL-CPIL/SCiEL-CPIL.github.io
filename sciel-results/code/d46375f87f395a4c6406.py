import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    total_size = nm + mp + pn
    
    def compute_residual_tensor(U, V, W, R):
        """
        U: shape (nm, R)
        V: shape (mp, R)
        W: shape (pn, R)
        Returns: (n, m, p) tensor of residuals
        """
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We want T[i,j,k] == 1 for all i,j,k
        T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        return T - 1.0

    def residual_function(params, R):
        """
        params: flat vector of length R * (nm + mp + pn)
        Returns: flat vector of residuals
        """
        size_U = nm * R
        size_V = mp * R
        size_W = pn * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(nm, R)
        V = V_flat.reshape(mp, R)
        W = W_flat.reshape(pn, R)
        
        res = compute_residual_tensor(U, V, W, R)
        return res.flatten()

    def snap_to_half_integer(arr):
        """Snap array to nearest half-integers (multiples of 0.5)."""
        return np.round(arr * 2) / 2.0

    def verify_exact(U, V, W, R):
        """
        Verify if U, V, W (assumed to be half-integers) exactly decompose the tensor.
        Returns True if valid.
        """
        # Scale by 2 to work with integers for exactness check
        U_int = (U * 2).astype(np.int64)
        V_int = (V * 2).astype(np.int64)
        W_int = (W * 2).astype(np.int64)
        
        U3 = U_int.reshape(n, m, R)
        V3 = V_int.reshape(m, p, R)
        W3 = W_int.reshape(p, n, R)
        
        # Product in integer domain: sum (2U)(2V)(2W) = 4 * (sum UVW)
        # We want sum UVW = 1, so sum (2U)(2V)(2W) should be 4
        T_int = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3, optimize=True)
        
        # Check if all entries are exactly 4
        return np.all(T_int == 4)

    def lattice_local_search(U, V, W, R, max_iters=100, start_temp=1.0, end_temp=0.01, timeout=None, start_time=0):
        """
        Perform simulated annealing on the half-integer lattice.
        U, V, W are arrays of half-integers.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Precompute indices for faster access
        idx_U = np.arange(nm * R)
        idx_V = np.arange(nm * R, nm * R + mp * R)
        idx_W = np.arange(nm * R + mp * R, R * (nm + mp + pn))
        all_indices = np.concatenate([idx_U, idx_V, idx_W])
        
        def get_value(idx):
            if idx < nm * R:
                return U[idx]
            elif idx < nm * R + mp * R:
                return V[idx - nm * R]
            else:
                return W[idx - nm * R - mp * R]
        
        def set_value(idx, val):
            if idx < nm * R:
                U[idx] = val
            elif idx < nm * R + mp * R:
                V[idx - nm * R] = val
            else:
                W[idx - nm * R - mp * R] = val

        def get_res_sq():
            res = compute_residual_tensor(U, V, W, R)
            return np.sum(res**2)

        current_energy = get_res_sq()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_energy = current_energy
        
        n_vars = len(all_indices)
        
        for it in range(max_iters):
            if timeout and time.time() - start_time > timeout:
                break
                
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Pick a random variable
            idx = np.random.choice(n_vars)
            
            orig_val = get_value(idx)
            # Try +0.5 or -0.5
            delta = 0.5 if np.random.rand() < 0.5 else -0.5
            new_val = orig_val + delta
            
            # Apply change
            set_value(idx, new_val)
            
            new_energy = get_res_sq()
            
            # Simulated annealing acceptance
            if new_energy < current_energy or np.random.rand() < np.exp(-(new_energy - current_energy) / max(temp, 1e-9)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    # If we found an exact solution, we can stop
                    if best_energy < 1e-10:
                        return best_U, best_V, best_W, True
            else:
                # Revert
                set_value(idx, orig_val)
                
        # Final check
        if best_energy < 1e-10:
            return best_U, best_V, best_W, True
            
        return U, V, W, False

    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for return overhead
    
    # Known lower bound for 2x3x5 matmul is likely around 18-20.
    # Schoolbook is 30.
    # We try ranks from 18 upwards.
    for R in range(18, 31):
        if time.time() > deadline:
            break
            
        # Number of attempts depends on remaining time
        remaining_time = deadline - time.time()
        # Estimate cost per attempt: least_squares(500 iters) ~ 0.1s, local_search ~ 0.5s
        # Allow ~ 10 attempts per 10 seconds?
        n_attempts = max(5, int(remaining_time * 0.5))
        
        for attempt in range(n_attempts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Use structured random init to help convergence
            x0 = np.random.normal(0, 1.0, R * total_size)
            
            try:
                # Use least_squares to find a nearby continuous solution
                # 'lm' (Levenberg-Marquardt) is good for small-to-medium nonlinear least squares
                result = least_squares(residual_function, x0, args=(R,), method='lm', max_nfev=300)
                
                if result.cost < 1e-2:  # Threshold to consider for snapping
                    U_flat = result.x[:nm*R]
                    V_flat = result.x[nm*R:nm*R+mp*R]
                    W_flat = result.x[nm*R+mp*R:]
                    
                    U0 = U_flat.reshape(nm, R)
                    V0 = V_flat.reshape(mp, R)
                    W0 = W_flat.reshape(pn, R)
                    
                    # Snap to half-integers
                    U_s = snap_to_half_integer(U0)
                    V_s = snap_to_half_integer(V0)
                    W_s = snap_to_half_integer(W0)
                    
                    # Check if snapped is already valid
                    if verify_exact(U_s, V_s, W_s, R):
                        return {
                            "U": U_s.tolist(),
                            "V": V_s.tolist(),
                            "W": W_s.tolist()
                        }
                        
                    # Run local search on the lattice
                    if time.time() < deadline - 5: # Need time for local search
                        U_ls, V_ls, W_ls, success = lattice_local_search(
                            U_s, V_s, W_s, R, 
                            max_iters=200, 
                            timeout=deadline, 
                            start_time=start_time
                        )
                        
                        if success or verify_exact(U_ls, V_ls, W_ls, R):
                            return {
                                "U": U_ls.tolist(),
                                "V": V_ls.tolist(),
                                "W": W_ls.tolist()
                            }
                            
            except Exception as e:
                # Catch any errors in optimization
                continue
                
    # Fallback: Schoolbook decomposition (Rank = n*m*p = 30)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, r] = 1.0
                V[j*p+k, r] = 1.0
                W[k*n+i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
