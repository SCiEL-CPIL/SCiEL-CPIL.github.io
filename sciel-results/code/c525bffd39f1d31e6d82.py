"""
Strategy: Randomized Local Search (Simulated Annealing) on the tensor decomposition error surface,
followed by exact rational reconstruction.

The current program attempts to construct a decomposition based on a specific algebraic pattern 
(induction argument). This new approach treats the entries of U, V, W as continuous variables 
and minimizes the sum of squared errors (MSE) between the reconstructed tensor and the target 
tensor. Once a low error is found, we use Fraction reconstruction to snap to exact rational values.
"""
import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = rank_schoolbook
    
    # Target tensor T[i,j,k] = 1 for all i,j,k
    # We want to find U, V, W of rank R such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    def calculate_error(U, V, W):
        # Efficiently calculate sum of squared errors
        # T is all 1s.
        # Reconstructed R[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Since dimensions are small, we can just loop or use einsum.
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # R[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a 3-way contraction.
        # We can compute it by iterating over c, or using einsum.
        # Let's use einsum for speed if possible, or simple loops since R is small.
        # n=2, m=2, p=15.
        # Total elements in output = 60.
        # R is likely < 30.
        
        # einsum: 'ijc,jkc,kic->ijk'
        try:
            R = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            error = np.sum((R - 1.0) ** 2)
        except:
            # Fallback
            R = np.zeros((n, m, p))
            for c in range(U.shape[1]):
                uc = U_reshaped[:, :, c]
                vc = V_reshaped[:, :, c]
                wc = W_reshaped[:, :, c]
                # uc: (n,m), vc: (m,p), wc: (p,n)
                # R[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            R[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            error = np.sum((R - 1.0) ** 2)
        return error

    def snap_to_rational(U, V, W, max_denom=10):
        """
        Try to snap floating point values to fractions with small denominators.
        Returns None if error is not small enough after snapping.
        """
        U_frac = np.zeros_like(U)
        V_frac = np.zeros_like(V)
        W_frac = np.zeros_like(W)
        
        def snap_val(x):
            if abs(x) < 1e-5:
                return 0.0
            # Try small integers and halves
            # We can just round to nearest 0.5 or 1.0, or check a few fractions
            # Given the problem likely has integer/half-integer solutions, 
            # rounding to nearest 0.5 is a good heuristic.
            # x = p/q. If q=1, x is int. If q=2, x is half-int.
            # Let's try rounding to nearest 0.5.
            snapped = round(x * 2) / 2
            return snapped
            
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_frac[i,j] = snap_val(U[i,j])
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_frac[i,j] = snap_val(V[i,j])
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_frac[i,j] = snap_val(W[i,j])
                
        # Verify exactness with snapped values
        error = calculate_error(U_frac, V_frac, W_frac)
        if error < 1e-9:
            return U_frac, V_frac, W_frac
        return None

    # Search for lower ranks
    # Theoretical lower bound is n*m*p / (something)? 
    # For 2x2x2, rank is 7.
    # For 2x2xp, rank is roughly 2p+2? 
    # Let's try ranks from 18 up to 30.
    
    current_time = time.time()
    # We have ~130s. 
    # Each error calculation is fast (einsum on small arrays).
    # Let's do Simulated Annealing for a fixed number of steps per rank.
    
    for rank in range(18, 35):
        if time.time() > deadline - 5:
            break
            
        # Initialize U, V, W with random values
        # Scale: if entries are ~1, then sum of R terms should be 1.
        # If rank is R, each term contributes ~1/R.
        # So entries should be ~ sqrt(1/R) or similar.
        # Let's start with random small values.
        
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        current_error = calculate_error(U, V, W)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_error = current_error
        
        # Simulated Annealing
        T = 1.0
        T_min = 1e-6
        cooling_rate = 0.995
        steps = 2000  # Adjust based on time budget
        
        # Time budget per rank
        time_budget_per_rank = 10.0
        start_rank_time = time.time()
        
        step_count = 0
        while step_count < steps and (time.time() - start_rank_time) < time_budget_per_rank:
            step_count += 1
            T *= cooling_rate
            if T < T_min:
                break
                
            # Pick a random matrix and element to perturb
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                mat = U
                rows, cols = n * m, rank
            elif mat_idx == 1:
                mat = V
                rows, cols = m * p, rank
            else:
                mat = W
                rows, cols = p * n, rank
                
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, cols - 1)
            
            old_val = mat[r_idx, c_idx]
            # Gaussian perturbation
            new_val = old_val + random.gauss(0, 0.1)
            mat[r_idx, c_idx] = new_val
            
            new_error = calculate_error(U, V, W)
            
            if new_error < best_error:
                best_error = new_error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                
            # Accept if better or with probability exp(-(delta)/T)
            delta = new_error - current_error
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_error = new_error
            else:
                mat[r_idx, c_idx] = old_val
                
        # Check if best solution for this rank is valid
        if best_error < 1e-5:
            snapped = snap_to_rational(best_U, best_V, best_W)
            if snapped:
                U_s, V_s, W_s = snapped
                # Final verification
                if calculate_error(U_s, V_s, W_s) < 1e-9:
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    
    return best_result
