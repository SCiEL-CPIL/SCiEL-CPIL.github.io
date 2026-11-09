import numpy as np

class ExercisePolicy:
    def __init__(self, coeffs, K, r, dt, degree=4):
        self.coeffs = coeffs
        self.K = K
        self.r = r
        self.dt = dt
        self.df = np.exp(-r * dt)
        self.degree = degree
        
    def exercise(self, current_state, time):
        if isinstance(time, (int, np.integer)):
            k = int(time)
        else:
            k = int(round(time * 12))
            
        k = max(1, min(12, k))
        
        P = np.maximum(self.K - np.asarray(current_state), 0.0)
        
        if k == 12:
            return P > 0.0
        
        idx = k - 1
        c = self.coeffs[idx]
        
        S_arr = np.asarray(current_state)
        # Use polynomial evaluation with proper degree handling
        if self.degree == 3:
            C = np.polyval(c, S_arr) * self.df
        else:
            # For higher degrees, we need to construct the design matrix
            # coeffs are stored as [c0, c1, ..., cd] where value = sum(ci * S^i)
            # But np.polyval expects descending order, so let's just use polyval
            C = np.polyval(c, S_arr) * self.df
        C = np.maximum(C, 0.0)
        
        return P >= C

def fit_exercise_policy(training_paths, contract):
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    N, D = paths.shape
    
    if D == 13:
        S = paths[:, 1:13]
    elif D == 12:
        S = paths[:, 0:12]
    else:
        S = paths[:, -12:]
        
    P = np.maximum(K - S, 0.0)
    
    df = np.exp(-r * dt)
    
    V_next = P[:, 11].copy()
    
    coeffs = []
    
    # Use degree 4 for better approximation
    degree = 4
    
    for k in range(11, 0, -1):
        idx_curr = k - 1
        
        S_curr = S[:, idx_curr]
        
        mask = np.isfinite(S_curr) & np.isfinite(V_next)
        if np.sum(mask) < 10:
            c = [np.mean(V_next)]
        else:
            # Use weighted least squares with more features
            # Include S, S^2, S^3, S^4 terms
            S_masked = S_curr[mask]
            V_masked = V_next[mask]
            
            # Construct design matrix
            X = np.column_stack([
                np.ones_like(S_masked),
                S_masked,
                S_masked**2,
                S_masked**3,
                S_masked**4
            ])
            
            # Solve least squares
            try:
                c, residuals, rank, sv = np.linalg.lstsq(X, V_masked, rcond=None)
            except np.linalg.LinAlgError:
                c = np.polyfit(S_masked, V_masked, degree)
            
            # Convert to polyval format (descending order)
            # c is [c0, c1, c2, c3, c4] for value = c0 + c1*S + c2*S^2 + c3*S^3 + c4*S^4
            # polyval expects [c4, c3, c2, c1, c0]
            c_polyval = c[::-1]
            
        coeffs.append(c_polyval)
        
        S_curr_for_eval = S[:, idx_curr]
        C_curr = np.polyval(c_polyval, S_curr_for_eval) * df
        C_curr = np.maximum(C_curr, 0.0)
        
        P_curr = P[:, idx_curr]
        V_curr = np.maximum(P_curr, C_curr)
        
        V_next = V_curr
    
    coeffs = coeffs[::-1]
    
    return ExercisePolicy(coeffs, K, r, dt, degree=degree)
