import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimizes a 10x10 probability distribution to maximize the ratio
    D / M, where D is the entropy of x-y and M is the max entropy
    of x, x+y, and x+2y.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    # Grid coordinates
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Precompute projection indices
    # P[y, x] corresponds to row y, col x
    # We need to map (x, y) to indices for x-y, x, x+y, x+2y
    # x-y ranges from -9 to 9. Shift by 9 to get 0..18
    idx_d = (X - Y) + 9  # Shape (10, 10)
    idx_0 = X            # Shape (10, 10), 0..9
    idx_1 = X + Y        # Shape (10, 10), 0..18
    idx_2 = X + 2*Y      # Shape (10, 10), 0..27

    max_d = 19
    max_0 = 10
    max_1 = 19
    max_2 = 28

    def compute_entropies(theta):
        """
        theta: flat array of shape (100,)
        Returns: D, H0, H1, H2
        """
        # Softmax to get probabilities
        # Subtract max for numerical stability
        theta_max = np.max(theta)
        exp_theta = np.exp(theta - theta_max)
        P = exp_theta / np.sum(exp_theta)
        
        # Reshape to 10x10
        P_grid = P.reshape((n, n))
        
        # Calculate marginals
        # D: distribution of x-y
        prob_d = np.zeros(max_d)
        np.add.at(prob_d, idx_d.ravel(), P_grid.ravel())
        
        # H0: distribution of x
        prob_0 = np.zeros(max_0)
        np.add.at(prob_0, idx_0.ravel(), P_grid.ravel())
        
        # H1: distribution of x+y
        prob_1 = np.zeros(max_1)
        np.add.at(prob_1, idx_1.ravel(), P_grid.ravel())
        
        # H2: distribution of x+2y
        prob_2 = np.zeros(max_2)
        np.add.at(prob_2, idx_2.ravel(), P_grid.ravel())
        
        def shannon_entropy(prob):
            prob = prob[prob > 1e-15]
            if len(prob) == 0:
                return 0.0
            return -np.sum(prob * np.log2(prob))
        
        D = shannon_entropy(prob_d)
        H0 = shannon_entropy(prob_0)
        H1 = shannon_entropy(prob_1)
        H2 = shannon_entropy(prob_2)
        
        return D, H0, H1, H2, P_grid

    def objective(theta):
        D, H0, H1, H2, _ = compute_entropies(theta)
        # We want to maximize D / max(H0, H1, H2)
        # Equivalent to minimizing -D + log(max(H0, H1, H2))
        # Use soft-max for differentiability if using gradient methods, 
        # but Nelder-Mead doesn't need derivatives.
        # However, the max function is non-smooth.
        # Let's use a soft-max with a small temperature to smooth it out slightly
        # or just use the hard max and let Nelder-Mead handle it.
        # Nelder-Mead can handle non-smooth functions but might be slow.
        # Let's try to maximize D - lambda * max(H0, H1, H2) with lambda tuned?
        # No, the ratio is the target.
        # Let's minimize: -D + soft_max(H0, H1, H2, T=1.0)
        
        H_vals = np.array([H0, H1, H2])
        # Soft max
        T = 1.0
        H_max_soft = T * np.log(np.sum(np.exp(H_vals / T)))
        
        return -D + H_max_soft

    # Initial guesses
    best_P = None
    best_score = -np.inf
    
    # Run 1: Uniform start
    theta0 = np.zeros(n*n)
    
    # Run 2: Random start
    np.random.seed(42)
    theta1 = np.random.randn(n*n) * 0.1
    
    # Run 3: Start with mass on a specific pattern, e.g., x=y
    theta2 = np.zeros(n*n)
    for i in range(n):
        theta2[i*n + i] = 1.0 # Diagonal
    
    initializations = [theta0, theta1, theta2]
    
    for i, theta_init in enumerate(initializations):
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, theta_init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            D, H0, H1, H2, P_grid = compute_entropies(res.x)
            M = max(H0, H1, H2)
            if M > 1e-9:
                score = D / M
            else:
                score = 0.0
            
            if score > best_score:
                best_score = score
                best_P = P_grid.copy()
                
        except Exception as e:
            pass

    # If no valid P found, return uniform
    if best_P is None:
        best_P = np.ones((n, n)) / (n*n)
        
    # Final validation and clipping
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s <= 1e-9:
        best_P = np.ones((n, n)) / (n*n)
    else:
        best_P = best_P / s
        
    return best_P
