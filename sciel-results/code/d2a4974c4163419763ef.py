import numpy as np
from scipy.optimize import minimize

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average probability measure.
    
    Uses a deterministic grid search over 3-point measures of the form {0, x, 1}
    and compares against the optimal 2-point measure.
    """
    
    def calc_P_fast(s0, s1, s2, w0, w1, w2):
        """
        Calculate P for 3 points {s0, s1, s2} with weights {w0, w1, w2}.
        Assumes s0=0, s2=1, s1=x.
        w0+w1+w2 = 1.
        """
        s = np.array([s0, s1, s2])
        w = np.array([w0, w1, w2])
        
        # Pair sums and weights
        # s: 0, x, 1
        # s_i + s_j:
        # 0+0=0, 0+x=x, 0+1=1
        # x+0=x, x+x=2x, x+1=x+1
        # 1+0=1, 1+x=x+1, 1+1=2
        
        # Weights w_i w_j
        # w0w0, w0w1, w0w2
        # w1w0, w1w1, w1w2
        # w2w0, w2w1, w2w2
        
        # Threshold T = 2*s_l - s_k
        # s_l in {0, x, 1}
        # s_k in {0, x, 1}
        
        P = 0.0
        
        # Precompute pair sum matrix and weight matrix
        # This is small (3x3), so we can just do it inline or with small arrays
        
        pair_sums = np.array([
            [0.0, s1, s2],
            [s1, 2.0*s1, s1+s2],
            [s2, s1+s2, 2.0*s2]
        ])
        
        pair_weights = np.array([
            [w0*w0, w0*w1, w0*w2],
            [w1*w0, w1*w1, w1*w2],
            [w2*w0, w2*w1, w2*w2]
        ])
        
        for k in range(3):
            for l in range(3):
                threshold = 2.0 * s[l] - s[k]
                # Count pairs (i,j) with s_i+s_j < threshold
                # Mask
                mask = pair_sums < threshold
                inner = np.sum(pair_weights * mask)
                P += w[k] * w[l] * inner
                
        return P

    best_P = -1.0
    best_support = None
    best_weights = None
    
    # 1. Analytical 2-point solution
    p1 = (1.0 + np.sqrt(33.0)) / 16.0
    p0 = 1.0 - p1
    s2 = np.array([0.0, 1.0])
    w2 = np.array([p0, p1])
    
    # Calculate P for 2 points using the same logic or specific formula
    # P = p1 * (p0^3 + 3*p1*p0^2)
    P2 = p1 * (p0**3 + 3.0*p1*p0**2)
    
    if P2 > best_P:
        best_P = P2
        best_support = s2.copy()
        best_weights = w2.copy()
        
    # 2. Grid Search for 3-point measure {0, x, 1}
    # x in [0, 1]
    # w1, w2 in [0, 1] with w1+w2 <= 1
    
    # To speed up, we can reduce grid size or use vectorization.
    # Let's try 50x50x50 = 125,000 points.
    # 125k * (3x3 loops) is manageable.
    
    x_vals = np.linspace(0.0, 1.0, 51)
    w1_vals = np.linspace(0.0, 1.0, 51)
    w2_vals = np.linspace(0.0, 1.0, 51)
    
    for x in x_vals:
        s1 = x
        s0 = 0.0
        s2_val = 1.0
        
        # Precompute pair_sums for this x
        pair_sums = np.array([
            [0.0, s1, s2_val],
            [s1, 2.0*s1, s1+s2_val],
            [s2_val, s1+s2_val, 2.0*s2_val]
        ])
        
        for w1 in w1_vals:
            for w2 in w2_vals:
                w0 = 1.0 - w1 - w2
                if w0 < -1e-9:
                    continue
                if w0 < 0:
                    w0 = 0.0
                    
                pair_weights = np.array([
                    [w0*w0, w0*w1, w0*w2],
                    [w1*w0, w1*w1, w1*w2],
                    [w2*w0, w2*w1, w2*w2]
                ])
                
                P = 0.0
                # s = [0, x, 1]
                # w = [w0, w1, w2]
                
                # k=0 (s_k=0)
                # l=0 (s_l=0): T = 0. No pairs < 0.
                # l=1 (s_l=x): T = 2x.
                # l=2 (s_l=1): T = 2.
                
                # k=1 (s_k=x)
                # l=0 (s_l=0): T = -x.
                # l=1 (s_l=x): T = x.
                # l=2 (s_l=1): T = 2-x.
                
                # k=2 (s_k=1)
                # l=0 (s_l=0): T = -1.
                # l=1 (s_l=x): T = 2x-1.
                # l=2 (s_l=1): T = 1.
                
                # We can just use the loop for clarity and correctness, 
                # but unrolling might be faster.
                
                for k in range(3):
                    wk = [w0, w1, w2][k]
                    for l in range(3):
                        wl = [w0, w1, w2][l]
                        s_k = [0.0, s1, 1.0][k]
                        s_l = [0.0, s1, 1.0][l]
                        threshold = 2.0 * s_l - s_k
                        mask = pair_sums < threshold
                        inner = np.sum(pair_weights * mask)
                        P += wk * wl * inner
                        
                if P > best_P:
                    best_P = P
                    best_support = np.array([0.0, x, 1.0])
                    best_weights = np.array([w0, w1, w2])

    # Refine the best 3-point solution
    if best_support is not None and len(best_support) == 3:
        def obj(params):
            x = params[0]
            w1 = params[1]
            w2 = params[2]
            w0 = 1.0 - w1 - w2
            if w0 < 0 or w1 < 0 or w2 < 0:
                return 1.0
            return -calc_P_fast(0.0, x, 1.0, w0, w1, w2)
            
        x_best = best_support[1]
        w1_best = best_weights[1]
        w2_best = best_weights[2]
        x0 = [x_best, w1_best, w2_best]
        
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds)
        
        if res.fun < -best_P:
            x_opt = res.x[0]
            w1_opt = res.x[1]
            w2_opt = res.x[2]
            w0_opt = 1.0 - w1_opt - w2_opt
            if w0_opt >= 0:
                P_opt = -res.fun
                if P_opt > best_P:
                    best_P = P_opt
                    best_support = np.array([0.0, x_opt, 1.0])
                    best_weights = np.array([w0_opt, w1_opt, w2_opt])

    return best_support.astype(np.float64), best_weights.astype(np.float64)
