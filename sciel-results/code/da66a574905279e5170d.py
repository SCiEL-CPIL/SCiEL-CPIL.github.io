import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # Try different K values
    # For each K, we solve a linear program:
    # Minimize: sum_{k=2}^{K} (log(k)/k) * g(k)
    # Subject to:
    #   -10 <= g(k) <= 10 for k=2..K
    #   sum_{k=2}^{K} g(k)/k <= 10  (so that f(1) = -S >= -10)
    #   -sum_{k=2}^{K} g(k)/k <= 10  (so that f(1) = -S <= 10)
    #   For all x in [1, 10K]: sum_{k=2}^{K} g(k) * floor(x/k) <= 1 - S*floor(x/1) = 1 - S
    #     Wait, M(x) = sum_{k in domain} f(k) * floor(x/k)
    #     domain includes 1, so M(x) = f(1)*floor(x/1) + sum_{k=2}^K f(k)*floor(x/k)
    #     f(1) = -S where S = sum_{k=1}^K f(k)/k = f(1)/1 + sum_{k=2}^K f(k)/k = -S + sum_{k=2}^K g(k)/k
    #     So S = -S + sum_{k=2}^K g(k)/k => 2S = sum_{k=2}^K g(k)/k => S = (1/2) sum_{k=2}^K g(k)/k
    #     Wait, let me re-read the task.
    #     "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the clipped values"
    #     So initially f(1) is some value (or 0 if 1 not in domain).
    #     Let's say we provide domain = [1, 2, ..., K] and values = [f(1), f(2), ..., f(K)].
    #     The evaluator clips values to [-10, 10].
    #     Then S = sum_{k=1}^K f(k)/k (using clipped values).
    #     Then f(1) becomes f(1) - S.
    #     The new sum of f(k)/k is S - S = 0.
    #     So effectively, we can choose g(k) = f(k) for k=2..K, and f(1) is determined.
    #     But we need to ensure that after the adjustment, f(1) is still in [-10, 10] (or at least the clipped value is used).
    #     Actually, the evaluator clips first, then adjusts f(1). So if f(1) - S is outside [-10, 10], it gets clipped.
    #     But the constraint is on M(x) = sum_{k} f(k) * floor(x/k) where f(k) are the FINAL values (after clipping and adjustment).
    #     This is tricky. Let's assume we choose f(1) such that after adjustment it's still in [-10, 10].
    #     Let g(k) = f(k) for k=2..K. Let f(1) = y.
    #     S = y/1 + sum_{k=2}^K g(k)/k = y + T, where T = sum_{k=2}^K g(k)/k.
    #     New f(1) = y - S = y - (y + T) = -T.
    #     So the final f(1) is -T, regardless of y!
    #     Wait, that's interesting. The final f(1) is always -T.
    #     So we need -T to be in [-10, 10] for it not to be clipped. If it's clipped, then the final f(1) is clipped(-T).
    #     Let's assume we keep T in [-10, 10] so that f(1) = -T is not clipped.
    #     Then M(x) = (-T) * floor(x/1) + sum_{k=2}^K g(k) * floor(x/k) = -T*x + sum_{k=2}^K g(k) * floor(x/k).
    #     We need M(x) <= 1 for all x in [1, 10K].
    #     So: sum_{k=2}^K g(k) * floor(x/k) - T*x <= 1.
    #     Note that floor(x/k) = x/k - {x/k}, so:
    #     sum_{k=2}^K g(k) * (x/k - {x/k}) - T*x <= 1
    #     x * sum_{k=2}^K g(k)/k - sum_{k=2}^K g(k) * {x/k} - T*x <= 1
    #     x * T - sum_{k=2}^K g(k) * {x/k} - T*x <= 1
    #     -sum_{k=2}^K g(k) * {x/k} <= 1
    #     sum_{k=2}^K g(k) * {x/k} >= -1
    #     
    #     So the constraint is: for all x in [1, 10K], sum_{k=2}^K g(k) * {x/k} >= -1.
    #     This is a linear constraint in g.
    #     
    #     Objective: minimize sum_{k=2}^K (log(k)/k) * g(k).
    #     Bounds: -10 <= g(k) <= 10.
    #     Additional: -10 <= T <= 10, where T = sum_{k=2}^K g(k)/k.
    
    for K in [50, 100, 200, 500, 1000, 2000, 3000]:
        if time.time() - start_time > deadline - 30:
            break
            
        n_vars = K - 1
        if n_vars <= 0:
            continue
            
        ks = np.arange(2, K + 1)
        c = np.log(ks) / ks
        
        max_x = 10 * K
        
        # Vectorized construction of A_ub
        # Constraint: sum_{k=2}^K g(k) * {x/k} >= -1
        # => -sum_{k=2}^K g(k) * {x/k} <= 1
        x_vals = np.arange(1, max_x + 1, dtype=np.float64)
        
        try:
            # Memory: max_x * n_vars
            # For K=3000, 30000 * 3000 = 90M floats ~ 720MB. Might be too much.
            # Let's be careful. For K=2000, 20000 * 2000 = 40M ~ 320MB.
            # For K=3000, let's try but might fail.
            
            # {x/k} = (x % k) / k
            A_ub = -(x_vals[:, None] % ks[None, :]) / ks[None, :]
            b_ub = np.ones(max_x)
            
            # Add constraints for T = sum g(k)/k in [-10, 10]
            # T <= 10 => sum g(k)/k <= 10
            # T >= -10 => -sum g(k)/k <= 10
            
            row_pos = 1.0 / ks
            row_neg = -1.0 / ks
            
            A_ub_full = np.vstack([A_ub, row_pos, row_neg])
            b_ub_full = np.concatenate([b_ub, [10.0, 10.0]])
            
            bounds = [(-10, 10)] * n_vars
            
            res = linprog(c, A_ub=A_ub_full, b_ub=b_ub_full, bounds=bounds, method='highs')
            
            if res.success:
                g = res.x
                T = np.sum(g / ks)
                
                # Check T is in [-10, 10]
                if abs(T) <= 10.0 + 1e-9:
                    f = np.zeros(K + 1)
                    f[2:K+1] = g
                    f[1] = -T  # This is the final f(1)
                    
                    # Verify f(1) is within bounds (should be by construction)
                    if abs(f[1]) <= 10.0 + 1e-9:
                        a = -np.sum(g * c)
                        if a > best_score:
                            best_score = a
                            domain = np.arange(1, K + 1)
                            values = np.clip(f[1:K+1], -10, 10)
                            best_domain = domain
                            best_values = values
                            
        except Exception as e:
            pass

    if best_domain is not None:
        return best_domain, best_values
    else:
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
