# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Strategy:
    # 1. Formulate the problem as a Linear Programming problem.
    #    Variables: v_k for k=2..K, where f(k) = -v_k.
    #    Objective: Maximize sum_{k=2}^K v_k * log(k)/k.
    #    Constraints: sum_{k=2}^K v_k * {x/k} <= 1 for all x in [1, 10K].
    #                 v_k in [0, 10].
    #    f(1) is set to 0 initially. The evaluator adjusts f(1) to make sum f(k)/k = 0.
    #    The adjusted f(1) does not affect the objective (log(1)=0) and the feasibility
    #    is guaranteed by the LP constraints on the tail.
    
    best_domain = None
    best_values = None
    best_score = -np.inf
    
    # We try increasing K.
    # K=2000 -> 20000 constraints, 1999 variables. Matrix ~ 160 MB. Solvable in seconds.
    # K=5000 -> 50000 constraints, 4999 variables. Matrix ~ 1 GB. Might be slow/memory heavy.
    # Let's stick to K=2000 or 3000.
    
    for K in [100, 500, 1000, 2000]:
        if time.time() > deadline:
            break
        
        try:
            n_vars = K - 1
            ks = np.arange(2, K + 1, dtype=np.float64)
            
            # Objective coefficients: maximize sum v_k * log(k)/k
            # linprog minimizes, so we use -c
            c = -np.log(ks) / ks
            
            # Constraints: A v <= b
            # A has shape (10K, K-1)
            # A[i, j] = { (i+1) / ks[j] }
            x_max = 10 * K
            xs = np.arange(1, x_max + 1, dtype=np.float64)
            
            # Construct A vectorized
            # A = (xs[:, None] / ks[None, :]) % 1.0
            # This creates a large array. For K=2000, 20000x1999 ~ 40M floats ~ 320MB.
            # This is acceptable.
            
            A = (xs[:, None] / ks[None, :]) % 1.0
            b = np.ones(x_max)
            
            bounds = [(0, 10)] * n_vars
            
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            
            if res.success:
                v = res.x
                # f(k) = -v_k for k>=2
                # f(1) = 0
                
                values = np.zeros(K, dtype=np.float64)
                values[1:] = -v
                domain = np.arange(1, K + 1, dtype=np.int64)
                
                # Calculate the score (objective)
                # a = -sum f(k) log(k) / k
                # a = - (0 + sum_{k=2} (-v_k) log(k)/k) = sum v_k log(k)/k
                a = np.sum(v * np.log(ks) / ks)
                
                # Verify feasibility quickly for this specific solution
                # M(x) = sum_{k=1}^K f(k) floor(x/k)
                # f(1) is adjusted by evaluator.
                # Effective f(1) = - sum_{k=2} f(k)/k = sum v_k/k.
                # M(x) = (sum v_k/k) * x + sum_{k=2} (-v_k) floor(x/k)
                # M(x) = sum v_k/k * x - sum v_k floor(x/k)
                # M(x) = sum v_k (x/k - floor(x/k)) = sum v_k {x/k}
                # The LP ensured sum v_k {x/k} <= 1.
                # Due to floating point errors, it might be slightly > 1.
                # We can check the max.
                # M_vals = A @ v
                # max_M = np.max(M_vals)
                # if max_M > 1.0001:
                #     # Scale down v to be safe
                #     scale = 1.0 / max_M
                #     v = v * scale
                #     values[1:] = -v
                #     a = a * scale
                #     # Re-check
                #     max_M = np.max(A @ v)
                #     if max_M > 1.0001:
                #         continue # Skip this K if still infeasible
                
                # It is safer to just scale if needed
                max_M = np.max(A @ v)
                if max_M > 1.0:
                    scale = 1.0 / max_M
                    v = v * scale
                    values[1:] = -v
                    a = a * scale
                
                if a > best_score:
                    best_score = a
                    best_domain = domain
                    best_values = values
                    
        except Exception as e:
            print(f"Error for K={K}: {e}")
            continue

    # Fallback to seed if no LP solution found
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values

# EVOLVE-BLOCK-END
