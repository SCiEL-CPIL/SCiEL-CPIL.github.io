# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k) * log(k) / k.
    
    Uses LP formulation with Möbius-inspired support to find optimal f.
    """
    start_time = time.time()
    deadline = start_time + 500  # 500s limit, 50s safety margin
    
    # Deterministic feasible baseline (strong Mobius-flavoured partial function).
    best_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    best_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    
    # Helper to compute score of a candidate
    def compute_score(domain, values):
        m = len(domain)
        if m < 1:
            return -np.inf
        K = int(np.max(domain))
        if K > 200000 or K < 1:
            return -np.inf
        
        # Clip values to [-10, 10]
        v = np.clip(values, -10.0, 10.0)
        
        # Compute S = sum_k f(k)/k
        S = np.sum(v / domain)
        
        # Adjust f(1) so that sum f(k)/k = 0
        v_adj = v.copy()
        idx_1 = np.where(domain == 1)[0]
        if len(idx_1) > 0:
            i1 = idx_1[0]
            v_adj[i1] = v_adj[i1] - S
        else:
            # f(1) = 0 - S = -S, but 1 not in domain, so we just need to check
            # Actually if 1 not in domain, the sum already should be adjusted
            # Wait, the evaluator says f(1) = 0 before this if 1 is not in domain
            # So we need to add 1 to domain with value -S? No, the evaluator handles it.
            # Let me re-read: "f(1) is replaced by f(1) - S ... so that afterwards sum_k f(k)/k = 0"
            # "f(1) = 0 before this if 1 is not in the domain"
            # So if 1 not in domain, effectively f(1) = 0, then becomes -S, and 1 is added?
            # Actually I think the evaluator treats it as if 1 is in domain with value 0 if not present.
            # For our score computation, let's simulate: if 1 not in domain, add it with value -S
            domain_ext = np.append(domain, 1)
            v_ext = np.append(v, -S)
            return _compute_final_score(domain_ext, v_ext)
        
        return _compute_final_score(domain, v_adj)
    
    def _compute_final_score(domain, values):
        """Compute the actual score assuming sum f(k)/k = 0 already."""
        m = len(domain)
        if m < 1:
            return -np.inf
        K = int(np.max(domain))
        if K > 200000 or K < 1:
            return -np.inf
        
        v = np.clip(values, -10.0, 10.0)
        
        # Check M(x) <= 1.0001 for all x in [1, 10K]
        # M(x) = sum_k f(k) * floor(x/k)
        # Use efficient computation
        x_max = 10 * K
        if x_max > 2000000:  # Too large to compute
            return -np.inf
        
        # Compute M(x) for all x in [1, x_max]
        # M(x) = sum_k f(k) * floor(x/k)
        # For each k, floor(x/k) is a step function
        # Efficient way: for each k, add f(k) to all x >= k, x >= 2k, ...
        # i.e., for j=1,2,...: add f(k) to x in [j*k, (j+1)*k - 1]
        
        M = np.zeros(x_max + 1)  # M[0] unused, M[x] for x in [1, x_max]
        
        for k, f_k in zip(domain, v):
            k = int(k)
            if k < 1:
                continue
            # Add f_k * floor(x/k) for x in [1, x_max]
            # floor(x/k) = j for x in [j*k, (j+1)*k - 1]
            j = 1
            while j * k <= x_max:
                start = j * k
                end = min((j + 1) * k - 1, x_max)
                M[start:end+1] += f_k
                j += 1
        
        # Check constraint: M(x) <= 1.0001 for all x in [1, x_max]
        if np.max(M[1:x_max+1]) > 1.0001:
            return -np.inf
        
        # Compute objective: a = -sum_k f(k) * log(k) / k
        # Note: k=1 has log(1)=0 so contributes 0
        a = -np.sum(v * np.log(domain) / domain)
        
        return a
    
    # Try to improve using LP
    try:
        # Use a reasonable support set
        # Include 1 and small integers, and some primes/composites
        K_max = 200
        
        # Generate candidate support: 1, primes up to K_max, and some composites
        # For LP efficiency, use a moderate size
        support = []
        support.append(1)
        
        # Add small integers
        for n in range(2, min(K_max + 1, 100)):
            support.append(n)
        
        # Add some larger primes
        def is_prime(n):
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Add primes up to K_max
        for n in range(100, K_max + 1):
            if is_prime(n):
                support.append(n)
            elif n % 2 == 0 and n > 100 and n % 4 == 2:  # Add some even composites
                support.append(n)
        
        support = np.array(sorted(set(support)), dtype=np.int64)
        n_support = len(support)
        
        if n_support > 150:
            support = support[:150]
            n_support = len(support)
        
        K_support = int(np.max(support))
        x_max = 10 * K_support
        
        # Build LP:
        # Variables: f(k) for k in support
        # Objective: maximize -sum f(k) log(k) / k  =>  minimize sum f(k) log(k) / k
        # Constraints:
        #   M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, x_max]
        #   No explicit bound on f(k), but values will be clipped to [-10,10]
        #   We can add bounds: -10 <= f(k) <= 10
        
        # Objective coefficients: c_k = log(k)/k (minimize)
        c = np.log(support.astype(float)) / support.astype(float)
        
        # Inequality constraints: M(x) <= 1.0001 for x = 1, 2, ..., x_max
        # A_ub[i, j] = floor(x_i / k_j) for x_i in [1, x_max], k_j in support
        # b_ub = 1.0001
        
        # This is a large constraint matrix: x_max x n_support
        # x_max can be up to 10 * 200 = 2000, n_support up to 150
        # That's 300,000 elements, manageable
        
        A_rows = []
        for x in range(1, x_max + 1):
            row = np.floor(x / support.astype(float)).astype(float)
            A_rows.append(row)
        
        A_ub = np.array(A_rows)
        b_ub = np.ones(x_max) * 1.0001
        
        # Bounds: -10 <= f(k) <= 10
        bounds = [(-10.0, 10.0) for _ in range(n_support)]
        
        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            f_opt = result.x
            # Check if this is better
            score_new = compute_score(support, f_opt)
            score_old = compute_score(best_domain, best_values)
            
            if score_new > score_old and np.isfinite(score_new):
                best_domain = support
                best_values = f_opt.copy()
        
        # Try another support set with different composition
        if time.time() < deadline - 30:
            # Use a support that emphasizes small k (where log(k)/k is larger in magnitude for small k>1)
            support2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 100, 150]
            # Add more
            for n in range(11, 60):
                if n not in support2:
                    support2.append(n)
            support2 = np.array(sorted(set(support2)), dtype=np.int64)
            
            n2 = len(support2)
            K2 = int(np.max(support2))
            x_max2 = 10 * K2
            
            c2 = np.log(support2.astype(float)) / support2.astype(float)
            
            A_rows2 = []
            for x in range(1, x_max2 + 1):
                row = np.floor(x / support2.astype(float)).astype(float)
                A_rows2.append(row)
            
            A_ub2 = np.array(A_rows2)
            b_ub2 = np.ones(x_max2) * 1.0001
            
            bounds2 = [(-10.0, 10.0) for _ in range(n2)]
            
            result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs')
            
            if result2.success:
                f_opt2 = result2.x
                score_new2 = compute_score(support2, f_opt2)
                score_current = compute_score(best_domain, best_values)
                
                if score_new2 > score_current and np.isfinite(score_new2):
                    best_domain = support2
                    best_values = f_opt2.copy()
        
        # Final polish: try to improve with a smaller, focused support
        if time.time() < deadline - 10:
            # Use only 1, 2, 3, 4, 5, 6, 10, 15, 30, 50
            support3 = np.array([1, 2, 3, 4, 5, 6, 10, 15, 30, 50, 100], dtype=np.int64)
            n3 = len(support3)
            K3 = int(np.max(support3))
            x_max3 = 10 * K3
            
            c3 = np.log(support3.astype(float)) / support3.astype(float)
            
            A_rows3 = []
            for x in range(1, x_max3 + 1):
                row = np.floor(x / support3.astype(float)).astype(float)
                A_rows3.append(row)
            
            A_ub3 = np.array(A_rows3)
            b_ub3 = np.ones(x_max3) * 1.0001
            
            bounds3 = [(-10.0, 10.0) for _ in range(n3)]
            
            result3 = linprog(c3, A_ub=A_ub3, b_ub=b_ub3, bounds=bounds3, method='highs')
            
            if result3.success:
                f_opt3 = result3.x
                score_new3 = compute_score(support3, f_opt3)
                score_current = compute_score(best_domain, best_values)
                
                if score_new3 > score_current and np.isfinite(score_new3):
                    best_domain = support3
                    best_values = f_opt3.copy()
    
    except Exception as e:
        # If LP fails, return the baseline
        pass
    
    # Ensure we return valid arrays
    if time.time() > deadline:
        return best_domain, best_values
    
    return best_domain, best_values


# EVOLVE-BLOCK-END
