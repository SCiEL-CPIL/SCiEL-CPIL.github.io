import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = factorial(12)
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    target_exp = (10, 5, 2, 1, 1)
    
    # Generate all divisor exponent vectors (e2, e3, e5, e7, e11) where value > 12
    valid_vectors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7) * (11 ** e11)
                        if val > 12:
                            valid_vectors.append(((e2, e3, e5, e7, e11), val))
    
    # Sort by value ascending
    valid_vectors.sort(key=lambda x: x[1])
    n_vecs = len(valid_vectors)
    
    vec_vals = [v[1] for v in valid_vectors]
    vec_exps = [v[0] for v in valid_vectors]
    
    # Precompute suffix sums for pruning: for each index i and each dimension d,
    # sum of exps[d] for vectors from i to end
    # This is expensive, so let's compute on the fly or use a simpler bound.
    
    def can_solve(M, max_factors=12):
        """Check if we can decompose target into distinct factors in (12, M]."""
        end_idx = 0
        for i in range(n_vecs):
            if vec_vals[i] <= M:
                end_idx = i + 1
            else:
                break
        
        if end_idx == 0:
            return None
        
        res = []
        memo = {}
        
        def backtrack(idx, rem_exp, count):
            if rem_exp == (0, 0, 0, 0, 0):
                return True
            if count >= max_factors:
                return False
            if idx >= end_idx:
                return False
            
            # Memoization key: (idx, rem_exp, count)
            key = (idx, rem_exp, count)
            if key in memo:
                return memo[key]
            
            # Pruning: for each dimension, the remaining exponent must be coverable
            # by the vectors from idx onwards. A simple bound: the sum of exponents
            # in dimensions d for vectors idx..end_idx-1 must be >= rem_exp[d].
            # Compute this quickly.
            for d in range(5):
                if rem_exp[d] == 0:
                    continue
                # Check if we can still cover rem_exp[d]
                # The max contribution per vector in dimension d is at most the target_exp[d]
                # but we need to check the actual sum from idx to end_idx-1
                # This is expensive, so let's skip for now and rely on other pruning.
                pass
            
            # Try vectors from idx to end_idx-1
            # Heuristic: try smaller vectors first (they're already sorted ascending)
            for i in range(idx, end_idx):
                exp_i = vec_exps[i]
                val_i = vec_vals[i]
                
                # Check feasibility
                feasible = True
                new_rem = []
                for d in range(5):
                    if rem_exp[d] < exp_i[d]:
                        feasible = False
                        break
                    new_rem.append(rem_exp[d] - exp_i[d])
                
                if not feasible:
                    continue
                
                new_rem = tuple(new_rem)
                
                if new_rem == (0, 0, 0, 0, 0):
                    res.append(val_i)
                    memo[key] = True
                    return True
                
                res.append(val_i)
                if backtrack(i + 1, new_rem, count + 1):
                    memo[key] = True
                    return True
                res.pop()
            
            memo[key] = False
            return False
        
        if backtrack(0, target_exp, 0):
            return res
        return None
    
    # Binary search on M
    low = 13
    high = target
    
    best_sol = None
    
    while low <= high:
        if time.time() > deadline - 5.0:
            break
        mid = (low + high) // 2
        sol = can_solve(mid)
        if sol is not None:
            best_sol = sol
            high = mid - 1
        else:
            low = mid + 1
    
    if best_sol is not None:
        prod = 1
        for f in best_sol:
            prod *= f
        if prod == target and len(set(best_sol)) == len(best_sol) and all(f > 12 for f in best_sol):
            return best_sol
    
    return [target]
