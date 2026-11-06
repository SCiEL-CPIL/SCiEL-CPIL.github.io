import math
from functools import lru_cache

def construct():
    N = 12
    target = math.factorial(N)
    
    # 1. Generate all divisors of target
    divisors = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
        i += 1
    
    # Filter divisors > 12
    valid_divisors = [d for d in divisors if d > 12]
    valid_divisors.sort()
    
    # Pre-compute divisors for each divisor of target
    # This allows O(1) lookup of "what are the divisors of X?"
    # Since we only care about divisors of target, we can map each divisor to its divisors.
    # However, storing a list for all 6615 divisors might be memory heavy if lists are large.
    # Average number of divisors of a divisor is small.
    # Let's create a dictionary: rem_prod -> sorted list of its divisors that are > 12.
    
    divisor_map = {}
    for d in valid_divisors:
        # Find all divisors of d that are > 12
        # Since d is a divisor of target, its divisors are in the global list.
        # We can just filter valid_divisors? No, that's O(N^2).
        # Better: iterate i up to sqrt(d)
        divs_of_d = []
        j = 1
        while j * j <= d:
            if d % j == 0:
                if j > 12:
                    divs_of_d.append(j)
                other = d // j
                if other != j and other > 12:
                    divs_of_d.append(other)
            j += 1
        divs_of_d.sort()
        divisor_map[d] = divs_of_d
    
    # Also, target itself is a divisor. We need its divisors too.
    # The loop above only populated for d in valid_divisors.
    # target is in valid_divisors? Yes, 12! > 12.
    # So divisor_map[target] is populated.
    
    # We also need to handle the case where rem_prod becomes 1.
    # We don't need divisor_map[1].
    
    # Helper to check if a solution exists with max factor <= M
    def can_partition(M):
        # Memoization? The state is (rem_prod, min_next).
        # rem_prod can be any divisor of target. min_next can be up to M.
        # This state space is large. Let's rely on pruning.
        
        def dfs(rem_prod, min_next, path):
            if rem_prod == 1:
                return True
            
            # Pruning 1: Not enough slots? 
            # We don't track count explicitly, but we can infer.
            # Actually, we should track count to ensure <= 12 factors.
            # Let's add count to state.
            
            # Pruning: Min possible product of remaining factors
            # We don't know how many factors remain.
            # But we know the max number of factors is 12 - len(path).
            # And min number of factors is 1.
            
            # Let's refine: 
            # If rem_prod is prime, it must be the last factor.
            # If rem_prod is composite, it can be split.
            
            # Stronger Pruning:
            # Let k_min = 1, k_max = 12 - len(path)
            # If we pick 1 factor, it must be rem_prod.
            # If rem_prod > M, we can't pick it as a single factor.
            # If rem_prod < min_next, we can't pick it as a single factor (distinctness/order).
            
            # Let's iterate candidates.
            # Candidates are divisors of rem_prod, in range [min_next, M].
            
            candidates = divisor_map.get(rem_prod, [])
            
            # If rem_prod is not in divisor_map, it means it has no divisors > 12?
            # Or it's 1? No, rem_prod > 1 here.
            # If rem_prod is prime and > 12, its divisors are [1, rem_prod].
            # So divisor_map[rem_prod] should contain rem_prod.
            
            for d in candidates:
                if d < min_next:
                    continue
                if d > M:
                    break # Since candidates are sorted
                
                # If we pick d, new_rem = rem_prod // d
                new_rem = rem_prod // d
                
                # Pruning on new_rem:
                # 1. new_rem must be 1 or factorable into distinct factors >= d+1 and <= M.
                
                # Quick check: If new_rem == 1, success.
                if new_rem == 1:
                    return True
                
                # If new_rem < min_next + (d - min_next + 1)? No.
                # Next min factor is d + 1.
                
                # Pruning: Can new_rem be formed by factors in [d+1, M]?
                # Min product of 1 factor: d+1.
                # If new_rem < d+1, impossible (unless new_rem==1, handled).
                # If new_rem is prime:
                #   It must be a single factor.
                #   So new_rem must be >= d+1 and <= M.
                
                # Let's use a recursive call with tighter bounds.
                # But before recursing, apply product bounds.
                
                # Max number of factors we can still use: 12 - len(path) - 1
                # Min number of factors: 1
                
                # Bound 1: Minimum product of k distinct integers >= d+1.
                # The smallest k integers are d+1, d+2, ..., d+k.
                # Product is P_min(k) = prod_{i=1}^k (d+i)
                
                # Bound 2: Maximum product of k distinct integers <= M.
                # The largest k integers are M, M-1, ..., M-k+1.
                # Product is P_max(k) = prod_{i=0}^{k-1} (M-i)
                
                # We need to exist k in [1, 12-len(path)-1] such that P_min(k) <= new_rem <= P_max(k).
                
                max_factors_left = 12 - len(path) - 1
                if max_factors_left < 1:
                    continue
                
                # Check if new_rem is within the possible range for ANY k in [1, max_factors_left]
                feasible = False
                for k in range(1, max_factors_left + 1):
                    # Calculate P_min(k)
                    p_min = 1
                    for i in range(1, k+1):
                        p_min *= (d + i)
                        if p_min > new_rem:
                            break
                    if p_min > new_rem:
                        break # For larger k, p_min will only increase
                    
                    # Calculate P_max(k)
                    p_max = 1
                    for i in range(k):
                        p_max *= (M - i)
                        if p_max < new_rem:
                            break # For larger k, p_max will only increase? 
                            # Wait. M, M-1... Product increases as k increases (for M > 1).
                            # So if p_max < new_rem for this k, it might be feasible for larger k?
                            # No, we are checking if new_rem <= P_max(k).
                            # If P_max(k) < new_rem, we need a larger k to get a larger max product?
                            # Yes, adding more factors (even if they are small) increases the total product capacity?
                            # Wait. The factors are distinct and <= M.
                            # The maximum product of k distinct factors <= M is indeed M(M-1)...(M-k+1).
                            # As k increases, this product increases.
                            # So if P_max(k) < new_rem, we need to try larger k.
                            
                    # If we broke early from p_min loop, p_min > new_rem, so this k is too large for min product?
                    # No, if p_min > new_rem, it means the smallest possible product for k factors is already too big.
                    # So any k' > k will have even larger min product. So we can break the k-loop entirely?
                    # Yes.
                    
                    if p_min > new_rem:
                        break
                    
                    # Check p_max
                    if p_max >= new_rem:
                        # It is possible that new_rem can be formed by k factors.
                        # We don't know for sure, but it's not ruled out by product bounds.
                        feasible = True
                        break
                    else:
                        # p_max < new_rem. Try larger k.
                        continue
                
                if not feasible:
                    continue
                
                # Recurse
                if dfs(new_rem, d + 1, path + [d]):
                    return True
            
            return False
        
        return dfs(target, 13, [])

    # Binary search for minimal M
    # Lower bound: 13
    # Upper bound: target (single factor)
    # Actually, we can estimate a better upper bound.
    # 12! = 479001600.
    # sqrt(12!) ~ 21886.
    # We can start high and go down, or binary search.
    
    lo = 13
    hi = target
    best_M = target
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if can_partition(mid):
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    # Now we need to find the actual solution with max factor = best_M
    # We can modify can_partition to return the path.
    
    def find_solution(M):
        def dfs(rem_prod, min_next, path):
            if rem_prod == 1:
                return path[:]
            
            candidates = divisor_map.get(rem_prod, [])
            
            for d in candidates:
                if d < min_next:
                    continue
                if d > M:
                    break
                
                new_rem = rem_prod // d
                
                if new_rem == 1:
                    return path + [d]
                
                max_factors_left = 12 - len(path) - 1
                if max_factors_left < 1:
                    continue
                
                feasible = False
                for k in range(1, max_factors_left + 1):
                    p_min = 1
                    for i in range(1, k+1):
                        p_min *= (d + i)
                        if p_min > new_rem:
                            break
                    if p_min > new_rem:
                        break
                    
                    p_max = 1
                    for i in range(k):
                        p_max *= (M - i)
                        if p_max < new_rem:
                            break
                    
                    if p_min <= new_rem <= p_max:
                        feasible = True
                        break
                    
                if not feasible:
                    continue
                
                res = dfs(new_rem, d + 1, path + [d])
                if res:
                    return res
            
            return None
        
        return dfs(target, 13, [])

    solution = find_solution(best_M)
    
    if not solution:
        # Fallback: return a valid solution even if not optimal? 
        # The task requires minimizing the largest. If binary search failed to find one, 
        # something is wrong. But target itself is a valid solution (1 factor).
        # So best_M should at most be target.
        # If find_solution returns None, it means can_partition(best_M) was True but find_solution failed?
        # This shouldn't happen if logic is identical.
        # Let's just return [target] as a safe fallback, though it's likely not minimal.
        return [target]
        
    return solution
