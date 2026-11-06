"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55
    target_num = 5
    target_den = 31
    
    best = [8, 31, 248]
    best_max = 248
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def verify_int(denoms):
        num = 0
        den = 1
        for d in denoms:
            new_num = num * d + den
            new_den = den * d
            g = _gcd(new_num, new_den)
            num = new_num // g
            den = new_den // g
        return num == target_num and den == target_den
    
    def feasible(C, max_terms=8):
        """Check if 5/31 can be written as sum of distinct unit fractions
        with denominators in [1, C], using at most max_terms terms."""
        if C < 7:
            return False
        
        memo = {}
        
        def solve(rem_num, rem_den, min_d, terms_left, path):
            if time.time() > deadline:
                return None
            if terms_left == 0:
                return None if rem_num != 0 else list(path)
            
            # Reduce
            g = _gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            if rn == 0:
                return list(path)
            if rn < 0:
                return None
            
            key = (rn, rd, min_d, terms_left)
            if key in memo:
                return memo[key]
            
            # Bounds on next denominator d:
            # 1/d <= rem => d >= rd/rn
            # Also need terms_left-1 remaining, each >= 1/(d+1)
            # So (terms_left-1)/(d+1) >= rem - 1/d... complex
            # Simpler: 1/d < rem and 1/d + 1/(d+1) + ... + 1/(d+terms_left-2) >= rem
            # Lower bound: d > rd/rn, so d_min = floor(rd/rn) + 1
            # Upper bound: 1/d >= rem/(terms_left) => d <= terms_left * rd / rn
            d_min = rd // rn + 1
            d_min = max(d_min, min_d)
            d_max = min(C, terms_left * rd // rn)
            
            if d_min > d_max:
                memo[key] = None
                return None
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    memo[key] = None
                    return None
                # New remainder: rem - 1/d = (rn*d - rd)/(rd*d)
                new_rn = rn * d - rd
                if new_rn < 0:
                    continue
                new_rd = rd * d
                if new_rn == 0:
                    if terms_left >= 1:
                        result = path + [d]
                        memo[key] = result
                        return result
                    continue
                # Check if new_rn/new_rd can be represented with terms_left-1 more terms
                # Each term >= 1/(d+1), so sum of remaining <= (terms_left-1)/(d+1)
                # Need new_rn/new_rd <= (terms_left-1)/(d+1)
                # i.e., new_rn*(d+1) <= (terms_left-1)*new_rd
                if new_rn * (d + 1) > (terms_left - 1) * new_rd:
                    continue
                # Also need at least one more term if new_rn > 0
                if terms_left - 1 == 0:
                    continue
                
                path.append(d)
                result = solve(new_rn, new_rd, d + 1, terms_left - 1, path)
                path.pop()
                
                if result is not None:
                    memo[key] = result
                    return result
            
            memo[key] = None
            return None
        
        return solve(target_num, target_den, 1, max_terms, [])
    
    # Binary search for minimal C
    lo, hi = 7, 248
    # First verify we know a feasible solution
    if not feasible(248, 8):
        # Shouldn't happen, but handle gracefully
        return best
    
    while lo < hi:
        if time.time() > deadline - 5:
            break
        mid = (lo + hi) // 2
        if feasible(mid, 8):
            hi = mid
        else:
            lo = mid + 1
    
    C = lo
    # Find the actual solution with max denominator C
    for max_terms in range(2, 9):
        if time.time() > deadline - 3:
            break
        memo = {}
        
        def solve_find(rem_num, rem_den, min_d, terms_left, path):
            if time.time() > deadline:
                return None
            if terms_left == 0:
                return None if rem_num != 0 else list(path)
            
            g = _gcd(rem_num, rem_den)
            rn = rem_num // g
            rd = rem_den // g
            if rn == 0:
                return list(path)
            if rn < 0:
                return None
            
            key = (rn, rd, min_d, terms_left)
            if key in memo:
                return memo[key]
            
            d_min = rd // rn + 1
            d_min = max(d_min, min_d)
            d_max = min(C, terms_left * rd // rn)
            
            if d_min > d_max:
                memo[key] = None
                return None
            
            for d in range(d_min, d_max + 1):
                if time.time() > deadline:
                    memo[key] = None
                    return None
                new_rn = rn * d - rd
                if new_rn < 0:
                    continue
                new_rd = rd * d
                if new_rn == 0:
                    if terms_left >= 1:
                        result = path + [d]
                        memo[key] = result
                        return result
                    continue
                if new_rn * (d + 1) > (terms_left - 1) * new_rd:
                    continue
                if terms_left - 1 == 0:
                    continue
                
                path.append(d)
                result = solve_find(new_rn, new_rd, d + 1, terms_left - 1, path)
                path.pop()
                
                if result is not None:
                    memo[key] = result
                    return result
            
            memo[key] = None
            return None
        
        result = solve_find(target_num, target_den, 1, max_terms, [])
        if result is not None and verify_int(result):
            if max(result) < best_max:
                best_max = max(result)
                best = result
            break
    
    return best
# EVOLVE-BLOCK-END
