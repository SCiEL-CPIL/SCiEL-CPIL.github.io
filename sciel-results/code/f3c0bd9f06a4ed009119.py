import time
import bisect
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = factorial(12)  # 479001600
    
    # Generate all divisors of target
    def get_divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return sorted(divs)
    
    all_divs = get_divisors(target)
    # Filter: > 12
    candidates = [d for d in all_divs if d > 12]
    candidates.sort()  # Ascending order
    n_cands = len(candidates)
    
    # Create a set for O(1) lookup
    cand_set = set(candidates)
    
    # Helper: check if a number is in candidates and >= min_val and <= max_val
    def in_candidates(val, min_val, max_val):
        if val < min_val or val > max_val:
            return False
        return val in cand_set
    
    # Find the index of a value in candidates (using bisect)
    def find_index(val):
        idx = bisect.bisect_left(candidates, val)
        if idx < n_cands and candidates[idx] == val:
            return idx
        return -1
    
    # Check feasibility: can we decompose 'remaining' into distinct candidates
    # from candidates[start_idx:] such that all are <= M?
    # We use memoization on (remaining, start_idx) for failed states
    def can_decompose(M):
        # Candidates available: candidates[:end_idx] where end_idx = bisect_right(candidates, M)
        end_idx = bisect.bisect_right(candidates, M)
        if end_idx == 0:
            return False
        
        available = candidates[:end_idx]
        avail_set = set(available)
        n_avail = len(available)
        
        # Memoization: set of (remaining, start_idx) that failed
        failed = set()
        
        def backtrack(remaining, start_idx, count):
            if remaining == 1:
                return True
            if count >= 12:
                return False
            if start_idx >= n_avail:
                return False
            
            key = (remaining, start_idx)
            if key in failed:
                return False
            
            # Quick check: if remaining itself is in available[start_idx:]
            # and remaining <= M (which it is since we're checking available)
            pos = bisect.bisect_left(available, remaining, start_idx)
            if pos < n_avail and available[pos] == remaining:
                return True
            
            # Pruning: if remaining is prime, it must be directly in available
            # (since it can't be factored further into integers > 1)
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
            
            if is_prime(remaining):
                # Must be directly in available[start_idx:]
                pos = bisect.bisect_left(available, remaining, start_idx)
                if pos < n_avail and available[pos] == remaining:
                    return True
                failed.add(key)
                return False
            
            for i in range(start_idx, n_avail):
                c = available[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    if backtrack(remaining // c, i + 1, count + 1):
                        return True
            
            failed.add(key)
            return False
        
        return backtrack(target, 0, 0)
    
    # Binary search for the minimum M
    # Lower bound: 13 (smallest valid factor)
    # Upper bound: target (trivial solution [target])
    low = 13
    high = target
    
    # First, try a linear scan for small M values (likely optimal is small)
    found_M = None
    for M in range(13, 300):
        if time.time() > deadline:
            break
        if can_decompose(M):
            found_M = M
            break
    
    if found_M is None:
        # Binary search in [300, target]
        low = 300
        high = target
        while low <= high:
            if time.time() > deadline:
                break
            mid = (low + high) // 2
            if can_decompose(mid):
                found_M = mid
                high = mid - 1
            else:
                low = mid + 1
    
    if found_M is None:
        # Fallback: return [target]
        return [target]
    
    # Reconstruct the solution for found_M
    def find_solution(M):
        end_idx = bisect.bisect_right(candidates, M)
        available = candidates[:end_idx]
        n_avail = len(available)
        
        res = []
        failed = set()
        
        def backtrack(remaining, start_idx, count):
            if remaining == 1:
                return True
            if count >= 12:
                return False
            if start_idx >= n_avail:
                return False
            
            key = (remaining, start_idx)
            if key in failed:
                return False
            
            # Quick check: if remaining itself is in available[start_idx:]
            pos = bisect.bisect_left(available, remaining, start_idx)
            if pos < n_avail and available[pos] == remaining:
                res.append(remaining)
                return True
            
            for i in range(start_idx, n_avail):
                c = available[i]
                if c > remaining:
                    break
                if remaining % c == 0:
                    res.append(c)
                    if backtrack(remaining // c, i + 1, count + 1):
                        return True
                    res.pop()
            
            failed.add(key)
            return False
        
        if backtrack(target, 0, 0):
            return res
        return None
    
    sol = find_solution(found_M)
    
    if sol is None:
        return [target]
    
    return sol
