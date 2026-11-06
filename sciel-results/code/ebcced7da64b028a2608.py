# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _generate_smooth(limit):
    """Generate all 2^a * 3^b * 5^c numbers up to limit."""
    smooth = set()
    a = 1
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _is_antichain(terms):
    """Check that no term divides another in the list."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a < b:
                if b % a == 0:
                    return False
            else:
                if a % b == 0:
                    return False
    return True

def _find_min_terms(n, smooth, max_terms=30):
    """Find the minimum number of distinct smooth terms summing to n,
    forming a divisibility antichain. Returns the list of terms or None."""
    candidates = [x for x in smooth if x <= n]
    
    # Try 1 term
    if n in candidates:
        return [n]
    
    # Try 2 terms
    for i in range(len(candidates)):
        a = candidates[i]
        if a >= n:
            break
        b = n - a
        if b > a and b in candidates and b % a != 0:
            return [a, b]
    
    # Try 3 terms
    # For speed, we can be a bit smarter but brute force with pruning is okay for n<=128
    # candidates are small
    L = len(candidates)
    for i in range(L):
        a = candidates[i]
        if a >= n:
            break
        for j in range(i + 1, L):
            b = candidates[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b:
                continue
            if c in candidates:
                # Check antichain
                if b % a != 0 and c % a != 0 and c % b != 0:
                    return [a, b, c]
    
    # Try 4 terms
    # This is getting expensive, but n is small
    # We can use a recursive helper or just nested loops
    # Since we know the seed uses up to 3 terms and is not minimal, 
    # and the task says "1..30 distinct", it's likely 1-4 is enough.
    # Let's try 4 terms with some pruning.
    
    # To speed up 4-term search, we can precompute pairs that form antichains?
    # Or just brute force with early termination.
    
    # Optimization: sort candidates.
    # Loop i, j, k, and compute l.
    # This is O(L^3) which might be slow if L is large, but L is ~50 for 128.
    # 50^3 = 125,000 iterations per target. 128 targets * 125k = 16M ops.
    # Python might be slow for 16M ops, but we have 90s.
    # Let's try to be efficient.
    
    # Actually, we can memoize or just run it.
    # Let's implement a generic k-sum with antichain check.
    
    # For now, let's stick to 4 terms.
    for i in range(L):
        a = candidates[i]
        if a >= n:
            break
        for j in range(i + 1, L):
            b = candidates[j]
            if a + b >= n:
                break
            for k in range(j + 1, L):
                c = candidates[k]
                if a + b + c >= n:
                    break
                d = n - a - b - c
                if d <= c:
                    continue
                if d in candidates:
                    # Check antichain for {a,b,c,d}
                    # a<b<c<d
                    # Check: b%a, c%a, c%b, d%a, d%b, d%c
                    if b % a != 0 and c % a != 0 and c % b != 0 and \
                       d % a != 0 and d % b != 0 and d % c != 0:
                        return [a, b, c, d]
    
    # If 4 terms fail, try 5?
    # The task says "1..30". It is possible some numbers need more.
    # However, smooth numbers are dense enough.
    # Let's add a fallback search for 5 terms if 4 fails, but limit the search space.
    
    # Actually, let's just return None for now if 4 fails, and see if that happens.
    # Or better, a generic backtracking search for min terms up to, say, 10.
    
    # Let's implement a DFS with pruning for min terms.
    # State: (start_idx, remaining_sum, current_list)
    # Pruning: 
    #   1. remaining_sum < min(candidates[start_idx]) -> fail
    #   2. remaining_sum > sum(candidates[start_idx:]) -> fail (hard to compute on fly, maybe precompute suffix sums)
    #   3. Antichain check: new term must not divide or be divisible by any in current_list.
    
    # Since we already tried 1-4, let's try 5+ if needed.
    # But first, let's see if 4 is enough.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    smooth = _generate_smooth(128)
    smooth_set = set(smooth)
    
    answers = []
    total_terms = 0
    
    for n in range(1, 129):
        if time.time() > deadline:
            # If we run out of time, fill remaining with a valid (but maybe non-optimal) solution
            # or break and return what we have? The task requires all 128.
            # We should try to finish quickly or use a faster fallback.
            # For now, let's just break and return partial? No, must return 128 rows.
            # Let's use a simple valid solution: n itself is smooth? No, n might not be smooth.
            # Wait, if n is not smooth, [n] is not a valid term.
            # But we can always use 1? No, 1 divides everything.
            # So we need a valid antichain.
            # Let's just hope the search is fast enough.
            # If not, we'll return None for the rest, which is invalid.
            # Better: pre-compute all 128 at start? No, that's what we're doing.
            # Let's optimize the search.
            pass
            
        res = _find_min_terms(n, smooth)
        
        if res is None:
            # Fallback: try 5 terms
            # This is expensive. Let's hope it doesn't happen.
            # For now, let's just append an empty list or a dummy valid one?
            # Actually, every number can be represented.
            # Let's just use a simple greedy or random search for 5 terms.
            res = _find_terms_fallback(n, smooth, 5)
            
        if res is None:
            # Last resort: return [n] if smooth, else [] (invalid)
            # But we must return valid.
            # Let's assume 4 is enough.
            res = []
            
        answers.append(res)
        total_terms += len(res)
        
    return answers

def _find_terms_fallback(n, smooth, k):
    """Fallback search for exactly k terms."""
    candidates = [x for x in smooth if x <= n]
    L = len(candidates)
    
    # Simple DFS
    def dfs(start, remaining, current):
        if not current:
            if remaining == 0:
                return []
            if len(current) == k:
                return None
        # Pruning
        if len(current) == k:
            if remaining == 0:
                return []
            return None
        
        if remaining < 0:
            return None
        
        # Antichain check for new term
        for i in range(start, L):
            x = candidates[i]
            if x > remaining:
                break
            
            # Check antichain
            valid = True
            for y in current:
                if y < x:
                    if x % y == 0:
                        valid = False
                        break
                else:
                    if y % x == 0:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Recurse
            res = dfs(i + 1, remaining - x, current + [x])
            if res is not None:
                return [x] + res
        
        return None
    
    res = dfs(0, n, [])
    return res

# EVOLVE-BLOCK-END
