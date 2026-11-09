# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _build_smooth_list(max_val=128):
    return [x for x in range(1, max_val + 1) if _is_smooth(x)]

def _is_antichain(terms):
    """Check that no term divides another."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _find_min_antichain(n, smooth, max_terms=30):
    """Find antichain representation of n with minimum number of terms."""
    candidates = [x for x in smooth if x <= n]
    
    # Try from 1 term up
    for count in range(1, min(max_terms, len(candidates)) + 1):
        # Use combinations
        for combo in combinations(candidates, count):
            if sum(combo) == n and _is_antichain(combo):
                return list(combo)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    smooth = _build_smooth_list(128)
    
    answers = [None] * 128
    
    # First pass: try to find minimal representations quickly
    # For small n, search is fast; for large n, we may need more time
    
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        # Try increasing number of terms
        found = False
        for count in range(1, 31):
            if time.time() > deadline:
                break
            
            candidates = [x for x in smooth if x <= n]
            if len(candidates) < count:
                break
            
            # For efficiency, if count is large, we can be smarter
            # But let's just iterate combinations for now
            # Optimization: if count == 1, just check if n is smooth
            if count == 1:
                if n in smooth:
                    answers[n - 1] = [n]
                    found = True
                    break
                continue
            
            # For count == 2:
            if count == 2:
                # Check all pairs
                cands = [x for x in smooth if x <= n // 2]
                for a in cands:
                    b = n - a
                    if b > a and b in smooth and b <= n and b % a != 0:
                        answers[n - 1] = [a, b]
                        found = True
                        break
                if found:
                    break
                continue
            
            # For count >= 3, use combinations but limit search time
            # We'll use a time-bounded search
            combo_start = time.time()
            for combo in combinations([x for x in smooth if x <= n], count):
                if time.time() > deadline:
                    break
                if sum(combo) == n and _is_antichain(combo):
                    answers[n - 1] = list(combo)
                    found = True
                    break
            if found:
                break
        
        # If we couldn't find a minimal one within time, try a greedy fallback
        if answers[n - 1] is None:
            # Greedy: pick largest smooth numbers that don't divide each other
            # This is a fallback that may not be minimal but is valid
            remaining = n
            terms = []
            available = sorted([x for x in smooth if x <= n], reverse=True)
            for x in available:
                if x > remaining:
                    continue
                # Check antichain constraint with existing terms
                ok = True
                for t in terms:
                    a, b = min(x, t), max(x, t)
                    if b % a == 0:
                        ok = False
                        break
                if ok:
                    terms.append(x)
                    remaining -= x
                    if remaining == 0:
                        break
            if remaining == 0 and len(terms) >= 1:
                answers[n - 1] = terms
            else:
                # Last resort: use single term if n is smooth, else try 2 terms
                if n in smooth:
                    answers[n - 1] = [n]
                else:
                    # Find any valid representation
                    for count in range(2, 31):
                        cands = [x for x in smooth if x <= n]
                        for combo in combinations(cands, count):
                            if sum(combo) == n and _is_antichain(combo):
                                answers[n - 1] = list(combo)
                                break
                        if answers[n - 1] is not None:
                            break
                if answers[n - 1] is None:
                    # Should not happen, but fill with something valid
                    answers[n - 1] = [n] if n in smooth else [1, n-1]

    # Ensure all 128 entries are filled
    for i in range(128):
        if answers[i] is None:
            n = i + 1
            if n in smooth:
                answers[i] = [n]
            else:
                # Find any valid representation
                for count in range(1, 31):
                    cands = [x for x in smooth if x <= n]
                    if len(cands) < count:
                        break
                    for combo in combinations(cands, count):
                        if sum(combo) == n and _is_antichain(combo):
                            answers[i] = list(combo)
                            break
                    if answers[i] is not None:
                        break
                if answers[i] is None:
                    answers[i] = [n]

    return answers
# EVOLVE-BLOCK-END
