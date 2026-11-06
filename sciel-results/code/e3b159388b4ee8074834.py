"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility relation
    # div[a] = set of b in smooth where b % a == 0 and b != a
    div = {x: set() for x in smooth}
    for a in smooth:
        for b in smooth:
            if b != a and b % a == 0:
                div[a].add(b)
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                a, b = terms[i], terms[j]
                if b % a == 0 or a % b == 0:
                    return False
        return True
    
    # For each target, find minimal term count representation
    answers = [None] * 128  # index 0 unused, 1..128
    
    # Precompute valid 1-term, 2-term, 3-term antichains
    # 1-term: just the number itself if smooth
    for n in range(1, 129):
        if n in smooth:
            answers[n-1] = [n]
    
    # 2-term solutions
    smooth_set = set(smooth)
    for n in range(1, 129):
        if answers[n-1] is not None:
            continue
        # Try pairs
        found = False
        for a in smooth:
            if a >= n:
                break
            b = n - a
            if b in smooth_set and b != a and not (b % a == 0 or a % b == 0):
                answers[n-1] = [a, b]
                found = True
                break
        if found:
            continue
    
    # 3-term solutions
    for n in range(1, 129):
        if answers[n-1] is not None:
            continue
        found = False
        # Enumerate triples
        for a in smooth:
            if a >= n:
                break
            for b in smooth:
                if b <= a:  # avoid duplicates, a < b
                    continue
                if a + b >= n:
                    break
                c = n - a - b
                if c in smooth_set and c > b and not (c % a == 0 or a % c == 0) and not (c % b == 0 or b % c == 0):
                    answers[n-1] = [a, b, c]
                    found = True
                    break
            if found:
                break
        if found:
            continue
    
    # For remaining targets, try 4+ terms using DFS/backtracking
    for n in range(1, 129):
        if answers[n-1] is not None:
            continue
        # Use backtracking to find any valid antichain summing to n
        smooth_sorted = sorted(smooth, key=lambda x: -x)  # larger first for pruning
        result = []
        
        def backtrack(remaining, start_idx, current_terms):
            if remaining == 0:
                if len(current_terms) >= 1:
                    return list(current_terms)
                return None
            if start_idx >= len(smooth_sorted):
                return None
            for i in range(start_idx, len(smooth_sorted)):
                x = smooth_sorted[i]
                if x > remaining:
                    continue
                # Check antichain constraint with existing terms
                valid = True
                for t in current_terms:
                    if x % t == 0 or t % x == 0:
                        valid = False
                        break
                if not valid:
                    continue
                current_terms.append(x)
                res = backtrack(remaining - x, i + 1, current_terms)
                if res is not None:
                    return res
                current_terms.pop()
            return None
        
        result = backtrack(n, 0, [])
        if result is not None:
            answers[n-1] = result
        else:
            # Fallback: try with all smooth numbers up to n
            smooth_le = [x for x in smooth if x <= n]
            smooth_le.sort(reverse=True)
            result2 = []
            
            def backtrack2(remaining, start_idx, current_terms):
                if remaining == 0:
                    if len(current_terms) >= 1:
                        return list(current_terms)
                    return None
                if start_idx >= len(smooth_le):
                    return None
                for i in range(start_idx, len(smooth_le)):
                    x = smooth_le[i]
                    if x > remaining:
                        continue
                    valid = True
                    for t in current_terms:
                        if x % t == 0 or t % x == 0:
                            valid = False
                            break
                    if not valid:
                        continue
                    current_terms.append(x)
                    res = backtrack2(remaining - x, i + 1, current_terms)
                    if res is not None:
                        return res
                    current_terms.pop()
                return None
            
            result2 = backtrack2(n, 0, [])
            if result2 is not None:
                answers[n-1] = result2
    
    # Verify all answers are valid
    for i, ans in enumerate(answers):
        if ans is None:
            # Should not happen, but handle gracefully
            answers[i] = [1]  # fallback
            continue
        n = i + 1
        assert sum(ans) == n, f"Sum mismatch for n={n}: {ans}"
        assert len(set(ans)) == len(ans), f"Duplicate terms for n={n}: {ans}"
        for x in ans:
            assert x in smooth_set, f"Non-smooth term {x} for n={n}"
        for a in ans:
            for b in ans:
                if a != b:
                    assert b % a != 0, f"Divisibility violation for n={n}: {a} divides {b}"
    
    return answers
# EVOLVE-BLOCK-END
