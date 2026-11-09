"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_reciprocal_triple(a, b, c):
    """Check if 1/a = 1/b + 1/c (i.e., c = ab/(a+b) with all distinct)."""
    # 1/a = 1/b + 1/c => c = ab/(a+b)
    # So for given a, b: c = ab/(a+b) must be integer, and c != a, c != b
    pass

def _find_conflicts(S):
    """Return set of elements involved in at least one forbidden triple within S."""
    S_set = set(S)
    conflicts = set()
    elems = sorted(S)
    for i, a in enumerate(elems):
        for j in range(i+1, len(elems)):
            b = elems[j]
            if a == b:
                continue
            # c = ab/(a+b)
            num = a * b
            den = a + b
            if num % den == 0:
                c = num // den
                if c in S_set and c != a and c != b:
                    conflicts.add(a)
                    conflicts.add(b)
                    conflicts.add(c)
    return conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s limit
    
    # Start with all elements
    S = set(range(1, 181))
    
    # Iteratively remove elements involved in conflicts
    # Use greedy: remove the element with highest conflict count
    best_S = S.copy()
    
    # Phase 1: Greedy removal
    while True:
        if time.time() > deadline - 5:
            break
        conflicts = _find_conflicts(S)
        if not conflicts:
            break
        
        # Count how many conflicts each element is in
        # For efficiency, count triples per element
        S_list = sorted(S)
        S_set = S
        conflict_count = {x: 0 for x in conflicts}
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                num = a * b
                den = a + b
                if num % den == 0:
                    c = num // den
                    if c in S_set and c != a and c != b:
                        conflict_count[a] = conflict_count.get(a, 0) + 1
                        conflict_count[b] = conflict_count.get(b, 0) + 1
                        conflict_count[c] = conflict_count.get(c, 0) + 1
        
        # Remove the element with highest conflict count
        if conflict_count:
            worst = max(conflict_count, key=conflict_count.get)
            S.discard(worst)
        
        if len(S) > len(best_S):
            best_S = S.copy()
    
    # Phase 2: Try to add back removed elements
    removed = set(range(1, 181)) - S
    # Try adding back in order of smallest element (less likely to create conflicts)
    for elem in sorted(removed):
        if time.time() > deadline - 5:
            break
        S_test = S | {elem}
        if not _find_conflicts(S_test):
            S = S_test
    
    if len(S) > len(best_S):
        best_S = S.copy()
    
    # Phase 3: Local search - try swapping
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        improved = False
        for a in sorted(best_S):
            if time.time() > deadline - 5:
                break
            for b in sorted(set(range(1,181)) - best_S):
                S_test = (best_S - {a}) | {b}
                if len(S_test) == len(best_S) and not _find_conflicts(S_test):
                    # Check if this swap is beneficial (it's neutral in size, 
                    # but might allow further additions)
                    # Only accept if b < a (prefer smaller elements)
                    if b < a:
                        best_S = S_test
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    
    # Final: try to add any remaining elements
    for elem in sorted(set(range(1, 181)) - best_S):
        if time.time() > deadline - 5:
            break
        S_test = best_S | {elem}
        if not _find_conflicts(S_test):
            best_S = S_test
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
