"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _has_3ap(a, b, c):
    """Check if a, b, c form a nonconstant 3-term AP."""
    return a + c == 2 * b and a != c

def _construct_by_removal():
    """Start with all elements, remove to break 3-APs, preferring removal of large elements."""
    n = 256
    S = set(range(1, n + 1))
    
    # Find all 3-APs and iteratively remove the largest element
    # Use a priority approach: always remove the element with largest x from any existing 3-AP
    changed = True
    while changed:
        changed = False
        # Find all 3-APs in current S
        # For efficiency, for each pair (a, c) with a < c, check if b = (a+c)//2 is in S
        # and a+c is even
        to_remove = set()
        for a in sorted(S):
            for c in sorted(S):
                if a >= c:
                    continue
                s = a + c
                if s % 2 == 0:
                    b = s // 2
                    if b in S and a < b < c:
                        # Found a 3-AP: remove the largest element
                        to_remove.add(max(a, b, c))
        if to_remove:
            # Remove all identified elements (or just one? Let's remove all to be safe)
            # Actually, removing one at a time might be better to avoid over-removal
            # But for speed, let's remove all at once
            S -= to_remove
            changed = True
    
    return S

def _score(S):
    """Compute exact sum of reciprocals as a Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total

def _local_search(S, time_limit):
    """Try to improve the set by adding elements or swapping."""
    start_time = time.time()
    n = 256
    current_score = _score(S)
    S = set(S)
    
    # Try to add elements not in S
    while time.time() - start_time < time_limit:
        improved = False
        candidates = [x for x in range(1, n + 1) if x not in S]
        # Sort by decreasing 1/x (i.e., increasing x) to try high-value additions first
        candidates.sort()
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            # Check if adding x creates a 3-AP
            valid = True
            for a in S:
                for b in S:
                    if a == b or x == a or x == b:
                        continue
                    # Check if x, a, b form a 3-AP in any order
                    triple = sorted([x, a, b])
                    if triple[0] + triple[2] == 2 * triple[1]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.add(x)
                new_score = _score(S)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    S.discard(x)
        
        if not improved:
            break
    
    return S, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, leaving 10s margin
    
    # Phase 1: Construct initial set by removal
    S = _construct_by_removal()
    
    if time.time() > deadline:
        return sorted(S)
    
    # Phase 2: Local search to improve
    S, score = _local_search(S, deadline - time.time() - 5)
    
    return sorted(S)
# EVOLVE-BLOCK-END
