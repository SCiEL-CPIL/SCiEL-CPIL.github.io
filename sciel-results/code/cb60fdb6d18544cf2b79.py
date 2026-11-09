"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def _is_3ap_free(selected):
    """Check if a set has no 3-term arithmetic progression."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2*b - a
                if c != a and c != b and c in s:
                    return False
    return True

def _reciprocal_sum(selected):
    """Compute exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _can_add(selected, x):
    """Check if x can be added to selected without creating a 3-AP."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                # Check if adding x creates a 3-AP with a, b, x
                # x = 2b - a (x is the third term)
                if x == 2*b - a and x not in s:
                    return False
                # x = a + (b-a) = 2a - b (x is the first term)
                if x == 2*a - b and x not in s and x >= 1 and x <= 256:
                    return False
                # x = a + (x-a) where b = a + (x-a), so x = 2b - a already covered
                # Actually let's be more careful: check all triples involving x
    # More robust: check all pairs in selected with x
    for a in selected:
        # x, a, ? or ?, a, x or a, x, ?
        # If x and a are in set, check if midpoint or extension creates AP
        # 3-AP: p, q, r where q-p = r-q, so p+r = 2q
        # Case 1: x is middle: a, x, c => c = 2x - a
        c = 2*x - a
        if c != x and c != a and c in s:
            return False
        # Case 2: a is middle: x, a, c => c = 2a - x
        c = 2*a - x
        if c != x and c != a and c in s:
            return False
        # Case 3: c is middle: x, c, a => c = (x+a)/2, must be integer
        if (x + a) % 2 == 0:
            c = (x + a) // 2
            if c != x and c != a and c in s:
                return False
    return True

def construct():
    deadline = time.time() + 85.0  # 85s deadline, 5s safety margin
    
    # Greedy initialization: pick smallest numbers that don't create 3-AP
    selected = []
    for x in range(1, 257):
        if _can_add(selected, x):
            selected.append(x)
    
    best_selected = list(selected)
    best_sum = _reciprocal_sum(selected)
    
    # Local search: try to improve by swapping
    # Strategy: try to add smaller numbers by removing larger ones
    # Or try to remove a number and add a smaller one if it improves the sum
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(selected)
        
        # Try each pair: remove i, add j where j < i (j has higher reciprocal)
        for i in selected:
            if time.time() > deadline:
                break
            # Try to remove i and add some j not in set where j < i
            for j in range(1, i):
                if j in current_set:
                    continue
                # Check if removing i and adding j is valid
                new_selected = [x for x in selected if x != i]
                if _can_add(new_selected, j):
                    new_selected.append(j)
                    new_sum = _reciprocal_sum(new_selected)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_selected = new_selected
                        selected = new_selected
                        current_set = set(selected)
                        improved = True
                        break
        
        # Also try: remove a large element and add a smaller one that was previously blocked
        # Or try pairwise swaps: remove a, add b where b was blocked by a
        for a in selected:
            if time.time() > deadline:
                break
            for b in range(1, 257):
                if b in current_set or b >= a:
                    continue
                # Try replacing a with b
                new_selected = [x for x in selected if x != a]
                if _can_add(new_selected, b):
                    new_selected.append(b)
                    new_sum = _reciprocal_sum(new_selected)
                    if new_sum > best_sum:
                        best_sum = new_sum
                        best_selected = new_selected
                        selected = new_selected
                        current_set = set(selected)
                        improved = True
                        break
    
    return best_selected
# EVOLVE-BLOCK-END
