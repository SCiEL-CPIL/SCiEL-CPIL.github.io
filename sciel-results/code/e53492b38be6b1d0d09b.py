"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find the longest nonempty interval [start, start+length-1] contained in [1,1008]
    such that for every x, y in the interval (including x=y), x*y % 1009 != 1.
    
    P = 1009 is prime. Self-inverse residues mod P are solutions to x^2 = 1 mod P,
    which are x = 1 and x = P-1 = 1008.
    
    For distinct a != b, a*b = 1 mod P means b = a^{-1} mod P.
    
    An interval is valid iff:
      1. It contains neither 1 nor 1008.
      2. For no element x in the interval is inv[x] also in the interval (x != inv[x]).
    
    We use a two-pointer sliding window with group tracking.
    """
    P = 1009
    MAX = 1008  # residues 1..1008
    
    # Precompute modular inverses using Fermat's little theorem
    inv = np.zeros(P + 1, dtype=np.int64)
    for x in range(1, P):
        inv[x] = pow(x, P - 2, P)
    
    # Identify self-inverse residues (forbidden elements)
    # x^2 ≡ 1 mod P => x ≡ 1 or x ≡ P-1
    forbidden = set()
    for x in range(1, P):
        if x * x % P == 1:
            forbidden.add(x)
    # forbidden = {1, 1008}
    
    # Assign group IDs: each pair {a, inv[a]} with a != inv[a] gets a unique group.
    # Forbidden elements get group -1.
    group_id = np.full(P + 1, -2, dtype=np.int32)  # -2 = unassigned
    group_counter = 0
    assigned = np.zeros(P + 1, dtype=bool)
    
    for x in range(1, P):
        if x in forbidden:
            group_id[x] = -1
        elif not assigned[x]:
            group_counter += 1
            y = int(inv[x])
            group_id[x] = group_counter
            group_id[y] = group_counter
            assigned[x] = True
            assigned[y] = True
    
    # Now use two-pointer sliding window.
    # We want the longest window [left, right] within [1, 1008] such that:
    #   - No element in the window is forbidden.
    #   - No group appears more than once in the window.
    
    n = MAX
    freq = {}  # group_id -> count in current window
    left = 1
    best_len = 0
    best_s = 1
    
    for right in range(1, n + 1):
        g = int(group_id[right])
        if g == -1:
            # Forbidden element: reset window
            freq.clear()
            left = right + 1
            if left > n:
                break
            continue
        else:
            freq[g] = freq.get(g, 0) + 1
            # While the current group appears more than once, shrink from left
            while freq.get(g, 0) > 1:
                lg = int(group_id[left])
                if lg != -1:
                    freq[lg] -= 1
                    if freq[lg] == 0:
                        del freq[lg]
                left += 1
        
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_s = left
    
    # Verify the result
    if best_len > 0:
        s = best_s
        e = s + best_len - 1
        valid = True
        for x in range(s, e + 1):
            if x * x % P == 1:
                valid = False
                break
            y = int(inv[x])
            if s <= y <= e and y != x:
                valid = False
                break
        if valid:
            return [s, best_len]
    
    # Fallback: brute force with early termination
    deadline = time.time() + 50
    best_len = 0
    best_s = 1
    for s in range(1, n + 1):
        if int(group_id[s]) == -1:
            continue
        seen = set()
        for e in range(s, n + 1):
            g = int(group_id[e])
            if g == -1 or g in seen:
                break
            seen.add(g)
            if e - s + 1 > best_len:
                best_len = e - s + 1
                best_s = s
        if time.time() > deadline:
            break
    
    if best_len > 0:
        return [best_s, best_len]
    
    # Absolute fallback
    return [2, 30]
# EVOLVE-BLOCK-END
