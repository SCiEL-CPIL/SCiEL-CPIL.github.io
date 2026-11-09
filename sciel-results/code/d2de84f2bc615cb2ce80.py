"""Rewritten program: maximum independent set in the conflict graph of [1..256]."""
import time
import random
from itertools import combinations

# Deadline guard: 90s wall-clock, use 85s as safety margin
_DEADLINE = 85.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _conflict(a, b):
    """Return True if (a*b) % (a+b) == 0 (i.e., a and b conflict)."""
    return (a * b) % (a + b) == 0

def construct():
    global _START
    _START = time.time()
    
    N = 256
    # Build adjacency: conflict[a][b] = True means a and b cannot both be in the set
    # We work with 0-indexed: index i represents number i+1
    # conflict_mask[i] is a bitmask of nodes conflicting with node i
    
    # Precompute conflicts
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Degree of each node
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N)]
    
    # Greedy initial solution: sort by degree ascending, add if no conflict with chosen
    order = sorted(range(N), key=lambda x: degrees[x])
    chosen = set()
    for i in order:
        # Check if i conflicts with any chosen
        ok = True
        for c in chosen:
            if conflict_mask[i] & (1 << c):
                ok = False
                break
        if ok:
            chosen.add(i)
    
    best_set = set(chosen)
    best_size = len(best_set)
    
    # Local search: try to add more elements, then swap
    def can_add(s, node):
        for c in s:
            if conflict_mask[node] & (1 << c):
                return False
        return True
    
    def swap_improve(s):
        """Try to improve set s by removing one and adding one or two others."""
        improved = True
        while improved:
            improved = False
            # Try adding any unchosen node
            for node in range(N):
                if node not in s and can_add(s, node):
                    s.add(node)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one, add one or two
            for rem in list(s):
                s_without = s - {rem}
                # Try adding two nodes
                for a in range(N):
                    if a in s_without or a == rem:
                        continue
                    if not can_add(s_without, a):
                        continue
                    s_with_a = s_without | {a}
                    for b in range(N):
                        if b in s_with_a or b == rem or b == a:
                            continue
                        if not can_add(s_with_a, b):
                            continue
                        s_new = s_with_a | {b}
                        if len(s_new) > len(s):
                            s.clear()
                            s.update(s_new)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return s
    
    # Run local search with multiple random restarts
    best_size = len(best_set)
    
    for restart in range(500):
        if _time_left() < 5.0:
            break
        
        # Randomized greedy construction
        random.shuffle(order)
        chosen = set()
        for i in order:
            if can_add(chosen, i):
                chosen.add(i)
        
        # Local search improvement
        chosen = swap_improve(chosen)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
            # If we found a very large set, we can be more optimistic
            if best_size >= 150:
                break
    
    # Final polish: try to add any remaining compatible nodes
    for node in range(N):
        if node not in best_set and can_add(best_set, node):
            best_set.add(node)
    
    # Convert back to 1-indexed and sort
    result = sorted([i + 1 for i in best_set])
    
    return result

# Pre-compute at import time for speed? No, keep it in construct.
# But we can pre-compute the conflict structure once.
_conflict_mask_cache = None

def _get_conflict_mask():
    global _conflict_mask_cache
    if _conflict_mask_cache is None:
        N = 256
        cm = [0] * N
        for i in range(N):
            a = i + 1
            for j in range(i + 1, N):
                b = j + 1
                if (a * b) % (a + b) == 0:
                    cm[i] |= (1 << j)
                    cm[j] |= (1 << i)
        _conflict_mask_cache = cm
    return _conflict_mask_cache

def construct():
    global _START
    _START = time.time()
    
    N = 256
    conflict_mask = _get_conflict_mask()
    degrees = [bin(conflict_mask[i]).count('1') for i in range(N)]
    
    def can_add(s, node):
        """Check if node can be added to set s."""
        mask = conflict_mask[node]
        for c in s:
            if mask & (1 << c):
                return False
        return True
    
    def swap_improve(s):
        """Improve set by adding nodes and performing swaps."""
        improved = True
        while improved:
            improved = False
            # Try adding any unchosen node
            for node in range(N):
                if node not in s and can_add(s, node):
                    s.add(node)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-2 swap
            for rem in list(s):
                s_without = s - {rem}
                for a in range(N):
                    if a in s_without or a == rem:
                        continue
                    if not can_add(s_without, a):
                        continue
                    s_with_a = s_without | {a}
                    for b in range(N):
                        if b in s_with_a or b == rem or b == a:
                            continue
                        if not can_add(s_with_a, b):
                            continue
                        if len(s_with_a | {b}) > len(s):
                            s.clear()
                            s.update(s_with_a | {b})
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return s
    
    # Initial greedy: sort by degree ascending
    order_by_deg = sorted(range(N), key=lambda x: degrees[x])
    chosen = set()
    for i in order_by_deg:
        if can_add(chosen, i):
            chosen.add(i)
    chosen = swap_improve(chosen)
    
    best_set = set(chosen)
    best_size = len(best_set)
    
    # Randomized restarts
    rng = random.Random(42)
    max_restarts = 1000
    for restart in range(max_restarts):
        if _time_left() < 5.0:
            break
        
        # Randomized order
        order = list(range(N))
        rng.shuffle(order)
        chosen = set()
        for i in order:
            if can_add(chosen, i):
                chosen.add(i)
        
        # Local search
        chosen = swap_improve(chosen)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
            if best_size >= 160:
                break
    
    # Final: try to add any compatible nodes
    for node in range(N):
        if node not in best_set and can_add(best_set, node):
            best_set.add(node)
    
    result = sorted([i + 1 for i in best_set])
    return result
