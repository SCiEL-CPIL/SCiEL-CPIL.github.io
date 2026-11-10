import time
import math
import random

N = 64

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Precompute coordinates
    xs = [i // N for i in range(N * N)]
    ys = [i % N for i in range(N * N)]
    
    all_points = list(range(N * N))
    
    def dist2(i, j):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return dx * dx + dy * dy
    
    selected = set()
    # dist_sets[q] = set of squared distances from q to other points in selected
    dist_sets = {}
    
    def can_add(p):
        if p in selected:
            return False
        seen_from_p = set()
        for q in selected:
            d = dist2(p, q)
            if d in seen_from_p:
                return False
            if d in dist_sets[q]:
                return False
            seen_from_p.add(d)
        return True

    def add(p):
        selected.add(p)
        dist_sets[p] = set()
        for q in selected:
            if q == p:
                continue
            d = dist2(p, q)
            dist_sets[p].add(d)
            dist_sets[q].add(d)

    def remove(p):
        selected.discard(p)
        if p not in dist_sets:
            return
        for q in selected:
            if q == p:
                continue
            d = dist2(p, q)
            if d in dist_sets[q]:
                dist_sets[q].discard(d)
        del dist_sets[p]
        
    def try_swap(p_out, p_in):
        """
        Attempts to swap p_out (in selected) with p_in (not in selected).
        Returns True if swap is valid and performed.
        """
        if p_in in selected or p_out not in selected:
            return False
            
        # Check validity of new set S' = S - {p_out} + {p_in}
        
        # 1. Check p_in's distances to all r in S' \ {p_in}
        seen_from_pin = set()
        for r in selected:
            if r == p_out:
                continue
            d = dist2(p_in, r)
            if d in seen_from_pin:
                return False
            seen_from_pin.add(d)
            
        # 2. Check for each r in S' \ {p_in}, if dist(p_in, r) conflicts
        for r in selected:
            if r == p_out:
                continue
            d_pr = dist2(p_in, r)
            d_pout_r = dist2(p_out, r)
            
            if d_pr in dist_sets[r]:
                # Conflict exists in current S.
                # It is resolved only if the conflicting point was p_out.
                if d_pout_r != d_pr:
                    return False
                # If d_pout_r == d_pr, the conflict was with p_out, which is removed.
                # So it's fine.
        
        # Swap is valid
        remove(p_out)
        add(p_in)
        return True

    # Greedy Initialization
    rng = random.Random(42)
    shuffled_points = all_points.copy()
    rng.shuffle(shuffled_points)
    
    for p in shuffled_points:
        if time.time() > deadline - 60:
            break
        if can_add(p):
            add(p)
            
    unselected = [p for p in all_points if p not in selected]
    
    # Local Search with Swaps
    while time.time() < deadline:
        if rng.random() < 0.3:
            # Try Add
            if not unselected:
                break
            p = rng.choice(unselected)
            if can_add(p):
                add(p)
                unselected.remove(p)
        else:
            # Try Swap
            if not selected or not unselected:
                continue
            p_out = rng.choice(list(selected))
            p_in = rng.choice(unselected)
            
            if try_swap(p_out, p_in):
                unselected.remove(p_in)
                unselected.append(p_out)
                
    result = [(i // N, i % N) for i in selected]
    return result

# EVOLVE-BLOCK-END
