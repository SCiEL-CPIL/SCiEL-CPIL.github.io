import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

# Precompute combinations for speed? 
# For n=21, C(21,6) = 54264. 
# Creating them on the fly is fine, but storing might help if we iterate multiple times.
# However, n changes. So we generate per n.

def find_violation_fast(adj_masks, n):
    """Return a violating vertex-subset (size >= 6) whose induced subgraph is regular, or None."""
    # Check size 6 first
    # Optimization: Use a list of combinations to avoid generator overhead in tight loop?
    # itertools.combinations is implemented in C, so it's fast.
    
    # We can speed up the inner check by unrolling or using early exit.
    
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        # Unroll mask creation? 
        # mask = (1<<c0) | (1<<c1) ...
        c = combo
        mask = (1 << c[0]) | (1 << c[1]) | (1 << c[2]) | (1 << c[3]) | (1 << c[4]) | (1 << c[5])
        
        d0 = (adj_masks[c[0]] & mask).bit_count()
        if (adj_masks[c[1]] & mask).bit_count() != d0: continue
        if (adj_masks[c[2]] & mask).bit_count() != d0: continue
        if (adj_masks[c[3]] & mask).bit_count() != d0: continue
        if (adj_masks[c[4]] & mask).bit_count() != d0: continue
        if (adj_masks[c[5]] & mask).bit_count() != d0: continue
        return combo

    # Check size 7
    for combo in itertools.combinations(range(n), 7):
        mask = 0
        for v in combo: mask |= (1 << v)
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        ok = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                ok = False
                break
        if ok: return combo
        
    # Check size 8+ (less likely, but must check)
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo: mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            ok = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    ok = False
                    break
            if ok: return combo
            
    return None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def local_search(adj_masks, n, rng, deadline):
    cur = adj_masks[:]
    
    # Initial check
    viol = find_violation_fast(cur, n)
    if viol is None:
        return n, get_edges_from_masks(cur, n), True

    max_iters = 500 # Increased
    it = 0
    stall_count = 0
    last_viol_count = -1 # Not tracking count, just existence.
    
    while it < max_iters:
        if time.time() > deadline:
            break

        viol = find_violation_fast(cur, n)
        if viol is None:
            return n, get_edges_from_masks(cur, n), True

        vset = set(viol)
        internal_edges = []
        internal_nonedges = []
        for u in vset:
            for v in vset:
                if u < v:
                    if cur[u] & (1 << v):
                        internal_edges.append((u, v))
                    else:
                        internal_nonedges.append((u, v))
        
        # Strategy: Flip an edge in the violating set.
        # To avoid getting stuck in a loop where we flip A->B->A, 
        # we can track recent flips.
        
        if internal_edges:
            u, v = rng.choice(internal_edges)
        elif internal_nonedges:
            u, v = rng.choice(internal_nonedges)
        else:
            # Should not happen for size 6
            u, v = rng.choice([(i, j) for i in range(n) for j in range(i+1, n)])

        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        it += 1
        
    viol = find_violation_fast(cur, n)
    return n, get_edges_from_masks(cur, n), viol is None

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN

    best_n = 0
    best_edges = []
    best_valid = False

    rng = random.Random(42)

    # Try n from 21 down to 6
    # We prioritize 21 and 20.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break

        if best_valid and best_n >= n:
            continue

        # More attempts for larger n
        if n >= 21:
            max_attempts = 50
        elif n >= 19:
            max_attempts = 30
        else:
            max_attempts = 20

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Random graph
            p = 0.5 # Regular subgraphs are common at 0.5, but random repair works well
            # Try varying p
            p = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break

            # Local search
            new_n, new_edges, new_valid = local_search(adj_masks, n, rng, deadline)
            if new_valid and new_n > best_n:
                best_n = new_n
                best_edges = new_edges
                best_valid = True
                break

    if not best_valid or best_n < 6:
        return {'n': 6, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
