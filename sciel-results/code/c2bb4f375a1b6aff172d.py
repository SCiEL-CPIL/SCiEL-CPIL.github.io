import time
import random
import itertools
import sys
import numpy as np

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

def get_edges_from_masks(adj_masks, n):
    """Convert adjacency masks to edge list."""
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def find_violation_fast(adj_masks, n):
    """
    Finds a subset S of size >= 6 such that the induced subgraph on S is regular.
    Returns the subset as a tuple of vertices, or None.
    Prioritizes smaller subsets (size 6) as they are more likely to be found quickly.
    """
    # Check subsets of size 6 first
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        # Check if all vertices in combo have the same degree within mask
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            return combo

    # Check larger subsets
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            is_regular = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            if is_regular:
                return combo
    return None

def count_size6_violations(adj_masks, n):
    """
    Efficiently counts size-6 violations.
    """
    count = 0
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            count += 1
    return count

def simulated_annealing(adj_masks, n, rng, deadline):
    """
    Minimizes the number of size-6 violations using Simulated Annealing.
    """
    cur = adj_masks[:]
    cur_energy = count_size6_violations(cur, n)
    
    if cur_energy == 0:
        return cur, 0

    T = 1.0
    cooling = 0.995
    min_T = 1e-4
    best_cur = cur[:]
    best_energy = cur_energy
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        # Pick a random edge to flip
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
            
        # Flip edge
        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        
        # Recalculate energy
        # For n=21, C(21,6) is 54264.
        # 54264 * (6 bit_counts + 5 comparisons) ~ 300k ops.
        # Python is slow, maybe 0.1s per eval?
        # If we can do 1000 steps in 100s, that's 0.1s/step.
        # This is tight but possible.
        
        new_energy = count_size6_violations(cur, n)
        
        if new_energy < cur_energy:
            cur_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_cur = cur[:]
        else:
            # Metropolis criterion
            delta_E = new_energy - cur_energy
            if delta_E == 0 or rng.random() < np.exp(-delta_E / T):
                cur_energy = new_energy
            else:
                # Reject move
                cur[u] ^= (1 << v)
                cur[v] ^= (1 << u)
                
        T *= cooling

    return best_cur, best_energy

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    best_n = 0
    best_edges = []
    best_valid = False

    # We want to maximize n. Try from 22 downwards.
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break

        # Number of attempts per n
        # Larger n is harder.
        max_attempts = 5 if n >= 21 else 10

        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            # Edge probability p
            # Try a mix of densities.
            p = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
            
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            # Check if already valid
            viol = find_violation_fast(adj_masks, n)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break

            # Run Simulated Annealing to minimize violations
            # SA optimizes for size 6 violations specifically, which are the most common and hard to remove.
            # If we reduce size 6 violations to 0, we must still check for larger ones.
            new_adj, energy = simulated_annealing(adj_masks, n, rng, deadline)
            
            if energy == 0:
                # Check for larger violations
                viol = find_violation_fast(new_adj, n)
                if viol is None:
                    if n > best_n:
                        best_n = n
                        best_edges = get_edges_from_masks(new_adj, n)
                        best_valid = True
                    break
            else:
                # Even if energy > 0, maybe we found a valid graph? Unlikely if size 6 violations exist.
                # But SA might have accidentally removed larger ones? No, we only minimize size 6.
                # If size 6 violations remain, graph is invalid.
                pass

    if not best_valid or best_n < 6:
        # Fallback: Return a small valid graph
        # A graph on 5 vertices cannot have a regular induced subgraph on 6 vertices.
        # Any graph on 5 vertices is valid.
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}, edges={len(result['edges'])}")
