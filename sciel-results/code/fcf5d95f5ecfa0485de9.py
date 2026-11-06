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

def is_regular_subgraph(combo, adj_masks):
    """Check if the induced subgraph on combo is regular."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    
    d0 = (adj_masks[combo[0]] & mask).bit_count()
    for v in combo[1:]:
        if (adj_masks[v] & mask).bit_count() != d0:
            return False
    return True

def count_size6_violations(adj_masks, n, combos_6):
    """
    Counts size-6 violations using precomputed combinations.
    """
    count = 0
    for combo in combos_6:
        if is_regular_subgraph(combo, adj_masks):
            count += 1
    return count

def find_violation_full(adj_masks, n, combos_6):
    """
    Checks for any regular induced subgraph of size >= 6.
    Returns the subset or None.
    """
    # Check size 6
    for combo in combos_6:
        if is_regular_subgraph(combo, adj_masks):
            return combo
    
    # Check larger subsets (less likely to be found, but must check)
    # Optimization: If no size 6 regular subgraph exists, larger ones are rare 
    # but possible. We'll check up to n.
    for k in range(7, n + 1):
        for combo in itertools.combinations(range(n), k):
            if is_regular_subgraph(combo, adj_masks):
                return combo
    return None

def greedy_local_search(adj_masks, n, rng, deadline, combos_6, affected_sets_map):
    """
    Greedy local search minimizing size-6 violations.
    affected_sets_map: dict mapping (u, v) to list of combos_6 indices containing both u and v.
    """
    cur = adj_masks[:]
    cur_energy = count_size6_violations(cur, n, combos_6)
    
    if cur_energy == 0:
        return cur, 0

    best_cur = cur[:]
    best_energy = cur_energy
    
    # Pre-compute boolean array for current regular status of each 6-set
    is_reg = [False] * len(combos_6)
    for i, combo in enumerate(combos_6):
        is_reg[i] = is_regular_subgraph(combo, cur)
    
    no_improvement_count = 0
    max_no_improvement = 50
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Pick random edge
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u == v:
            continue
            
        # Flip edge
        cur[u] ^= (1 << v)
        cur[v] ^= (1 << u)
        
        # Update energy incrementally
        # Only 6-sets containing both u and v can change status
        delta = 0
        for idx in affected_sets_map.get((u, v), []):
            combo = combos_6[idx]
            was_reg = is_reg[idx]
            is_now_reg = is_regular_subgraph(combo, cur)
            
            if was_reg and not is_now_reg:
                delta -= 1
            elif not was_reg and is_now_reg:
                delta += 1
            
            is_reg[idx] = is_now_reg
        
        new_energy = cur_energy + delta
        
        if new_energy < cur_energy:
            cur_energy = new_energy
            no_improvement_count = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_cur = cur[:]
            if new_energy == 0:
                break
        else:
            # Revert
            cur[u] ^= (1 << v)
            cur[v] ^= (1 << u)
            # Restore is_reg for affected sets
            for idx in affected_sets_map.get((u, v), []):
                combo = combos_6[idx]
                is_reg[idx] = is_regular_subgraph(combo, cur)
            no_improvement_count += 1

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

    # Pre-compute combinations for max n=22
    # We will reuse subsets for smaller n? No, better to compute per n.
    
    # We want to maximize n. Try from 22 downwards.
    # Known: 20-vertex graphs exist. 21 is unknown/hard.
    # Strategy: Try 21 briefly, then focus on 20.
    
    for n in range(22, 5, -1):
        if time.time() > deadline:
            break
        
        # Prepare data structures for this n
        combos_6 = list(itertools.combinations(range(n), 6))
        
        # Build map from edge (u,v) to list of combo indices
        # This is expensive to build for n=22 (C(22,6)=74613), but done once per n.
        # Actually, for n=22, C(22,6) is large. Let's skip n=22 if time is tight.
        # C(20,6) = 38760. C(21,6) = 54264.
        
        if n > 21:
            # 22 is very hard. Just do a few random checks.
            for _ in range(5):
                if time.time() > deadline: break
                adj_masks = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if rng.random() < 0.5:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
                if not find_violation_full(adj_masks, n, combos_6):
                    if n > best_n:
                        best_n = n
                        best_edges = get_edges_from_masks(adj_masks, n)
                        best_valid = True
                    break
            continue

        # For n <= 21, build the affected sets map
        affected_sets_map = {}
        for i, combo in enumerate(combos_6):
            for u in range(6):
                for v in range(u + 1, 6):
                    edge = (combo[u], combo[v])
                    if edge not in affected_sets_map:
                        affected_sets_map[edge] = []
                    affected_sets_map[edge].append(i)
        
        # Try to find a valid graph
        max_attempts = 20 if n == 20 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break

            p = 0.5
            if n == 21:
                p = rng.choice([0.4, 0.5, 0.6])
            
            adj_masks = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < p:
                        adj_masks[u] |= (1 << v)
                        adj_masks[v] |= (1 << u)

            # Check if already valid
            viol = find_violation_full(adj_masks, n, combos_6)
            if viol is None:
                if n > best_n:
                    best_n = n
                    best_edges = get_edges_from_masks(adj_masks, n)
                    best_valid = True
                break

            # Run Greedy Local Search
            new_adj, energy = greedy_local_search(adj_masks, n, rng, deadline, combos_6, affected_sets_map)
            
            if energy == 0:
                # Check for larger violations
                viol = find_violation_full(new_adj, n, combos_6)
                if viol is None:
                    if n > best_n:
                        best_n = n
                        best_edges = get_edges_from_masks(new_adj, n)
                        best_valid = True
                    break

    if not best_valid or best_n < 6:
        # Fallback: Return a small valid graph
        # A graph on 5 vertices cannot have a regular induced subgraph on 6 vertices.
        # Any graph on 5 vertices is valid.
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}, edges={len(result['edges'])}")
