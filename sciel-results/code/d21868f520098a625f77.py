"""
Edge-colourings of K16 in which every K4 sees five colours.

Strategy:
1. Start from the known valid (i+j) mod 17 construction (17 colours).
2. Use simulated annealing / local search to reduce the number of colours.
   - Objective: minimise the number of distinct colours used, subject to
     the constraint that every K4 spans >= 5 colours.
   - We penalise constraint violations heavily and add a small cost per
     colour used to drive colour reduction.
3. Return the best valid colouring found before the deadline.
"""

import time
import random
import sys

DEADLINE = 115.0  # safety margin before 120s kill

def _edge_index(i, j):
    """Map pair (i,j), 0<=i<j<16, to index in the 120-element list."""
    # lexicographic: (0,1),(0,2),...,(0,15),(1,2),...,(14,15)
    # index = sum_{k=0}^{i-1} (15-k) + (j-1-i)
    #       = i*15 - i*(i-1)/2 + j - 1 - i
    #       = i*15 - i*(i-1)//2 + j - 1 - i
    return i * 15 - i * (i - 1) // 2 + j - 1 - i

def _precompute_k4s():
    """Return list of 6-tuples of edge indices for each K4."""
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    edges = [
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ]
                    k4s.append(tuple(edges))
    return k4s

_K4S = _precompute_k4s()

def _count_violations(colours):
    """Count number of K4s that span fewer than 5 distinct colours."""
    violations = 0
    for edges in _K4S:
        if len(set(colours[e] for e in edges)) < 5:
            violations += 1
    return violations

def _num_colours_used(colours):
    return len(set(colours))

def _objective(colours, weight=100):
    """Penalised objective: violations*weight + colours_used."""
    v = _count_violations(colours)
    if v > 0:
        return v * weight + _num_colours_used(colours)
    return _num_colours_used(colours)

def _initial_colouring():
    """(i+j) mod 17 construction."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _local_search(colours, max_iters=5000, deadline=None):
    """Hill-climbing with random restarts on single-edge colour changes."""
    best = list(colours)
    best_obj = _objective(best)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge and a random new colour
        idx = rng.randrange(120)
        old_c = best[idx]
        new_c = rng.randrange(17)  # search in 0..16
        if new_c == old_c:
            continue
        
        best[idx] = new_c
        obj_new = _objective(best)
        
        if obj_new < best_obj:
            best_obj = obj_new
            # Accept improvement
        else:
            # Revert
            best[idx] = old_c
    
    return best, best_obj

def _simulated_annealing(colours, deadline):
    """SA with single-edge moves to reduce colours while maintaining validity."""
    current = list(colours)
    current_obj = _objective(current)
    best = list(current)
    best_obj = current_obj
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    rng = random.Random(123)
    n_moves = 0
    
    while time.time() < deadline:
        n_moves += 1
        
        # Pick random edge and new colour
        idx = rng.randrange(120)
        old_c = current[idx]
        new_c = rng.randrange(17)
        if new_c == old_c:
            continue
        
        current[idx] = new_c
        new_obj = _objective(current)
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < min(1.0, T / max(1, current_obj)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best = list(current)
        else:
            current[idx] = old_c
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic restart if stuck
        if n_moves % 10000 == 0 and _count_violations(best) == 0:
            # Try to reduce colours by mapping a colour to another
            used = set(best)
            if len(used) > 5:
                # Try to merge a colour into another
                c1 = rng.choice(list(used))
                c2 = rng.choice([c for c in used if c != c1])
                trial = list(best)
                for i in range(120):
                    if trial[i] == c1:
                        trial[i] = c2
                if _count_violations(trial) == 0:
                    best = trial
                    best_obj = _num_colours_used(best)
                    current = list(best)
                    current_obj = best_obj
                    T = 50.0
    
    return best, best_obj

def construct():
    t0 = time.time()
    
    # Start from valid 17-colour construction
    colours = _initial_colouring()
    
    # Verify validity
    assert _count_violations(colours) == 0, "Initial construction should be valid"
    
    best = list(colours)
    best_obj = _num_colours_used(colours)
    best_colours = best_obj
    
    # Phase 1: Local search to try to reduce colours
    deadline = t0 + DEADLINE
    result, obj = _local_search(colours, max_iters=20000, deadline=deadline)
    if obj < best_colours and _count_violations(result) == 0:
        best = result
        best_colours = obj
    
    # Phase 2: Simulated annealing for more aggressive search
    if time.time() < deadline:
        result, obj = _simulated_annealing(best, deadline)
        if obj < best_colours and _count_violations(result) == 0:
            best = result
            best_colours = obj
    
    # Final: ensure validity
    if _count_violations(best) != 0:
        # Fall back to the known valid construction
        best = _initial_colouring()
    
    return best

# Pre-compute K4s at module load time for speed
_K4S = _precompute_k4s()
