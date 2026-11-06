import time
import random
from itertools import combinations

# Precompute popcounts for all 16-bit masks
N = 16
ALL_MASK = (1 << N) - 1
POP_COUNTS = [0] * (1 << N)
for i in range(1 << N):
    POP_COUNTS[i] = bin(i).count('1')

# Precompute all possible triples as bitmasks
ALL_TRIPLES = []
for a in range(N):
    for b in range(a + 1, N):
        for c in range(b + 1, N):
            mask = (1 << a) | (1 << b) | (1 << c)
            ALL_TRIPLES.append(mask)

def is_linear(existing_edges, new_edge):
    """Check if new_edge shares at most 1 point with all existing_edges."""
    for e in existing_edges:
        if (e & new_edge) != 0:
            if POP_COUNTS[e & new_edge] > 1:
                return False
    return True

def calc_alpha(edges_masks):
    """Calculate independence number of the hypergraph defined by edges_masks."""
    if not edges_masks:
        return N
    
    # Use a bytearray for speed
    bad = bytearray(1 << N)
    
    for e in edges_masks:
        comp = ALL_MASK ^ e
        sub = comp
        while True:
            bad[e | sub] = 1
            if sub == 0:
                break
            sub = (sub - 1) & comp
            
    max_alpha = 0
    for i in range(1 << N):
        if not bad[i]:
            pc = POP_COUNTS[i]
            if pc > max_alpha:
                max_alpha = pc
                if max_alpha == N:
                    break
    return max_alpha

def random_greedy_packing():
    """Generate a random maximal linear packing."""
    edges = []
    available = ALL_TRIPLES[:]
    random.shuffle(available)
    
    for t in available:
        if is_linear(edges, t):
            edges.append(t)
            
    return edges

def local_search(edges, time_limit):
    """Improve the independence number via local search."""
    current_alpha = calc_alpha(edges)
    if current_alpha == 4:
        return edges, 4
        
    best_alpha = current_alpha
    best_edges = edges[:]
    
    while time.time() < time_limit:
        improved = False
        
        # Try adding an edge first (cheaper)
        # Shuffle to randomize
        candidates = ALL_TRIPLES[:]
        random.shuffle(candidates)
        
        for t in candidates:
            if time.time() >= time_limit:
                break
            if is_linear(edges, t):
                new_edges = edges + [t]
                new_alpha = calc_alpha(new_edges)
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_edges = new_edges[:]
                    edges = new_edges[:]
                    improved = True
                    if best_alpha == 4:
                        return edges, 4
                    break
        
        if improved:
            continue
        
        # Try swapping one edge
        # Limit swaps to avoid excessive time
        for i in range(len(edges)):
            if time.time() >= time_limit:
                break
                
            e_to_remove = edges[i]
            remaining_edges = edges[:i] + edges[i+1:]
            
            # Try a subset of random replacements to save time
            # Instead of all 560, try 50 random ones
            replacements = ALL_TRIPLES[:]
            random.shuffle(replacements)
            
            for new_e in replacements[:50]:
                if new_e == e_to_remove:
                    continue
                if is_linear(remaining_edges, new_e):
                    candidate_edges = remaining_edges + [new_e]
                    new_alpha = calc_alpha(candidate_edges)
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_edges = candidate_edges
                        edges = candidate_edges[:]
                        improved = True
                        if best_alpha == 4:
                            return edges, 4
                        break
            if improved:
                break
        
        if not improved:
            # No single swap or add improved. 
            # Try a 2-swap? Too expensive.
            # Break if stuck.
            break
        
    return best_edges, best_alpha

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_edges = []
    best_alpha = N
    
    # Initial seed: try a few deterministic or semi-deterministic starts
    # Then random restarts
    
    # 1. Try a simple structured start
    # e.g. based on affine geometry or random
    # We'll just rely on random restarts as they are fast
    
    while time.time() < deadline:
        # Generate a random packing
        candidate_edges = random_greedy_packing()
        candidate_alpha = calc_alpha(candidate_edges)
        
        if candidate_alpha < best_alpha:
            best_alpha = candidate_alpha
            best_edges = candidate_edges[:]
            
        if best_alpha == 4:
            break
            
        # If we have time, try to improve the best found so far
        # Or the current candidate if it's close
        if time.time() < deadline - 2:
            # Improve the current candidate
            improved_edges, improved_alpha = local_search(candidate_edges, deadline - 1)
            if improved_alpha < best_alpha:
                best_alpha = improved_alpha
                best_edges = improved_edges[:]
                
            if best_alpha == 4:
                break

    # Convert best_masks back to list of triples [a, b, c]
    result = []
    for mask in best_edges:
        triple = []
        for i in range(N):
            if mask & (1 << i):
                triple.append(i)
        result.append(triple)
        
    return result
