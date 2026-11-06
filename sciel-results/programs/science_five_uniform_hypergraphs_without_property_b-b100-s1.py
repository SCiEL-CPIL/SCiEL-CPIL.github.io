import time
import random
from itertools import combinations

# Precompute all possible 5-combinations for n up to 20
# This is used for fast random edge generation
MAX_N = 20
ALL_COMBS = []
for n in range(5, MAX_N + 1):
    combs = list(combinations(range(n), 5))
    ALL_COMBS.append(combs)

def check_non_2_colorable_fast(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable.
    Uses bitmask representation for speed.
    edges: list of tuples of 5 vertices.
    """
    if not edges:
        return False
    
    # Map vertices to indices 0..n-1
    # We represent the coloring as a bitmask of n bits.
    # For each edge, we need to check if all 5 bits are the same.
    
    # Optimization: Precompute the bitmasks for each edge?
    # An edge (v0, v1, v2, v3, v4) is monochromatic in mask M if:
    # (M & mask_edge) == 0 or (M & mask_edge) == mask_edge
    # where mask_edge has bits set for v0..v4.
    
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # If n is small (<= 15), brute force is extremely fast
    if n <= 15:
        for mask in range(1 << n):
            valid = True
            for em in edge_masks:
                # Check if edge is monochromatic
                # If it is, this coloring is invalid for "non-2-colorable" check
                # We are looking for a coloring where NO edge is monochromatic.
                # So if we find one where ALL edges are non-monochromatic, it IS 2-colorable.
                # Here we check if this mask makes the edge monochromatic.
                # If any edge is monochromatic, this mask is NOT a valid 2-coloring.
                # Wait, the definition of 2-colorable is: exists a coloring such that NO edge is monochromatic.
                # So we iterate masks. If we find a mask where for ALL edges, the edge is NOT monochromatic, 
                # then it is 2-colorable.
                # If we exhaust all masks and didn't find such a mask, it is NOT 2-colorable.
                
                # Let's rephrase:
                # Is there a mask M such that for all e in edges, M does not make e monochromatic?
                
                # Monochromatic check for edge e with mask M:
                # Let S be the set of vertices in e.
                # M restricts to S. If all bits in S are 0 or all are 1, it's mono.
                # val = M & em
                # if val == 0 or val == em: mono.
                
                if (mask & em) == 0 or (mask & em) == em:
                    valid = False
                    break
            if valid:
                # Found a valid 2-coloring
                return False 
        return True # No valid 2-coloring found

    else:
        # Backtracking with unit propagation for n > 15
        # This is a simplified version for robustness, but local search usually keeps n small (8-12)
        # For n > 15, we might need a more robust solver, but let's assume local search stays in small n.
        # If n gets large, we might fail. Let's force n to be <= 15 in the search if possible.
        # However, the problem allows up to 26. 
        # For now, if n > 15, we use a slower backtracking.
        
        # Simple Backtracking
        vertex_edges = [[] for _ in range(n)]
        for i, e in enumerate(edges):
            for v in e:
                vertex_edges[v].append(i)
        
        colors = [-1] * n
        edge_counts = [[0, 0] for _ in edges]
        
        def backtrack(idx):
            # Find forced move or pick vertex
            target_v = -1
            target_c = -1
            forced = False
            
            for i in range(len(edges)):
                c0, c1 = edge_counts[i]
                if c0 == 5 or c1 == 5:
                    return False
                if c0 == 4 and c1 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 1
                            forced = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in edges[i]:
                        if colors[v] == -1:
                            target_v = v
                            target_c = 0
                            forced = True
                            break
                if forced:
                    break
            
            if not forced:
                # Pick first unassigned
                for v in range(n):
                    if colors[v] == -1:
                        target_v = v
                        break
                if target_v == -1:
                    return True
            
            # Apply
            colors[target_v] = target_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if target_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            if forced:
                return False
            
            # Try other color
            other_c = 1 - target_c
            colors[target_v] = other_c
            saved = []
            for i in vertex_edges[target_v]:
                saved.append((edge_counts[i][0], edge_counts[i][1]))
                if other_c == 0:
                    edge_counts[i][0] += 1
                else:
                    edge_counts[i][1] += 1
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            colors[target_v] = -1
            for i, (c0, c1) in zip(vertex_edges[target_v], saved):
                edge_counts[i] = [c0, c1]
            
            return False
        
        return not backtrack(0)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_edges = None
    best_n = 0
    best_m = float('inf')
    
    # Initial candidates: K_n^5 for small n
    # K_9^5 has 126 edges. K_10^5 has 252 edges.
    # We want to start with a non-2-colorable graph.
    # K_9^5 is definitely non-2-colorable (it's the complete 5-graph on 9 vertices).
    
    # Let's start with n=9, all edges
    n_init = 9
    init_edges = [list(e) for e in combinations(range(n_init), 5)]
    
    # Verify initial state
    if check_non_2_colorable_fast(n_init, init_edges):
        best_edges = init_edges
        best_n = n_init
        best_m = len(init_edges)
    else:
        # Fallback: try n=10
        n_init = 10
        init_edges = [list(e) for e in combinations(range(n_init), 5)]
        if check_non_2_colorable_fast(n_init, init_edges):
            best_edges = init_edges
            best_n = n_init
            best_m = len(init_edges)
        else:
            # Should not happen, but return empty if stuck
            return {'n': 0, 'edges': []}

    current_n = best_n
    current_edges = best_edges[:]
    current_m = len(current_edges)
    
    # Local Search Parameters
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    
    # Precompute all possible edges for current_n to allow adding/swapping
    all_possible_edges = set(tuple(e) for e in combinations(range(current_n), 5))
    
    while time.time() < deadline and temp > min_temp:
        # Generate a neighbor
        # Moves:
        # 1. Remove an edge (if it keeps non-2-colorable)
        # 2. Add an edge (if it was 2-colorable, making it non-2-colorable) - usually not needed if we start non-2-colorable
        # 3. Swap an edge: Remove one, Add another.
        
        # Strategy: Try to remove edges. If stuck, try swapping to "move" the obstruction.
        
        # Pick a random edge to remove
        if current_m > 0:
            idx = random.randint(0, current_m - 1)
            e_to_remove = current_edges[idx]
            new_edges = current_edges[:idx] + current_edges[idx+1:]
            
            # Check if new_edges is still non-2-colorable
            if check_non_2_colorable_fast(current_n, new_edges):
                # Accept
                current_edges = new_edges
                current_m -= 1
                if current_m < best_m:
                    best_m = current_m
                    best_edges = current_edges[:]
                    best_n = current_n
                # Don't cool down on improvement? Or cool down slowly.
                # Standard SA: accept improvement always.
            else:
                # Rejection: calculate probability
                # Energy difference: we want to minimize m.
                # Removing edge fails -> we stay at current_m.
                # We might want to try adding an edge to escape local minima?
                # If we can't remove, maybe we should swap.
                
                # Try a swap: Remove e_to_remove, Add a random other edge not in current_edges
                # This changes the structure but keeps m constant.
                # Then we can try to remove again.
                
                # Pick a random edge to add
                # We need an edge not in current_edges
                current_set = set(tuple(e) for e in current_edges)
                available_adds = [e for e in all_possible_edges if e not in current_set]
                
                if available_adds:
                    e_to_add = random.choice(available_adds)
                    # Swap
                    new_edges_swap = [e for i, e in enumerate(current_edges) if i != idx]
                    new_edges_swap.append(e_to_add)
                    
                    # Check if swapped graph is non-2-colorable
                    if check_non_2_colorable_fast(current_n, new_edges_swap):
                        # Accept swap (m unchanged, but structure changed)
                        # This is a "neutral" move, always accept to explore
                        current_edges = new_edges_swap
                        # Don't update best_m as it's the same
                    else:
                        # Try adding an edge (increase m) to escape?
                        # Only if we are stuck for a while. 
                        # For now, just reject.
                        # Calculate acceptance probability for "worse" state?
                        # Here "worse" means we failed to remove. We didn't change state.
                        # So no state change.
                        pass
        
        # Occasionally try to change n?
        # If n is too small, we might be stuck. If n is too large, m is large.
        # Let's stick to fixed n for the inner loop, but restart with different n if stuck.
        
        # Simple stagnation check: if no improvement in last X steps, change n?
        # For simplicity, we just run SA on n=9 and n=10 separately if time permits.
        
        temp *= cooling
        
        # If we've explored n=9 thoroughly, try n=10
        # Let's add a mechanism to switch n if we hit a local optimum.
        # Actually, let's just run separate SA loops for n=9, 10, 11 if time allows.
        
        # To keep it simple in this single loop:
        # If current_m is close to best_m for n, and we've cooled down, restart with new n?
        # Let's implement a multi-start approach.
        
        # Break out of inner loop if temp is low
        if temp <= min_temp:
            break

    # If time remains, try other n values
    for n_try in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
        if n_try == current_n:
            continue
            
        # Initialize with complete graph
        init_edges_try = [list(e) for e in combinations(range(n_try), 5)]
        if not check_non_2_colorable_fast(n_try, init_edges_try):
            continue # Shouldn't happen for complete graphs, but safe
            
        cur_n = n_try
        cur_edges = init_edges_try[:]
        cur_m = len(cur_edges)
        
        # Run a shorter SA for this n
        t_local = 1.0
        cool_local = 0.999
        min_t_local = 0.01
        
        all_possible_try = set(tuple(e) for e in combinations(range(n_try), 5))
        
        while time.time() < deadline - 5 and t_local > min_t_local:
            if cur_m > 0:
                idx = random.randint(0, cur_m - 1)
                e_to_remove = cur_edges[idx]
                new_edges = cur_edges[:idx] + cur_edges[idx+1:]
                
                if check_non_2_colorable_fast(cur_n, new_edges):
                    cur_edges = new_edges
                    cur_m -= 1
                    if cur_m < best_m:
                        best_m = cur_m
                        best_edges = cur_edges[:]
                        best_n = cur_n
                else:
                    # Try swap
                    current_set = set(tuple(e) for e in cur_edges)
                    available_adds = [e for e in all_possible_try if e not in current_set]
                    if available_adds:
                        e_to_add = random.choice(available_adds)
                        new_edges_swap = [e for i, e in enumerate(cur_edges) if i != idx]
                        new_edges_swap.append(e_to_add)
                        if check_non_2_colorable_fast(cur_n, new_edges_swap):
                            cur_edges = new_edges_swap
                            
            t_local *= cool_local

    # Final verification
    if best_edges is not None:
        # Ensure the final result is valid
        if check_non_2_colorable_fast(best_n, best_edges):
            return {'n': best_n, 'edges': best_edges}
    
    # Fallback to initial if something went wrong
    return {'n': best_n, 'edges': best_edges if best_edges else []}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
    # Verify
    print("Verifying...")
    assert not check_non_2_colorable_fast(res['n'], res['edges']), "Result is 2-colorable!"
    print("Valid.")
