import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Checks if the hypergraph is 2-colorable using backtracking with unit propagation.
    Uses bitmasks for speed.
    Returns True if there exists a 2-coloring with no monochromatic edge.
    """
    if not edges:
        return True
        
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
        
    # Stack for backtracking: (zeros_mask, ones_mask)
    stack = [(0, 0)]
    
    while stack:
        zeros, ones = stack.pop()
        unassigned = ((1 << n) - 1) & ~(zeros | ones)
        
        # Unit Propagation
        changed = True
        contradiction = False
        
        while changed:
            changed = False
            for mask in edge_masks:
                z_count = (mask & zeros).bit_count()
                o_count = (mask & ones).bit_count()
                
                if z_count == 5 or o_count == 5:
                    contradiction = True
                    break
                    
                if z_count == 4 and o_count == 0:
                    u_mask = mask & unassigned
                    if u_mask:
                        new_ones = ones | u_mask
                        if new_ones != ones:
                            ones = new_ones
                            unassigned = unassigned & ~u_mask
                            changed = True
                elif o_count == 4 and z_count == 0:
                    u_mask = mask & unassigned
                    if u_mask:
                        new_zeros = zeros | u_mask
                        if new_zeros != zeros:
                            zeros = new_zeros
                            unassigned = unassigned & ~u_mask
                            changed = True
        
        if contradiction:
            continue
            
        if unassigned == 0:
            return True
            
        # Branch on the first unassigned vertex
        # Find the lowest set bit in unassigned
        first_bit = unassigned & -unassigned
        
        # Branch 0
        stack.append((zeros | first_bit, ones))
        # Branch 1
        stack.append((zeros, ones | first_bit))
        
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    for n in range(10, 21):
        if time.time() > deadline:
            break
            
        for restart in range(5):
            if time.time() > deadline:
                break
                
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            k = int(len(all_edges) * 0.6)
            current_edges = set(all_edges[:k])
            
            while is_2_colorable_fast(n, [list(e) for e in current_edges]):
                if time.time() > deadline:
                    break
                added = False
                for e in all_edges:
                    if e not in current_edges:
                        current_edges.add(e)
                        added = True
                        break
                if not added:
                    break
            
            if is_2_colorable_fast(n, [list(e) for e in current_edges]):
                continue
                
            improved = True
            while improved and time.time() < deadline:
                improved = False
                edges_list = list(current_edges)
                random.shuffle(edges_list)
                
                for e in edges_list:
                    if time.time() > deadline:
                        break
                    if e not in current_edges:
                        continue
                    
                    test_edges = current_edges - {e}
                    
                    if not is_2_colorable_fast(n, [list(x) for x in test_edges]):
                        current_edges = test_edges
                        improved = True
                        break
                        
            if len(current_edges) < best_count:
                best_count = len(current_edges)
                best_edges = [list(e) for e in current_edges]
                best_n = n
                print(f"Found better: n={n}, edges={best_count}")

    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
