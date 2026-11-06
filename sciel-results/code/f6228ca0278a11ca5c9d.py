import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    best_result = None
    best_count = float('inf')
    
    # Search n from 9 to 15
    for n in range(9, 16):
        if time.time() > deadline - 20:
            break
            
        edges = list(combinations(range(n), 5))
        
        # Precompute masks
        # mask[e] is a bitmask of colorings forbidden by edge e
        masks = {}
        for e in edges:
            mask = 0
            # Optimization: 
            # We can compute the mask by iterating over the 2^(n-5) free bits and the 2 monochromatic states.
            # But for n <= 15, iterating over all 2^n colorings is fine.
            # 2^15 = 32768.
            # 3003 edges * 32768 = 100M operations. Might be too slow in Python.
            
            # Let's optimize mask calculation.
            # An edge e forbids colorings where bits at positions in e are all 0 or all 1.
            # Let fixed_bits = e
            # free_bits = [i for i in range(n) if i not in e]
            
            # For each assignment of free bits, there are 2 colorings (all 0 or all 1 on e).
            # So we can iterate over 2^(n-5) values.
            
            free_bits = [i for i in range(n) if i not in e]
            num_free = len(free_bits)
            
            for free_val in range(1 << num_free):
                # Construct the two colorings
                # Coloring 1: e is all 0
                c1 = 0
                for i, bit in enumerate(free_bits):
                    if (free_val >> i) & 1:
                        c1 |= (1 << bit)
                # e is 0, so no bits set for e
                
                # Coloring 2: e is all 1
                c2 = c1
                for v in e:
                    c2 |= (1 << v)
                    
                mask |= (1 << c1) | (1 << c2)
                
            masks[e] = mask
            
        if time.time() > deadline - 10:
            continue
            
        all_colorings_mask = (1 << (1 << n)) - 1
        
        # Greedy Set Cover
        uncovered = all_colorings_mask
        selected_edges = []
        
        # We need to efficiently find the edge with max gain.
        # Gain = popcount(mask[e] & uncovered)
        
        while uncovered:
            best_edge = None
            best_gain = 0
            best_mask = 0
            
            for e in edges:
                if e in [x for x in selected_edges]: # Skip selected? No, we can re-select? No, set cover doesn't repeat.
                    continue
                # Actually, we just iterate over all edges and pick the best.
                # We don't need to skip selected ones explicitly if we remove them from consideration or just check gain.
                # If an edge is already selected, its gain on *uncovered* might be 0 if it was fully covered by previous selections?
                # No, previous selections covered *some* of it. The *uncovered* part of a previously selected edge might be 0?
                # Yes, if we select e, we remove its mask from uncovered. So any subsequent selection of e would have gain 0.
                # So we can just iterate over all edges.
                
                gain_mask = masks[e] & uncovered
                gain = bin(gain_mask).count('1')
                
                if gain > best_gain:
                    best_gain = gain
                    best_edge = e
                    best_mask = masks[e]
                    
            if best_gain == 0:
                break # Should not happen
            
            selected_edges.append(best_edge)
            uncovered &= ~best_mask
            
            if time.time() > deadline - 10:
                break
                
        if not uncovered:
            # Local Search to minimize
            improved = True
            while improved:
                improved = False
                for i in range(len(selected_edges)):
                    # Try removing selected_edges[i]
                    # Check if remaining edges cover all
                    # We can compute the union of masks of remaining edges
                    # To speed up, we can maintain the current cover mask?
                    # Or just recompute.
                    
                    remaining = selected_edges[:i] + selected_edges[i+1:]
                    cover_mask = 0
                    for e in remaining:
                        cover_mask |= masks[e]
                        
                    if cover_mask == all_colorings_mask:
                        # We can remove it
                        selected_edges = remaining
                        improved = True
                        break # Restart loop as list changed
                    
            if len(selected_edges) < best_count:
                best_count = len(selected_edges)
                best_result = {'n': n, 'edges': [list(e) for e in selected_edges]}
                
    if best_result is not None:
        return best_result
        
    # Fallback
    return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
