import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # We will try n from 9 to 15
    for n in range(9, 16):
        if time.time() > deadline:
            break
            
        # Precompute vertex masks
        # M0[v] is a bitmask of all colorings where vertex v is 0
        # M1[v] is a bitmask of all colorings where vertex v is 1
        total_colorings = 1 << n
        all_ones = (1 << total_colorings) - 1
        
        M0 = [0] * n
        M1 = [0] * n
        
        # Efficiently generate M0 and M1
        # For each vertex v, M1[v] has bits set where bit v of coloring is 1.
        # Coloring c is an integer 0..2^n-1.
        # Bit v of c is (c >> v) & 1.
        
        # We can construct these by iterating c?
        # Or use pattern generation.
        # For n=12, 4096 colorings.
        # Looping 4096 times and setting bits for n vertices is 4096*12 ~ 50k ops. Very fast.
        
        for c in range(total_colorings):
            for v in range(n):
                if (c >> v) & 1:
                    M1[v] |= (1 << c)
                else:
                    M0[v] |= (1 << c)
                    
        # Generate all edges
        all_edges = list(combinations(range(n), 5))
        
        # Precompute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
            
        # Now we have edge_masks, a list of ints.
        # We want to find a subset of indices such that OR of masks is all_ones.
        
        # Greedy Set Cover with restarts
        best_local_count = float('inf')
        best_local_edges = None
        
        # Number of restarts depends on n
        # For small n, we can do more restarts
        restarts = 50 if n <= 11 else 20
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected_indices = []
            remaining = list(range(len(all_edges)))
            random.shuffle(remaining)
            
            # Greedy
            while covered != all_ones:
                best_gain = -1
                best_idx = -1
                best_edge_idx = -1
                
                # To speed up, we can sample if remaining is large
                # But len(all_edges) is at most 3003 for n=14.
                # Iterating 3000 items is fast.
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        best_edge_idx = idx
                        if best_gain == 0:
                            # If no gain, we are stuck?
                            # This shouldn't happen if a cover exists.
                            # But if we picked bad edges earlier, we might not be able to cover all.
                            # Actually, if best_gain is 0 for all, we failed.
                            pass
                            
                if best_gain == 0:
                    break # Failed to cover all
                    
                selected_indices.append(best_edge_idx)
                covered |= edge_masks[best_edge_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones:
                # Refine: try to remove edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    # Try removing each edge in random order
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        # Check if e_idx is redundant
                        # Re-calculate covered without e_idx?
                        # Or just check if (covered & ~edge_masks[e_idx]) == all_ones? No.
                        # We need to know if the other edges cover all_ones.
                        # Let's just re-calculate OR of others.
                        # Since len(selected_indices) is small (< 60), this is fast.
                        
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j == i:
                                new_covered |= edge_masks[selected_indices[j]]
                                
                        if new_covered == all_ones:
                            selected_indices.pop(i)
                            covered = new_covered
                            improved = True
                            break # Restart refinement loop
                    
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                print(f"Found better: n={n}, edges={best_count}")
                
    if best_edges is None:
        # Fallback to a known valid construction if nothing found
        # e.g. n=9, all edges
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
