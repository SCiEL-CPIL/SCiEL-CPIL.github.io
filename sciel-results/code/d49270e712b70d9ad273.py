import time
import sys
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin before 150s limit
    
    n = 11
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Generate all edges
    edges = list(combinations(range(n), 5))
    num_edges = len(edges)
    
    # Precompute masks for each edge
    edge_masks = []
    for e in edges:
        mask_vs = 0
        for v in e:
            mask_vs |= (1 << v)
        
        mask = 0
        for c in range(num_colorings):
            if (c & mask_vs) == 0 or (c & mask_vs) == mask_vs:
                mask |= (1 << c)
        edge_masks.append(mask)
    
    best_edges = None
    best_count = num_edges + 1
    
    # Multiple restarts with different random seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Greedy Set Cover with random perturbation
        uncovered = full_mask
        chosen_indices = []
        remaining_indices = list(range(num_edges))
        random.shuffle(remaining_indices)
        
        # Greedy phase with random tie-breaking
        while uncovered and remaining_indices:
            if time.time() > deadline:
                break
            
            best_idx = -1
            best_count_cov = 0
            best_indices = []
            
            for idx in remaining_indices:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count_cov:
                    best_count_cov = c
                    best_indices = [idx]
                elif c == best_count_cov and c > 0:
                    best_indices.append(idx)
            
            if best_idx == -1 and best_indices:
                # Random tie-breaking among best candidates
                best_idx = random.choice(best_indices)
            
            if best_idx == -1:
                break
                
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining_indices.remove(best_idx)
        
        if uncovered != 0:
            continue  # This solution doesn't cover all colorings
        
        # Local search: try to remove edges
        counts = [0] * num_colorings
        for idx in chosen_indices:
            m = edge_masks[idx]
            for c in range(num_colorings):
                if (m >> c) & 1:
                    counts[c] += 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                idx = chosen_indices[i]
                m = edge_masks[idx]
                can_remove = True
                for c in range(num_colorings):
                    if (m >> c) & 1:
                        if counts[c] == 1:
                            can_remove = False
                            break
                if can_remove:
                    for c in range(num_colorings):
                        if (m >> c) & 1:
                            counts[c] -= 1
                    chosen_indices.pop(i)
                    improved = True
                    break
        
        count = len(chosen_indices)
        if count < best_count:
            best_count = count
            best_edges = [list(edges[i]) for i in chosen_indices]
            
            # If we've found the known optimal, we can stop early
            if best_count <= 51:
                break
    
    if best_edges is None:
        # Fallback: use a deterministic greedy
        uncovered = full_mask
        chosen_indices = []
        remaining_indices = list(range(num_edges))
        
        while uncovered and remaining_indices:
            best_idx = -1
            best_count_cov = 0
            for idx in remaining_indices:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count_cov:
                    best_count_cov = c
                    best_idx = idx
            
            if best_idx == -1:
                break
                
            chosen_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining_indices.remove(best_idx)
        
        best_edges = [list(edges[i]) for i in chosen_indices]
    
    return {'n': n, 'edges': best_edges}
