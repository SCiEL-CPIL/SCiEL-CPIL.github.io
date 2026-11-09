import time
import random
from itertools import combinations
import numpy as np

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n = 11
    num_colorings = 1 << n
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute edge masks as Python ints (bit i set means coloring i is monochromatic on this edge)
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
        
    full_mask = (1 << num_colorings) - 1
    best_edges = None
    best_len = float('inf')
    
    def try_reduce(selected_indices):
        """Try to reduce the set of edges by removal and swaps."""
        changed = True
        while changed and time_left() > 5:
            changed = False
            # Compute total coverage
            total_coverage = 0
            for idx in selected_indices:
                total_coverage |= edge_masks[idx]
            
            # Try removal
            for i in range(len(selected_indices)):
                if time_left() < 5:
                    break
                cov_without = 0
                for j in range(len(selected_indices)):
                    if i != j:
                        cov_without |= edge_masks[selected_indices[j]]
                if cov_without == total_coverage:
                    selected_indices.pop(i)
                    changed = True
                    break
            
            if changed:
                continue
                
            # Try swap: remove one edge, add another that covers what was lost
            for i in range(len(selected_indices)):
                if time_left() < 5:
                    break
                cov_without = 0
                for j in range(len(selected_indices)):
                    if i != j:
                        cov_without |= edge_masks[selected_indices[j]]
                lost = total_coverage & ~cov_without
                if lost == 0:
                    selected_indices.pop(i)
                    changed = True
                    break
                
                # Find an edge not in selected that covers 'lost'
                for k in range(num_edges):
                    if k in selected_indices:
                        continue
                    if (edge_masks[k] & lost) == lost:
                        selected_indices[i] = k
                        changed = True
                        break
                if changed:
                    break
        return selected_indices
    
    # 1. Fast Greedy with Randomization
    for trial in range(200):
        if time_left() < 15:
            break
        uncovered = full_mask
        selected_indices = []
        indices = list(range(num_edges))
        random.shuffle(indices)
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            for idx in indices:
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                elif gain == best_gain and gain > 0 and random.random() < 0.1:
                    best_idx = idx
                    
            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered:
            # Local Search: try to remove edges and do swaps
            selected_indices = try_reduce(selected_indices)
            
            if len(selected_indices) < best_len:
                best_len = len(selected_indices)
                best_edges = [list(all_edges[idx]) for idx in selected_indices]
                if best_len <= 51:
                    break

    # 2. Try more aggressive local search from best found
    if best_edges is not None and time_left() > 10:
        for extra_trial in range(50):
            if time_left() < 5:
                break
            # Start from best and perturb
            perturbed = list(range(len(best_edges)))
            # Randomly swap a few edges
            for _ in range(3):
                i = random.randint(0, len(perturbed) - 1)
                k = random.randint(0, num_edges - 1)
                if k not in [all_edges[j] for j in perturbed]:
                    perturbed[i] = k
            
            # Verify coverage
            cov = 0
            for idx in perturbed:
                cov |= edge_masks[idx]
            if cov != full_mask:
                continue
            
            # Reduce
            perturbed = try_reduce(perturbed)
            if len(perturbed) < best_len:
                best_len = len(perturbed)
                best_edges = [list(all_edges[idx]) for idx in perturbed]
                if best_len <= 51:
                    break

    if best_edges is None:
        # Fallback: greedy without randomization
        uncovered = full_mask
        selected_indices = []
        indices = list(range(num_edges))
        while uncovered:
            best_gain = -1
            best_idx = -1
            for idx in indices:
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx == -1: break
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
        selected_indices = try_reduce(selected_indices)
        return {'n': n, 'edges': [list(all_edges[idx]) for idx in selected_indices]}

    return {'n': n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
