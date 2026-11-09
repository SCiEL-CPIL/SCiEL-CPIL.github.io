import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 11
    num_colorings = 1 << n
    full_mask = (1 << num_colorings) - 1
    
    # Generate all edges
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute masks for each edge
    # mask[e] is a bitset of colorings where edge e is monochromatic
    edge_masks = [0] * num_edges
    for i, e in enumerate(all_edges):
        mask_vs = 0
        for v in e:
            mask_vs |= (1 << v)
        mask = 0
        for c in range(num_colorings):
            if (c & mask_vs) == 0 or (c & mask_vs) == mask_vs:
                mask |= (1 << c)
        edge_masks[i] = mask
    
    best_edges_count = num_edges
    best_chosen_indices = None
    
    # Phase 1: Deterministic greedy (fast, reliable)
    def deterministic_greedy():
        uncovered = full_mask
        chosen = []
        remaining = list(range(num_edges))
        while uncovered:
            if time.time() > deadline:
                return None
            best_idx = -1
            best_count = -1
            for idx in remaining:
                c = (uncovered & edge_masks[idx]).bit_count()
                if c > best_count:
                    best_count = c
                    best_idx = idx
            if best_idx == -1:
                break
            chosen.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            remaining.remove(best_idx)
        return chosen
    
    # Phase 2: Aggressive pruning
    def prune(chosen_indices):
        if not chosen_indices:
            return chosen_indices
        
        # Maintain counts of how many times each coloring is covered
        counts = [0] * num_colorings
        for idx in chosen_indices:
            m = edge_masks[idx]
            temp_m = m
            while temp_m:
                lsb = temp_m & -temp_m
                c_idx = lsb.bit_length() - 1
                counts[c_idx] += 1
                temp_m ^= lsb
        
        improved = True
        max_rounds = 100
        for _ in range(max_rounds):
            if not improved:
                break
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                idx = chosen_indices[i]
                m = edge_masks[idx]
                can_remove = True
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    c_idx = lsb.bit_length() - 1
                    if counts[c_idx] == 1:
                        can_remove = False
                        break
                    temp_m ^= lsb
                
                if can_remove:
                    temp_m = m
                    while temp_m:
                        lsb = temp_m & -temp_m
                        c_idx = lsb.bit_length() - 1
                        counts[c_idx] -= 1
                        temp_m ^= lsb
                    chosen_indices.pop(i)
                    improved = True
                    break
        
        return chosen_indices
    
    # First, try deterministic greedy
    result = deterministic_greedy()
    if result:
        pruned = prune(result)
        if len(pruned) < best_edges_count:
            best_edges_count = len(pruned)
            best_chosen_indices = pruned[:]
            if best_edges_count <= 51:
                return {'n': n, 'edges': [list(all_edges[i]) for i in best_chosen_indices]}
    
    # Phase 3: Multiple randomized attempts with different strategies
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Strategy varies by attempt number
        strategy = attempts % 3
        
        if strategy == 0:
            # Randomized greedy: shuffle then pick best
            indices = list(range(num_edges))
            random.shuffle(indices)
            uncovered = full_mask
            chosen = []
            remaining = set(indices)
            while uncovered:
                if time.time() > deadline:
                    break
                best_idx = -1
                best_count = -1
                for idx in remaining:
                    c = (uncovered & edge_masks[idx]).bit_count()
                    if c > best_count:
                        best_count = c
                        best_idx = idx
                if best_idx == -1:
                    break
                chosen.append(best_idx)
                uncovered &= ~edge_masks[best_idx]
                remaining.remove(best_idx)
        elif strategy == 1:
            # Randomized greedy: pick from top K
            indices = list(range(num_edges))
            random.shuffle(indices)
            uncovered = full_mask
            chosen = []
            remaining = set(indices)
            while uncovered:
                if time.time() > deadline:
                    break
                candidates = []
                for idx in remaining:
                    c = (uncovered & edge_masks[idx]).bit_count()
                    if c > 0:
                        candidates.append((c, idx))
                if not candidates:
                    break
                candidates.sort(reverse=True)
                top_k = min(3, len(candidates))
                _, best_idx = random.choice(candidates[:top_k])
                chosen.append(best_idx)
                uncovered &= ~edge_masks[best_idx]
                remaining.remove(best_idx)
        else:
            # Weighted random: prioritize edges covering more uncovered
            indices = list(range(num_edges))
            random.shuffle(indices)
            uncovered = full_mask
            chosen = []
            remaining = set(indices)
            while uncovered:
                if time.time() > deadline:
                    break
                # Compute weights
                weights = []
                for idx in remaining:
                    c = (uncovered & edge_masks[idx]).bit_count()
                    weights.append(max(c, 1))
                # Weighted random choice
                total_w = sum(weights)
                r = random.random() * total_w
                cum = 0
                best_idx = indices[0]
                for i, idx in enumerate(remaining):
                    cum += weights[i]
                    if r <= cum:
                        best_idx = idx
                        break
                chosen.append(best_idx)
                uncovered &= ~edge_masks[best_idx]
                remaining.remove(best_idx)
        
        if not chosen:
            continue
        
        pruned = prune(chosen)
        if len(pruned) < best_edges_count:
            best_edges_count = len(pruned)
            best_chosen_indices = pruned[:]
            if best_edges_count <= 51:
                break
    
    if best_chosen_indices is None:
        # Fallback: deterministic greedy
        result = deterministic_greedy()
        best_chosen_indices = result if result else []
    
    result_edges = [list(all_edges[i]) for i in best_chosen_indices]
    return {'n': n, 'edges': result_edges}
