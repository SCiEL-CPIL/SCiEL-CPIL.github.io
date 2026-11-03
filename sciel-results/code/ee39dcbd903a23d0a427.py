import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {sub: i for i, sub in enumerate(t_subsets)}
    num_t = len(t_subsets)
    full_mask = (1 << num_t) - 1
    
    k_subsets = list(itertools.combinations(range(v), k))
    
    block_masks = []
    for block in k_subsets:
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_index_map[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(k_subsets)
    
    # 2. Beam Search
    beam_width = 20
    candidate_width = 20
    
    beam = [(0, full_mask, [])]
    best_solution = None
    best_cost = float('inf')
    
    while time.time() < deadline:
        next_beam_candidates = {}
        
        for cost, mask, blocks in beam:
            if mask == 0:
                if cost < best_cost:
                    best_cost = cost
                    best_solution = blocks
                continue
            
            # Find top candidate_width blocks by gain
            gains = []
            # Optimization: local variable for speed
            bm = block_masks
            for i in range(num_blocks):
                g = (mask & bm[i]).bit_count()
                if g > 0:
                    gains.append((g, i))
            
            if not gains:
                continue
                
            gains.sort(key=lambda x: -x[0])
            
            # Take top candidate_width
            for g, idx in gains[:candidate_width]:
                new_mask = mask & ~bm[idx]
                new_cost = cost + 1
                new_blocks = blocks + [k_subsets[idx]]
                
                # Prune: keep best cost for each mask
                if new_mask not in next_beam_candidates or new_cost < next_beam_candidates[new_mask][0]:
                    next_beam_candidates[new_mask] = (new_cost, new_blocks)
        
        if not next_beam_candidates:
            break
            
        next_beam = []
        for mask, (cost, blocks) in next_beam_candidates.items():
            next_beam.append((cost, mask, blocks))
            
        next_beam.sort(key=lambda x: (x[0], x[1].bit_count()))
        
        beam = next_beam[:beam_width]
        
        if best_solution is not None:
            beam = [s for s in beam if s[0] < best_cost]
            if not beam:
                break

    if best_solution is None:
        # Fallback: Simple Greedy
        uncovered = full_mask
        blocks = []
        bm = block_masks
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(num_blocks):
                g = (uncovered & bm[i]).bit_count()
                if g > best_gain:
                    best_gain = g
                    best_idx = i
            if best_gain == 0:
                break
            blocks.append(k_subsets[best_idx])
            uncovered &= ~bm[best_idx]
        best_solution = blocks

    return best_solution
