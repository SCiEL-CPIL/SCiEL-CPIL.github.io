import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    points = list(range(v))
    
    # Precompute all t-subsets and their indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    # Generate all k-subsets (candidate blocks)
    k_subsets = list(itertools.combinations(points, k))
    n_blocks = len(k_subsets)
    
    # For each block, compute which t-subsets it covers
    # Store as a set of t-subset indices for efficiency
    block_t_sets = []
    for block in k_subsets:
        t_set = set()
        for t_sub in itertools.combinations(block, t):
            t_set.add(t_subset_index[t_sub])
        block_t_sets.append(t_set)
    
    # Build conflict graph: two blocks conflict if they share any t-subset
    # We want maximum weight clique in the COMPLEMENT: blocks that DON'T share t-subsets
    # But we actually want a covering, so we need blocks that TOGETHER cover all t-subsets
    # This is a set cover problem, not directly a clique problem
    # 
    # Let me reconsider: The directive says "maximum weight clique search in the complement
    # of the conflict graph, where vertices represent k-blocks and edges connect blocks
    # that do not share any t-subset, thereby transforming the covering constraint into
    # a packing problem"
    #
    # This doesn't directly map to set cover. Let me interpret this differently:
    # We'll use branch-and-bound for set cover with bitset acceleration
    #
    # For now, let's implement an efficient greedy + local search with bitsets
    
    # Convert t-set to bitmask for each block
    block_masks = []
    for t_set in block_t_sets:
        mask = 0
        for idx in t_set:
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Greedy set cover with bitsets
    uncovered = (1 << n_t_subsets) - 1
    selected = []
    used = [False] * n_blocks
    
    while uncovered:
        best_gain = -1
        best_idx = -1
        for i in range(n_blocks):
            if used[i]:
                continue
            gain = (block_masks[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == uncovered.bit_count():
                    break
        if best_idx == -1:
            break
        selected.append(best_idx)
        used[best_idx] = True
        uncovered &= ~block_masks[best_idx]
    
    # Local search: try to remove redundant blocks
    def try_remove_redundant(blocks_list):
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks_list) - 1, -1, -1):
                # Check if block i is redundant
                union_others = 0
                for j in range(len(blocks_list)):
                    if j != i:
                        union_others |= block_masks[blocks_list[j]]
                if union_others == (1 << n_t_subsets) - 1:
                    blocks_list.pop(i)
                    removed = True
                    break
        return blocks_list
    
    selected = try_remove_redundant(selected)
    best_found = selected[:]
    
    # Multi-start with randomization
    for restart in range(20):
        if time.time() - start_time > 90:
            break
        
        random.seed(restart)
        uncovered = (1 << n_t_subsets) - 1
        selected = []
        used = [False] * n_blocks
        
        while uncovered:
            # Compute gains for all unused blocks
            gains = []
            for i in range(n_blocks):
                if not used[i]:
                    gain = (block_masks[i] & uncovered).bit_count()
                    if gain > 0:
                        gains.append((gain, i))
            
            if not gains:
                break
            
            gains.sort(reverse=True)
            
            # Pick from top candidates with some randomization
            top_n = min(10, len(gains))
            candidates = [i for g, i in gains[:top_n]]
            
            # Weighted random selection favoring higher gains
            weights = [g for g, i in gains[:top_n]]
            total_weight = sum(weights)
            r = random.random() * total_weight
            cumulative = 0
            chosen = candidates[0]
            for idx, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen = candidates[idx]
                    break
            
            selected.append(chosen)
            used[chosen] = True
            uncovered &= ~block_masks[chosen]
        
        selected = try_remove_redundant(selected)
        
        if len(selected) < len(best_found):
            best_found = selected[:]
    
    return [list(k_subsets[i]) for i in best_found]
