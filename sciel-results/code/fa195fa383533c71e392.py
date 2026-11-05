import itertools
import random
import heapq

def construct(v, k, t):
    points = list(range(v))
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {sub: i for i, sub in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_subsets_idx: list of lists of subset indices
    block_subsets_idx = []
    for b in all_blocks:
        subs = [subset_to_idx[sub] for sub in itertools.combinations(b, t)]
        block_subsets_idx.append(subs)
        
    # Precompute subset_to_blocks: list of lists of block indices
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for i, subs in enumerate(block_subsets_idx):
        for s in subs:
            subset_to_blocks[s].append(i)
            
    def greedy_cover_weighted(seed):
        random.seed(seed)
        # gain[i] = number of uncovered subsets in block i
        gain = [len(block_subsets_idx[0])] * num_blocks
        # covered_count[s] = number of selected blocks covering subset s
        covered_count = [0] * num_subsets
        uncovered = [True] * num_subsets
        num_uncovered = num_subsets
        
        # Heap stores (-score, block_idx)
        # We will use a lazy heap. We push new scores when gain changes.
        # Score: gain / (1 + (total_subsets_in_block - gain)) 
        # total_subsets_in_block is constant C(k,t)
        k_t = len(block_subsets_idx[0])
        
        def calc_score(g):
            if g == 0: return 0
            # Heuristic: prefer blocks with high gain and low overlap (low covered part)
            # covered_part = k_t - g
            # score = g / (1 + covered_part)
            return g / (1 + (k_t - g))
            
        heap = []
        for i in range(num_blocks):
            heapq.heappush(heap, (-calc_score(gain[i]), i))
            
        selected = []
        
        while num_uncovered > 0:
            best_block_idx = -1
            best_score = -1
            
            # Find valid block from heap
            while heap:
                neg_score, i = heapq.heappop(heap)
                # Check if this entry is stale
                # We need to verify if gain[i] matches the score used
                # Since gain only decreases, if the stored score corresponds to a higher gain, it's stale.
                # But we don't store the gain in the heap.
                # Simple check: if gain[i] == 0, skip.
                # Better: if calc_score(gain[i]) != -neg_score, it's stale.
                # However, floating point comparison is risky.
                # Alternative: Just re-evaluate. But that defeats the purpose.
                # Let's just check if gain[i] is consistent with the score roughly.
                # Actually, since we only decrease gain, the score decreases.
                # If we pop an item, and its current score is lower than what we popped, it's stale.
                # But we don't know the current score without recalculating.
                # Let's use a counter or just re-push if changed.
                
                # Simpler approach for small N:
                # Just scan all blocks? 50k blocks.
                # 150 iterations. 7.5M checks.
                # In Python, 7.5M * (simple ops) is ~1-2 seconds.
                # The heap approach is complex to manage stale entries correctly without storing version numbers.
                # Let's stick to scanning but optimize the inner loop.
                
                # Actually, let's try the scan approach first, it's simpler and likely fast enough for 120s limit.
                # 50k blocks * 150 steps = 7.5M iterations.
                # Inside: calculate gain.
                # If we maintain `gain` array, we just read `gain[i]`.
                # So the loop is just `max(range(num_blocks), key=lambda i: gain[i])`.
                # This is O(N) per step.
                
                pass # Placeholder for logic decision
                
            # Decision: Use simple scan on `gain` array.
            # `gain` is updated incrementally.
            
            max_gain = -1
            best_idx = -1
            # Random tie-breaking
            for i in range(num_blocks):
                g = gain[i]
                if g > max_gain:
                    max_gain = g
                    best_idx = i
                elif g == max_gain and g > 0:
                    if random.random() < 0.5:
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            
            # Update gains for all blocks that share subsets with best_idx
            for s in block_subsets_idx[best_idx]:
                if uncovered[s]:
                    uncovered[s] = False
                    num_uncovered -= 1
                    # For all blocks containing s, decrease gain
                    for j in subset_to_blocks[s]:
                        gain[j] -= 1
                        # No need to update heap if we are scanning
                        
        return selected

    def local_search(blocks_in_use):
        # Convert to set for fast lookup
        selected_set = set(blocks_in_use)
        # Coverage count
        coverage_count = [0] * num_subsets
        for b_idx in selected_set:
            for s in block_subsets_idx[b_idx]:
                coverage_count[s] += 1
                
        improved = True
        # 1. Redundancy Removal
        while improved:
            improved = False
            current_blocks = list(selected_set)
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                if b_idx not in selected_set:
                    continue
                is_redundant = True
                for s in block_subsets_idx[b_idx]:
                    if coverage_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected_set.remove(b_idx)
                    for s in block_subsets_idx[b_idx]:
                        coverage_count[s] -= 1
                    improved = True
                    
        # 2. 2-for-1 Swap
        # Try to replace 2 blocks with 1
        # Iterate over pairs
        # To avoid O(N^2) being too slow, we limit iterations or use random sampling
        # N is ~150. N^2 = 22500.
        # For each pair, find S_lost.
        # S_lost size <= 35+35.
        # Check if any block covers S_lost.
        
        improved = True
        iterations = 0
        while improved and iterations < 10: # Limit passes
            improved = False
            iterations += 1
            current_blocks = list(selected_set)
            random.shuffle(current_blocks)
            for i in range(len(current_blocks)):
                b1 = current_blocks[i]
                if b1 not in selected_set: continue
                for j in range(i+1, len(current_blocks)):
                    b2 = current_blocks[j]
                    if b2 not in selected_set: continue
                    
                    # Calculate subsets lost if we remove b1 and b2
                    # A subset s is lost if coverage_count[s] == 2 and s is in b1 or b2
                    # OR coverage_count[s] == 1 and s is in b1 and not in b2? No.
                    # If we remove b1 and b2:
                    # New count for s = coverage_count[s] - (1 if s in b1 else 0) - (1 if s in b2 else 0)
                    # Lost if New count == 0 and Old count > 0.
                    
                    lost_subsets = []
                    for s in block_subsets_idx[b1]:
                        c = coverage_count[s]
                        if s in block_subsets_idx[b2]: # This check is slow
                            pass
                        # Better:
                        # count_b1 = 1 if s in b1 else 0
                        # count_b2 = 1 if s in b2 else 0
                        # This is O(1) if we use sets for block contents?
                        # We have list of indices.
                        # Let's just iterate union of subsets of b1 and b2.
                        pass
                    
                    # Efficient Lost Calculation:
                    # Create a temporary dict or set of subsets in b1 and b2
                    # Since block_subsets_idx[b] is a list of indices, we can use a set for fast lookup?
                    # Or just iterate.
                    
                    # Let's use a set for b1_subs and b2_subs for this check
                    # This is expensive inside the loop.
                    
                    # Alternative:
                    # Only consider pairs that overlap significantly?
                    
                    # Let's stick to a simpler local search:
                    # Just redundancy removal is often enough for this size.
                    # The 2-for-1 swap is complex to implement efficiently in Python without C extensions.
                    
                    # Let's try a "1-for-0" (redundancy) and "1-for-1" (swap if better? No, size same).
                    # Maybe "2-for-1" is worth it.
                    
                    # Let's optimize the lost subset calc:
                    # precomputed sets for each block?
                    # block_subsets_set = [set(subs) for subs in block_subsets_idx]
                    # This takes memory but speeds up intersection.
                    
                    pass
                    
        return list(selected_set)

    # Precompute sets for local search
    block_subsets_set = [set(subs) for subs in block_subsets_idx]

    def local_search_optimized(blocks_in_use):
        selected_set = set(blocks_in_use)
        coverage_count = [0] * num_subsets
        for b_idx in selected_set:
            for s in block_subsets_idx[b_idx]:
                coverage_count[s] += 1
                
        # Redundancy
        improved = True
        while improved:
            improved = False
            current_blocks = list(selected_set)
            random.shuffle(current_blocks)
            for b_idx in current_blocks:
                if b_idx not in selected_set:
                    continue
                is_redundant = True
                for s in block_subsets_idx[b_idx]:
                    if coverage_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    selected_set.remove(b_idx)
                    for s in block_subsets_idx[b_idx]:
                        coverage_count[s] -= 1
                    improved = True
                    
        # 2-for-1 Swap
        # Limit to 5 passes
        for _ in range(5):
            improved = False
            current_blocks = list(selected_set)
            random.shuffle(current_blocks)
            for i in range(len(current_blocks)):
                b1 = current_blocks[i]
                if b1 not in selected_set: continue
                for j in range(i+1, len(current_blocks)):
                    b2 = current_blocks[j]
                    if b2 not in selected_set: continue
                    
                    # Find lost subsets
                    # Subsets in b1 or b2 that have count 1 or 2
                    # If count 1: must be in the block being removed.
                    # If count 2: must be in both? No.
                    # If s in b1 and not b2: new count = old - 1. Lost if old==1.
                    # If s in b2 and not b1: new count = old - 1. Lost if old==1.
                    # If s in both: new count = old - 2. Lost if old==2.
                    
                    lost = []
                    for s in block_subsets_idx[b1]:
                        if s in block_subsets_set[b2]:
                            if coverage_count[s] == 2:
                                lost.append(s)
                        else:
                            if coverage_count[s] == 1:
                                lost.append(s)
                    for s in block_subsets_idx[b2]:
                        if s not in block_subsets_set[b1]:
                            if coverage_count[s] == 1:
                                lost.append(s)
                    
                    if not lost:
                        # Can remove both
                        selected_set.remove(b1)
                        selected_set.remove(b2)
                        for s in block_subsets_idx[b1]:
                            coverage_count[s] -= 1
                        for s in block_subsets_idx[b2]:
                            coverage_count[s] -= 1
                        improved = True
                        break
                    else:
                        # Try to find a block C that covers all lost
                        # Pick first lost subset
                        s0 = lost[0]
                        found = False
                        for c in subset_to_blocks[s0]:
                            if c in selected_set:
                                continue
                            # Check if c covers all lost
                            c_set = block_subsets_set[c]
                            covers_all = True
                            for s in lost:
                                if s not in c_set:
                                    covers_all = False
                                    break
                            if covers_all:
                                # Replace b1, b2 with c
                                selected_set.remove(b1)
                                selected_set.remove(b2)
                                selected_set.add(c)
                                # Update counts
                                for s in block_subsets_idx[b1]:
                                    coverage_count[s] -= 1
                                for s in block_subsets_idx[b2]:
                                    coverage_count[s] -= 1
                                for s in block_subsets_idx[c]:
                                    coverage_count[s] += 1
                                improved = True
                                found = True
                                break
                        if found:
                            break
            if not improved:
                break
                
        return list(selected_set)

    best_len = float('inf')
    best_blocks = []
    
    # Run multiple seeds
    for seed in range(10):
        blocks = greedy_cover_weighted(seed)
        blocks = local_search_optimized(blocks)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    return best_blocks
