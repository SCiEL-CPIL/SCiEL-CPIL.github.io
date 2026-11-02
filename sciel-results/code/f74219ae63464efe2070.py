import time
import random
import itertools

def construct(v, k, t):
    deadline = time.time() + 140.0
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_set = set(all_t_subsets)

    # Precompute index map for O(1) lookup
    t_sub_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    def block_t_subs(block):
        return [t_sub_idx[sub] for sub in itertools.combinations(block, t)]

    def greedy(seed):
        rng = random.Random(seed)
        # cover_counts[i] = number of blocks covering t-subset i
        cover = [0] * n_t
        uncovered_count = n_t
        blocks = []
        while uncovered_count > 0 and time.time() < deadline - 5.0:
            # Pick a random uncovered t-subset
            # Find one with cover==0
            # For speed, maintain a list of uncovered indices
            # But that's expensive to update; instead, pick randomly and check
            # Better: maintain a set of uncovered indices
            # Let's just pick a random point set and check
            # Actually, let's maintain a list of uncovered t-subset indices
            # This is getting complex. Let's use a simpler greedy:
            # Pick random t-subset, if covered skip, else build block around it.
            
            # Simple approach: pick random t-subset
            idx = rng.randint(0, n_t - 1)
            if cover[idx] > 0:
                continue  # already covered, try again (waste time but simple)
            
            start_sub = list(all_t_subsets[idx])
            block = list(start_sub)
            block_set = set(block)
            
            # Greedily extend
            while len(block) < k:
                best_gain = -1
                best_pt = -1
                # Compute current (t-1)-subsets of block
                t1_subs = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in t1_subs:
                        new_sub = tuple(sorted(sub + (x,)))
                        si = t_sub_idx.get(new_sub)
                        if si is not None and cover[si] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_pt = x
                if best_pt == -1 or best_gain == 0:
                    # Pick random
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_pt = rng.choice(candidates)
                block.append(best_pt)
                block_set.add(best_pt)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for si in block_t_subs(block):
                    if cover[si] == 0:
                        uncovered_count -= 1
                    cover[si] += 1
        
        return blocks, cover, uncovered_count

    def remove_redundant(blocks, cover):
        """Remove blocks that don't uniquely cover any t-subset."""
        changed = True
        cur_blocks = list(blocks)
        cur_cover = list(cover)
        while changed and time.time() < deadline - 2.0:
            changed = False
            for i in range(len(cur_blocks)):
                if time.time() >= deadline - 2.0:
                    break
                b = cur_blocks[i]
                redundant = True
                for si in block_t_subs(b):
                    if cur_cover[si] <= 1:
                        redundant = False
                        break
                if redundant:
                    for si in block_t_subs(b):
                        cur_cover[si] -= 1
                    cur_blocks.pop(i)
                    changed = True
                    break
        return cur_blocks, cur_cover

    def sim_anneal(blocks, cover, seed):
        """Simulated annealing with 1-1 and 2-2 block swaps."""
        rng = random.Random(seed)
        cur_blocks = list(blocks)
        cur_cover = list(cover)
        cur_len = len(cur_blocks)
        best_blocks = list(cur_blocks)
        best_len = cur_len
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        # Precompute block_t_subs for current blocks
        cur_bsubs = [block_t_subs(b) for b in cur_blocks]
        
        steps = 0
        while T > T_min and time.time() < deadline - 1.0:
            steps += 1
            if steps % 500 == 0:
                # Periodically try to remove redundant
                new_blocks, new_cover = remove_redundant(cur_blocks, cur_cover)
                if len(new_blocks) < best_len:
                    best_blocks = new_blocks
                    best_len = len(new_blocks)
                cur_blocks = new_blocks
                cur_cover = new_cover
                cur_len = len(cur_blocks)
                cur_bsubs = [block_t_subs(b) for b in cur_blocks]
            
            # Random swap
            if cur_len < 2:
                break
            
            swap_type = rng.random()
            if swap_type < 0.7:
                # 1-1 swap
                i, j = rng.sample(range(cur_len), 2)
                bi = cur_blocks[i]
                bj = cur_blocks[j]
                
                # Try random new blocks
                for _ in range(3):
                    new_bi = list(bi)
                    new_bj = list(bj)
                    # Randomly replace 1 point in each
                    for _ in range(2):
                        pos = rng.randint(0, k - 1)
                        new_pt = rng.randint(0, v - 1)
                        if new_pt not in new_bi:
                            new_bi[pos] = new_pt
                    for _ in range(2):
                        pos = rng.randint(0, k - 1)
                        new_pt = rng.randint(0, v - 1)
                        if new_pt not in new_bj:
                            new_bj[pos] = new_pt
                    
                    if len(set(new_bi)) < k or len(set(new_bj)) < k:
                        continue
                    new_bi.sort()
                    new_bj.sort()
                    
                    # Check if new blocks are distinct and not already present
                    if new_bi == bi or new_bj == bj or new_bi == new_bj:
                        continue
                    
                    # Compute cover delta
                    old_subs_i = set(block_t_subs(bi))
                    old_subs_j = set(block_t_subs(bj))
                    new_subs_i = set(block_t_subs(new_bi))
                    new_subs_j = set(block_t_subs(new_bj))
                    
                    # Check validity: all t-subsets still covered
                    # A t-subset is uncovered if it was only covered by bi or bj and not by new_bi or new_bj
                    # More precisely: for each t-subset, count how many of {bi, bj} cover it
                    # vs how many of {new_bi, new_bj} cover it
                    # If any t-subset loses coverage and cur_cover[si] <= loss, invalid
                    
                    # Efficient check:
                    # For si in old_subs_i ∪ old_subs_j:
                    #   old_count = (si in old_subs_i) + (si in old_subs_j)
                    #   new_count = (si in new_subs_i) + (si in new_subs_j)
                    #   if new_count < old_count and cur_cover[si] - old_count + new_count < 0: invalid
                    # But cur_cover[si] includes bi and bj, so:
                    #   effective_count = cur_cover[si] - old_count + new_count
                    #   if effective_count < 0: invalid
                    
                    valid = True
                    all_subs = old_subs_i | old_subs_j
                    for si in all_subs:
                        old_c = (1 if si in old_subs_i else 0) + (1 if si in old_subs_j else 0)
                        new_c = (1 if si in new_subs_i else 0) + (1 if si in new_subs_j else 0)
                        if cur_cover[si] - old_c + new_c < 0:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Compute cost delta (number of t-subsets with cover count becoming 0)
                    # Cost = number of t-subsets with cover count == 0 (should be 0 for valid)
                    # Actually, we want to minimize number of blocks, but maintain validity.
                    # Since we only accept valid swaps, cost delta is 0 for valid swaps.
                    # We accept all valid 1-1 swaps (they don't change block count).
                    # Wait, 1-1 swap doesn't change block count. So we need a different acceptance.
                    # Actually, for 1-1 swaps, we can accept if valid (no cost increase).
                    # For 2-2 swaps, we can reduce block count by 0 (still 2 blocks) or... 
                    # Hmm, 1-1 and 2-2 swaps don't change block count.
                    # To reduce block count, we need to remove blocks.
                    # So sim anneal should focus on finding configurations where blocks become redundant.
                    
                    # Let's just accept valid swaps with some probability to explore
                    if rng.random() < 0.5:
                        # Apply swap
                        cur_blocks[i] = new_bi
                        cur_blocks[j] = new_bj
                        # Update cover
                        for si in old_subs_i:
                            cur_cover[si] -= 1
                        for si in old_subs_j:
                            cur_cover[si] -= 1
                        for si in new_subs_i:
                            cur_cover[si] += 1
                        for si in new_subs_j:
                            cur_cover[si] += 1
                        cur_bsubs[i] = block_t_subs(new_bi)
                        cur_bsubs[j] = block_t_subs(new_bj)
                        break
            else:
                # 2-2 swap: remove 2 blocks, add 2 new blocks
                # This can potentially lead to redundancy removal
                i, j = rng.sample(range(cur_len), 2)
                bi = cur_blocks[i]
                bj = cur_blocks[j]
                
                for _ in range(3):
                    # Generate 2 new random blocks
                    new_bi = sorted(rng.sample(range(v), k))
                    new_bj = sorted(rng.sample(range(v), k))
                    
                    if new_bi == bi or new_bj == bj or new_bi == new_bj:
                        continue
                    # Check no duplicate with existing blocks
                    if any(new_bi == b for b in cur_blocks) or any(new_bj == b for b in cur_blocks):
                        continue
                    
                    old_subs_i = set(block_t_subs(bi))
                    old_subs_j = set(block_t_subs(bj))
                    new_subs_i = set(block_t_subs(new_bi))
                    new_subs_j = set(block_t_subs(new_bj))
                    
                    valid = True
                    all_subs = old_subs_i | old_subs_j
                    for si in all_subs:
                        old_c = (1 if si in old_subs_i else 0) + (1 if si in old_subs_j else 0)
                        new_c = (1 if si in new_subs_i else 0) + (1 if si in new_subs_j else 0)
                        if cur_cover[si] - old_c + new_c < 0:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Accept
                    cur_blocks[i] = new_bi
                    cur_blocks[j] = new_bj
                    for si in old_subs_i:
                        cur_cover[si] -= 1
                    for si in old_subs_j:
                        cur_cover[si] -= 1
                    for si in new_subs_i:
                        cur_cover[si] += 1
                    for si in new_subs_j:
                        cur_cover[si] += 1
                    cur_bsubs[i] = block_t_subs(new_bi)
                    cur_bsubs[j] = block_t_subs(new_bj)
                    break
            
            T *= cooling
        
        # Final redundancy removal
        final_blocks, final_cover = remove_redundant(cur_blocks, cur_cover)
        if len(final_blocks) < best_len:
            best_blocks = final_blocks
            best_len = len(final_blocks)
        
        return best_blocks, best_len

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy seeds
    for seed in range(200):
        if time.time() >= deadline - 10.0:
            break
        blocks, cover, unc = greedy(seed)
        if unc > 0:
            continue  # not a valid covering
        blocks, cover = remove_redundant(blocks, cover)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
    
    # If we have a valid covering, run sim anneal on the best
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Rebuild cover for best_blocks
        cover = [0] * n_t
        for b in best_blocks:
            for si in block_t_subs(b):
                cover[si] += 1
        # Run sim anneal with a few seeds
        for sa_seed in range(5):
            if time.time() >= deadline - 2.0:
                break
            ann_blocks, ann_len = sim_anneal(best_blocks, cover, sa_seed)
            if ann_len < best_count:
                best_count = ann_len
                best_blocks = ann_blocks
    
    # Fallback: pure greedy if no valid covering found
    if best_blocks is None:
        uncovered = set(all_t_subsets)
        best_blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    count = 0
                    for sub in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(sub + (x,)))
                        if new_sub in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        best_blocks, _ = remove_redundant(best_blocks, [0] * n_t)
    
    return best_blocks
