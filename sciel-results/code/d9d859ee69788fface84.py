"""Simulated annealing + smart greedy for covering design C(19,7,4)."""
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Build coverage matrix: cov_matrix[block_idx, tsub_idx] = 1 if block covers tsub
    # For C(19,7,4): n_blocks = C(19,7) = 50388, n_tsubsets = C(19,4) = 3876
    # Matrix is 50388 x 3876 = ~195M entries. Use sparse representation instead.
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs)
    
    # For each t-subset, list of block indices covering it
    tsub_blocks = [[] for _ in range(n_tsubsets)]
    for bi, tsubs in enumerate(block_tsubs):
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)
    
    # Convert to numpy arrays for speed
    block_tsubs_np = [np.array(t, dtype=np.int32) for t in block_tsubs]
    tsub_blocks_np = [np.array(b, dtype=np.int32) for b in tsub_blocks]

    def greedy_cover(seed=0, use_synergy=True):
        """Greedy with optional pairwise synergy consideration."""
        rng = random.Random(seed)
        uncovered_count = np.ones(n_tsubsets, dtype=np.int32)  # 1 = uncovered
        blocks_used = []
        used_block_set = set()
        
        # Precompute gain for all blocks
        gain = np.array([sum(uncovered_count[ts] for ts in block_tsubs[bi]) for bi in range(n_blocks)], dtype=np.int32)
        
        # For synergy: track how many t-subsets each block shares with already-used blocks
        # This is expensive, so only use for first few blocks or skip
        
        while uncovered_count.sum() > 0:
            if time.time() > deadline:
                break
            
            max_gain = gain.max()
            if max_gain == 0:
                break
            
            max_blocks = np.where(gain == max_gain)[0]
            # Filter out already used
            candidates = [b for b in max_blocks if b not in used_block_set]
            if not candidates:
                # All max gain blocks are used, pick next best
                sorted_blocks = np.argsort(-gain)
                candidates = [b for b in sorted_blocks if b not in used_block_set]
                if not candidates:
                    break
            
            chosen = candidates[rng.randint(0, len(candidates) - 1)]
            blocks_used.append(all_blocks[chosen])
            used_block_set.add(chosen)
            
            # Update uncovered_count and gains
            chosen_tsubs = block_tsubs_np[chosen]
            newly_covered = chosen_tsubs[uncovered_count[chosen_tsubs] > 0]
            
            if len(newly_covered) > 0:
                uncovered_count[newly_covered] = 0
                # Decrement gain for all blocks covering each newly covered t-subset
                for ts_idx in newly_covered:
                    for b in tsub_blocks_np[ts_idx]:
                        gain[b] -= 1
            
            gain[chosen] = -1  # mark as used
        
        return blocks_used

    def reduce_blocks(blocks, seed=0):
        """Remove redundant blocks via greedy set cover reduction."""
        if not blocks:
            return blocks
        
        rng = random.Random(seed)
        n = len(blocks)
        
        # Build coverage counts
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsub_indices = []
        for b in blocks:
            tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsub_indices.append(tsubs)
            for ts in tsubs:
                cover_count[ts] += 1
        
        # Repeatedly remove redundant blocks
        kept = list(range(n))
        changed = True
        while changed:
            changed = False
            # Try to remove blocks in random order
            order = list(range(len(kept)))
            rng.shuffle(order)
            for pos in order:
                if time.time() > deadline - 5:
                    return [blocks[kept[i]] for i in kept]
                bi = kept[pos]
                redundant = True
                for ts in block_tsub_indices[bi]:
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    for ts in block_tsub_indices[bi]:
                        cover_count[ts] -= 1
                    kept.pop(pos)
                    changed = True
        
        return [blocks[i] for i in kept]

    def simulated_annealing(initial_blocks, seed=0, max_iters=50000):
        """Simulated annealing with 1-block and 2-block moves."""
        rng = random.Random(seed)
        
        # Start from reduced set
        current_blocks = reduce_blocks(initial_blocks, seed)
        n_cur = len(current_blocks)
        
        # Build coverage structure
        def build_coverage(blocks):
            cover_count = np.zeros(n_tsubsets, dtype=np.int32)
            block_tsubs_list = []
            for b in blocks:
                tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
                block_tsubs_list.append(tsubs)
                for ts in tsubs:
                    cover_count[ts] += 1
            return cover_count, block_tsubs_list
        
        cover_count, block_tsubs_list = build_coverage(current_blocks)
        current_cost = n_cur  # minimize number of blocks
        best_blocks = list(current_blocks)
        best_cost = n_cur
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        alpha = (T_end / T_start) ** (1.0 / max(1, max_iters - 1))
        T = T_start
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            # Choose move type: 1-block (add/remove/swap) or 2-block exchange
            move_type = rng.random()
            
            if move_type < 0.3:
                # 1-block swap: remove one, add one
                if n_cur == 0:
                    # Add a random block
                    bi = rng.randint(0, n_blocks - 1)
                    new_blocks = [all_blocks[bi]]
                    new_cc, new_btl = build_coverage(new_blocks)
                    new_cost = len(new_blocks)
                    delta = new_cost - current_cost
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_blocks = new_blocks
                        n_cur = len(current_blocks)
                        cover_count, block_tsubs_list = new_cc, new_btl
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                    T *= alpha
                    continue
                
                # Remove block i, add block j
                i = rng.randint(0, n_cur - 1)
                j = rng.randint(0, n_blocks - 1)
                
                # Check if adding j covers all t-subsets uniquely covered by i
                unique_tsubs_i = [ts for ts in block_tsubs_list[i] if cover_count[ts] == 1]
                if unique_tsubs_i:
                    # j must cover all unique t-subsets of i
                    j_tsubs_set = set(block_tsubs_np[j].tolist())
                    if not all(ts in j_tsubs_set for ts in unique_tsubs_i):
                        T *= alpha
                        continue
                
                # Perform swap
                new_blocks = list(current_blocks)
                new_blocks[i] = all_blocks[j]
                
                # Check validity: all t-subsets still covered
                # Quick check: for each t-subset, count coverage in new set
                # This is O(n_cur * C(7,4)) which is expensive, so use incremental
                # Remove i's contribution, add j's contribution
                new_cc = cover_count.copy()
                for ts in block_tsubs_list[i]:
                    new_cc[ts] -= 1
                for ts in block_tsubs_np[j]:
                    new_cc[ts] += 1
                
                if (new_cc >= 1).all():
                    new_cost = len(new_blocks)
                    delta = new_cost - current_cost
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_blocks = new_blocks
                        cover_count = new_cc
                        block_tsubs_list[i] = block_tsubs_np[j].tolist()
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                
                T *= alpha
                continue
            
            elif move_type < 0.6:
                # 2-block exchange: remove i, add j and k
                if n_cur == 0:
                    T *= alpha
                    continue
                
                i = rng.randint(0, n_cur - 1)
                j = rng.randint(0, n_blocks - 1)
                k = rng.randint(0, n_blocks - 1)
                if j == k:
                    T *= alpha
                    continue
                
                # Check coverage
                unique_tsubs_i = [ts for ts in block_tsubs_list[i] if cover_count[ts] == 1]
                if unique_tsubs_i:
                    jk_tsubs_set = set(block_tsubs_np[j].tolist()) | set(block_tsubs_np[k].tolist())
                    if not all(ts in jk_tsubs_set for ts in unique_tsubs_i):
                        T *= alpha
                        continue
                
                new_blocks = list(current_blocks)
                new_blocks[i] = all_blocks[j]
                new_blocks.append(all_blocks[k])
                
                new_cc = cover_count.copy()
                for ts in block_tsubs_list[i]:
                    new_cc[ts] -= 1
                for ts in block_tsubs_np[j]:
                    new_cc[ts] += 1
                for ts in block_tsubs_np[k]:
                    new_cc[ts] += 1
                
                if (new_cc >= 1).all():
                    new_cost = len(new_blocks)
                    delta = new_cost - current_cost
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_blocks = new_blocks
                        n_cur = len(current_blocks)
                        cover_count = new_cc
                        block_tsubs_list[i] = block_tsubs_np[j].tolist()
                        block_tsubs_list.append(block_tsubs_np[k].tolist())
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                
                T *= alpha
                continue
            
            else:
                # 2-for-2 exchange: remove i, m and add j, k
                if n_cur < 2:
                    T *= alpha
                    continue
                
                i = rng.randint(0, n_cur - 1)
                m = rng.randint(0, n_cur - 1)
                if i == m:
                    T *= alpha
                    continue
                j = rng.randint(0, n_blocks - 1)
                k = rng.randint(0, n_blocks - 1)
                if j == k:
                    T *= alpha
                    continue
                
                # Check coverage
                new_cc = cover_count.copy()
                for ts in block_tsubs_list[i]:
                    new_cc[ts] -= 1
                for ts in block_tsubs_list[m]:
                    new_cc[ts] -= 1
                for ts in block_tsubs_np[j]:
                    new_cc[ts] += 1
                for ts in block_tsubs_np[k]:
                    new_cc[ts] += 1
                
                if (new_cc >= 1).all():
                    new_blocks = list(current_blocks)
                    # Remove i and m (careful with indices)
                    if i < m:
                        new_blocks.pop(m)
                        new_blocks.pop(i)
                    else:
                        new_blocks.pop(i)
                        new_blocks.pop(m)
                    new_blocks.append(all_blocks[j])
                    new_blocks.append(all_blocks[k])
                    
                    new_cost = len(new_blocks)
                    delta = new_cost - current_cost
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_blocks = new_blocks
                        n_cur = len(current_blocks)
                        cover_count = new_cc
                        # Rebuild block_tsubs_list
                        block_tsubs_list = [block_tsubs_np[all_blocks.index(b)] if b in all_blocks else [] for b in current_blocks]
                        # Actually rebuild properly
                        block_tsubs_list = []
                        for b in current_blocks:
                            idx = all_blocks.index(b)
                            block_tsubs_list.append(block_tsubs_np[idx].tolist())
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                
                T *= alpha
                continue
        
        # Final reduction
        best_blocks = reduce_blocks(best_blocks, seed)
        return best_blocks

    def two_block_local_search(blocks, seed=0):
        """Dedicated 2-block exchange local search (hill climbing)."""
        rng = random.Random(seed)
        blocks = reduce_blocks(blocks, seed)
        n = len(blocks)
        
        if n == 0:
            return blocks
        
        cover_count, block_tsubs_list = build_coverage(blocks)
        improved = True
        max_rounds = 50
        
        for rnd in range(max_rounds):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # Try all pairs of blocks to remove, find best 2-block replacement
            # This is too expensive for large n, so sample
            n_trials = min(200, n * 5)
            for trial in range(n_trials):
                if time.time() > deadline - 3:
                    break
                
                # Pick two blocks to potentially remove
                i = rng.randint(0, n - 1)
                m = rng.randint(0, n - 1)
                if i == m:
                    continue
                
                # Find t-subsets that would become uncovered
                temp_cc = cover_count.copy()
                for ts in block_tsubs_list[i]:
                    temp_cc[ts] -= 1
                for ts in block_tsubs_list[m]:
                    temp_cc[ts] -= 1
                
                uncovered_after = np.where(temp_cc == 0)[0]
                if len(uncovered_after) == 0:
                    # Both blocks are redundant, remove one
                    for ts in block_tsubs_list[i]:
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    block_tsubs_list.pop(i)
                    n -= 1
                    improved = True
                    break
                
                # Need to cover uncovered_after with 2 new blocks
                # Find blocks that cover the most uncovered t-subsets
                # For each candidate block, count how many uncovered t-subsets it covers
                # Sample candidates
                n_cands = min(100, n_blocks)
                cands = rng.sample(range(n_blocks), n_cands)
                
                best_pair = None
                best_cover = -1
                
                for c1 in cands:
                    c1_tsubs = set(block_tsubs_np[c1].tolist())
                    c1_cover = sum(1 for ts in uncovered_after if ts in c1_tsubs)
                    if c1_cover < len(uncovered_after) * 0.5:
                        continue
                    for c2 in cands:
                        if c2 <= c1:
                            continue
                        c2_tsubs = set(block_tsubs_np[c2].tolist())
                        # Check if c1 and c2 together cover all uncovered
                        remaining = [ts for ts in uncovered_after if ts not in c1_tsubs]
                        if all(ts in c2_tsubs for ts in remaining):
                            # Valid pair
                            total_cover = len(uncovered_after)  # all covered
                            if total_cover > best_cover:
                                best_cover = total_cover
                                best_pair = (c1, c2)
                            break
                
                if best_pair is not None:
                    c1, c2 = best_pair
                    # Perform the exchange
                    if i < m:
                        blocks.pop(m)
                        blocks.pop(i)
                        block_tsubs_list.pop(m)
                        block_tsubs_list.pop(i)
                    else:
                        blocks.pop(i)
                        blocks.pop(m)
                        block_tsubs_list.pop(i)
                        block_tsubs_list.pop(m)
                    n -= 2
                    
                    blocks.append(all_blocks[c1])
                    blocks.append(all_blocks[c2])
                    block_tsubs_list.append(block_tsubs_np[c1].tolist())
                    block_tsubs_list.append(block_tsubs_np[c2].tolist())
                    n += 2
                    
                    # Update cover_count
                    for ts in block_tsubs_np[c1]:
                        cover_count[ts] += 1
                    for ts in block_tsubs_np[c2]:
                        cover_count[ts] += 1
                    improved = True
                    break
        
        if not improved:
            break
        
        blocks = reduce_blocks(blocks, seed)
        return blocks

    def build_coverage(blocks):
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        block_tsubs_list = []
        for b in blocks:
            tsubs = [tsub_idx[ts] for ts in itertools.combinations(b, t)]
            block_tsubs_list.append(tsubs)
            for ts in tsubs:
                cover_count[ts] += 1
        return cover_count, block_tsubs_list

    # Main: run multiple greedy trials, then SA and 2-block LS
    best_blocks = None
    best_len = float('inf')
    
    num_greedy = 20
    for seed in range(num_greedy):
        if time.time() > deadline - 30:
            break
        
        blocks = greedy_cover(seed)
        if not blocks:
            continue
        
        blocks = reduce_blocks(blocks, seed)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
        
        if best_len <= 135:
            break
    
    # Apply SA to best result
    if best_blocks is not None and time.time() < deadline - 15:
        sa_result = simulated_annealing(best_blocks, seed=42, max_iters=30000)
        if len(sa_result) < best_len:
            best_len = len(sa_result)
            best_blocks = sa_result
    
    # Apply 2-block local search
    if best_blocks is not None and time.time() < deadline - 5:
        ls_result = two_block_local_search(best_blocks, seed=0)
        if len(ls_result) < best_len:
            best_len = len(ls_result)
            best_blocks = ls_result
    
    # One more SA round if time permits
    if best_blocks is not None and time.time() < deadline - 5:
        sa_result2 = simulated_annealing(best_blocks, seed=123, max_iters=20000)
        if len(sa_result2) < best_len:
            best_len = len(sa_result2)
            best_blocks = sa_result2
    
    if best_blocks is None:
        best_blocks = greedy_cover(0)
        best_blocks = reduce_blocks(best_blocks, 0)
    
    return best_blocks
