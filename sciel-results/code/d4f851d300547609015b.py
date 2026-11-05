import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # For each k-subset, compute the list of t-subset indices it covers
    k_sub_t_idxs = []
    for k_sub in all_k_subs:
        t_idxs = []
        for t_sub in itertools.combinations(k_sub, t):
            t_idxs.append(t_sub_to_idx[t_sub])
        k_sub_t_idxs.append(np.array(t_idxs, dtype=np.int32))
    
    # Build reverse mapping: for each t-subset, which k-subsets cover it
    t_sub_to_k_idxs = [[] for _ in range(n_t)]
    for i, k_sub in enumerate(all_k_subs):
        for t_idx in k_sub_t_idxs[i]:
            t_sub_to_k_idxs[t_idx].append(i)

    def greedy_hardest(seed):
        """
        Greedy covering design: repeatedly find the hardest uncovered t-subset
        and cover it with the block that covers the most other uncovered t-subsets.
        """
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        
        # For each t-subset, track how many blocks cover it (for hardness)
        t_block_count = np.array([len(t_sub_to_k_idxs[i]) for i in range(n_t)], dtype=np.int32)
        
        # Gains: number of currently uncovered t-subsets covered by each block
        gains = np.zeros(n_k, dtype=np.int32)
        for i in range(n_k):
            gains[i] = len(k_sub_t_idxs[i])
        
        selected = []
        
        while uncovered.any():
            # Find the hardest uncovered t-subset (fewest blocks cover it)
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            # Among uncovered, find those with minimum block count
            min_count = np.min(t_block_count[unc_idx])
            hard_ts = unc_idx[t_block_count[unc_idx] == min_count]
            
            # Pick one at random among the hardest
            target_t = rng.choice(hard_ts.tolist())
            
            # Among blocks covering target_t, pick the one with max gain
            candidates = t_sub_to_k_idxs[target_t]
            if not candidates:
                break
            
            best_gain = -1
            best_idx = -1
            for b in candidates:
                g = gains[b]
                if g > best_gain or (g == best_gain and rng.random() < 0.5):
                    best_gain = g
                    best_idx = b
            
            if best_gain <= 0:
                break
            
            selected.append(best_idx)
            
            # Update gains for all blocks that share t-subsets with best_idx
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    t_block_count[t_idx] -= 1
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
        
        return selected

    def greedy_gain(seed):
        """Standard greedy: pick block with max uncovered coverage."""
        rng = random.Random(seed)
        uncovered = np.ones(n_t, dtype=bool)
        gains = np.array([len(k_sub_t_idxs[i]) for i in range(n_k)], dtype=np.int32)
        selected = []
        
        while uncovered.any():
            scores = gains.astype(np.float64)
            if seed > 0:
                scores += rng.random() * 0.01
            best_idx = int(np.argmax(scores))
            if gains[best_idx] < 1:
                break
            selected.append(best_idx)
            for t_idx in k_sub_t_idxs[best_idx]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    for b_idx in t_sub_to_k_idxs[t_idx]:
                        gains[b_idx] -= 1
        
        return selected

    def reduce_blocks(blocks_idx):
        """Remove redundant blocks via iterative pass."""
        if not blocks_idx:
            return blocks_idx

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            cover_count[k_sub_t_idxs[b_idx]] += 1

        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
            blocks_idx = new_blocks

        return blocks_idx

    def local_search(initial_blocks, max_time):
        """
        Local search: try removals and swaps to reduce block count.
        """
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        if np.any(cover_count == 0):
            return current_blocks

        best_blocks = list(current_blocks)
        best_len = len(current_blocks)
        
        rng = random.Random(999)
        no_improve = 0
        max_no_improve = 800
        
        while time.time() < max_time and no_improve < max_no_improve:
            move_type = rng.random()
            
            if move_type < 0.35:
                # Try removal
                if len(current_blocks) == 0:
                    no_improve += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                removable = True
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        removable = False
                        break
                
                if removable:
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
                    current_blocks.pop(i)
                    no_improve = 0
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = list(current_blocks)
                else:
                    no_improve += 1

            elif move_type < 0.85:
                # Try swap
                if len(current_blocks) == 0:
                    no_improve += 1
                    continue
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                # Find t-subsets uniquely covered by old_b
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    # Candidate must cover all unique t-subsets
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    # No unique subsets: try a few random candidates
                    candidates = set()
                    for _ in range(8):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                            break
                
                if not candidates:
                    no_improve += 1
                    continue
                
                new_b = rng.choice(list(candidates))
                
                # Check if swap maintains coverage
                temp_cover = cover_count.copy()
                temp_cover[k_sub_t_idxs[old_b]] -= 1
                if np.any(temp_cover == 0):
                    # Would create uncovered; only accept if new_b fixes them
                    temp_cover[k_sub_t_idxs[new_b]] += 1
                    if np.any(temp_cover == 0):
                        no_improve += 1
                        continue
                
                # Perform swap
                cover_count[k_sub_t_idxs[old_b]] -= 1
                cover_count[k_sub_t_idxs[new_b]] += 1
                current_blocks[i] = new_b
                no_improve = 0
                
                # Try to reduce after swap
                for _ in range(3):
                    found = False
                    rng.shuffle(current_blocks)
                    for j in range(len(current_blocks)):
                        b_idx = current_blocks[j]
                        essential = False
                        for t_idx in k_sub_t_idxs[b_idx]:
                            if cover_count[t_idx] == 1:
                                essential = True
                                break
                        if not essential:
                            cover_count[k_sub_t_idxs[b_idx]] -= 1
                            current_blocks.pop(j)
                            found = True
                            if len(current_blocks) < best_len:
                                best_len = len(current_blocks)
                                best_blocks = list(current_blocks)
                            break
                    if not found:
                        break
            else:
                no_improve += 1

        return best_blocks

    def simulated_annealing(initial_blocks, max_time):
        """SA with removal, swap, and add moves."""
        current_blocks = list(initial_blocks)
        if not current_blocks:
            return current_blocks

        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in current_blocks:
            cover_count[k_sub_t_idxs[b_idx]] += 1
        
        num_uncovered = int(np.sum(cover_count == 0))
        penalty = 200.0
        current_energy = len(current_blocks) + penalty * num_uncovered
        
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        rng = random.Random(7)
        T = 20.0
        cooling_rate = 0.997
        min_T = 0.1
        
        while time.time() < max_time and T > min_T:
            move_type = rng.random()
            
            if move_type < 0.45 and len(current_blocks) > 0:
                # Removal
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                new_unc = 0
                for t_idx in k_sub_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        new_unc += 1
                
                if new_unc == 0:
                    cover_count[k_sub_t_idxs[b_idx]] -= 1
                    current_blocks.pop(i)
                    current_energy -= 1.0
                else:
                    delta_E = 1.0 + penalty * new_unc
                    if rng.random() < np.exp(-delta_E / T):
                        cover_count[k_sub_t_idxs[b_idx]] -= 1
                        current_blocks.pop(i)
                        current_energy += delta_E
            
            elif move_type < 0.85 and len(current_blocks) > 0:
                # Swap
                i = rng.randint(0, len(current_blocks) - 1)
                old_b = current_blocks[i]
                
                unique_ts = []
                for t_idx in k_sub_t_idxs[old_b]:
                    if cover_count[t_idx] == 1:
                        unique_ts.append(t_idx)
                
                if unique_ts:
                    candidates = set(t_sub_to_k_idxs[unique_ts[0]])
                    for t_idx in unique_ts[1:]:
                        candidates.intersection_update(t_sub_to_k_idxs[t_idx])
                    candidates.discard(old_b)
                    candidates -= set(current_blocks)
                else:
                    candidates = set()
                    for _ in range(10):
                        c = rng.randint(0, n_k - 1)
                        if c not in current_blocks:
                            candidates.add(c)
                
                if candidates:
                    new_b = rng.choice(list(candidates))
                    
                    temp_cover = cover_count.copy()
                    temp_cover[k_sub_t_idxs[old_b]] -= 1
                    newly_unc = int(np.sum(temp_cover == 0))
                    temp_cover[k_sub_t_idxs[new_b]] += 1
                    after_unc = int(np.sum(temp_cover == 0))
                    
                    delta_E = penalty * (after_unc - num_uncovered)
                    
                    if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                        cover_count[k_sub_t_idxs[old_b]] -= 1
                        cover_count[k_sub_t_idxs[new_b]] += 1
                        current_blocks[i] = new_b
                        current_energy += delta_E
            
            else:
                # Add (only if uncovered exist)
                if num_uncovered > 0:
                    unc_idx = np.where(cover_count == 0)[0]
                    if len(unc_idx) > 0:
                        target_t = rng.choice(unc_idx.tolist())
                        candidates = [b for b in t_sub_to_k_idxs[target_t] if b not in current_blocks]
                        if candidates:
                            new_b = rng.choice(candidates)
                            delta_E = 1.0  # cost of adding
                            if delta_E <= 0 or rng.random() < np.exp(-delta_E / T):
                                cover_count[k_sub_t_idxs[new_b]] += 1
                                current_blocks.append(new_b)
                                current_energy += delta_E
            
            num_uncovered = int(np.sum(cover_count == 0))
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_blocks = list(current_blocks)
            
            T *= cooling_rate
        
        return best_blocks

    # Phase 1: Run multiple greedy strategies
    best_blocks = None
    best_len = float('inf')
    
    # Hardest greedy (most effective for covering designs)
    for seed in [0, 1, 2, 3, 4, 5]:
        if time.time() > deadline - 40.0:
            break
        b = greedy_hardest(seed)
        b = reduce_blocks(b)
        if b and len(b) < best_len:
            best_len = len(b)
            best_blocks = b
    
    # Gain greedy
    if time.time() < deadline - 30.0:
        for seed in [0, 1, 2]:
            if time.time() > deadline - 30.0:
                break
            b = greedy_gain(seed)
            b = reduce_blocks(b)
            if b and len(b) < best_len:
                best_len = len(b)
                best_blocks = b
    
    if best_blocks is None:
        return []

    # Phase 2: Local search
    if time.time() < deadline - 20.0:
        ls_time = min(15.0, deadline - time.time() - 15.0)
        if ls_time > 3.0:
            ls_result = local_search(best_blocks, time.time() + ls_time)
            ls_result = reduce_blocks(ls_result)
            if len(ls_result) < best_len:
                best_blocks = ls_result
                best_len = len(best_blocks)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 10.0:
        sa_time = min(20.0, deadline - time.time() - 8.0)
        if sa_time > 5.0:
            sa_result = simulated_annealing(best_blocks, time.time() + sa_time)
            sa_result = reduce_blocks(sa_result)
            if len(sa_result) < best_len:
                best_blocks = sa_result
                best_len = len(best_blocks)
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 5.0:
        ls_time = min(10.0, deadline - time.time() - 3.0)
        if ls_time > 2.0:
            ls_result = local_search(best_blocks, time.time() + ls_time)
            ls_result = reduce_blocks(ls_result)
            if len(ls_result) < best_len:
                best_blocks = ls_result
                best_len = len(best_blocks)
    
    # Final reduction
    if best_blocks and time.time() < deadline - 1.0:
        best_blocks = reduce_blocks(best_blocks)

    if best_blocks is None:
        return []

    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
