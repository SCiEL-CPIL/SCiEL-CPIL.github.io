"""
Lagrangian-relaxation guided greedy covering with local-search refinement.
"""
import itertools
import time
import random


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    all_points = list(range(v))
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # Precompute coverage masks: ksub_masks[j] has bit i set iff ksub j covers tsub i
    ksub_masks = [0] * n_ksubs
    ksub_tsub_indices = [[] for _ in range(n_ksubs)]  # list of tsub indices for each ksub
    for j, ks in enumerate(all_ksubs):
        mask = 0
        indices = []
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
            indices.append(idx)
        ksub_masks[j] = mask
        ksub_tsub_indices[j] = indices

    def bit_count(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # --- Phase 1: Lagrangian-relaxation guided greedy ---
    # Weights w[i] for each t-subset; initially 1.0
    # Gain for block j = sum of w[i] for uncovered i in block j
    # After each selection, update weights via subgradient:
    #   w[i] += step * (1 if i still uncovered else 0) - step * lambda * (1 if i covered by selected block else 0)
    # Simplified: w[i] increases if uncovered, decreases slightly if covered.
    
    weights = [1.0] * n_tsubs
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs
    blocks = []
    used_blocks = set()

    # Precompute for faster gain calculation: for each block, list of tsub indices
    # Gain = sum(weights[i] for i in block's tsubs if uncovered)
    
    # To speed up, maintain a numpy-like structure? No, just use lists.
    # Actually, for 50k blocks * 150 iterations, we need fast gain computation.
    # Let's use the bitset approach for uncovered, but weighted sum requires iteration.
    # Alternative: since weights change, we can't precompute.
    # But we can maintain an array of weights and for each block, sum weights of its tsubs that are uncovered.
    
    # Optimization: maintain uncovered as a boolean array for O(1) check
    uncovered_arr = [1] * n_tsubs

    step_size = 0.1
    decay = 0.95

    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = 0.0

        # Scan all candidates
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            
            # Calculate weighted gain
            gain = 0.0
            for idx in ksub_tsub_indices[j]:
                if uncovered_arr[idx]:
                    gain += weights[idx]
            
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain <= 0:
            break

        # Add best block
        block = list(all_ksubs[best_j])
        blocks.append(block)
        used_blocks.add(best_j)

        # Update uncovered and weights
        for idx in ksub_tsub_indices[best_j]:
            if uncovered_arr[idx]:
                uncovered_arr[idx] = 0
                remaining -= 1
                # Weight update: decrease for covered
                weights[idx] = max(0.0, weights[idx] - step_size)
            else:
                # Already covered, small decrease
                weights[idx] = max(0.0, weights[idx] * decay)
        
        # Increase weights for still-uncovered (subgradient ascent)
        # This is O(n_tsubs) per iteration, which is fine
        for i in range(n_tsubs):
            if uncovered_arr[i]:
                weights[i] = min(10.0, weights[i] + step_size)

    # --- Phase 2: Local search refinement ---
    if remaining > 0:
        # If greedy didn't finish, try to repair with simple greedy on remaining
        # But first, let's do local search on what we have
        pass

    # Build coverage counts
    coverage = [0] * n_tsubs
    block_set = set(blocks)
    for b in blocks:
        for idx in ksub_tsub_indices[all_ksubs.index(b)]:  # This is slow, let's use a map
            coverage[idx] += 1

    # Better: create a map from block tuple to index
    block_to_idx = {all_ksubs[j]: j for j in range(n_ksubs)}

    coverage = [0] * n_tsubs
    for b in blocks:
        idx = block_to_idx[b]
        for ts_idx in ksub_tsub_indices[idx]:
            coverage[ts_idx] += 1

    def is_redundant(b):
        idx = block_to_idx[b]
        for ts_idx in ksub_tsub_indices[idx]:
            if coverage[ts_idx] == 1:
                return False
        return True

    # Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates = list(blocks)
        for b in candidates:
            if b in block_set and is_redundant(b):
                idx = block_to_idx[b]
                for ts_idx in ksub_tsub_indices[idx]:
                    coverage[ts_idx] -= 1
                block_set.remove(b)
                blocks.remove(b)
                changed = True

    # 1-Exchange local search
    # For each block, try to swap with another that covers its critical tsubs
    for _pass in range(20):
        if time.time() > deadline:
            break
        improved = False
        blocks_to_try = blocks[:]
        random.shuffle(blocks_to_try)
        
        for b in blocks_to_try:
            if b not in block_set:
                continue
            idx = block_to_idx[b]
            
            # Find critical tsubs (covered only by b)
            critical_ts = [ts_idx for ts_idx in ksub_tsub_indices[idx] if coverage[ts_idx] == 1]
            
            if not critical_ts:
                # Redundant, remove
                for ts_idx in ksub_tsub_indices[idx]:
                    coverage[ts_idx] -= 1
                block_set.remove(b)
                blocks.remove(b)
                improved = True
                continue
            
            # Find candidate blocks that cover all critical_ts
            # Intersect tsub_to_blocks for each critical_ts
            # Build tsub_to_blocks once
            tsub_to_blocks = {i: [] for i in range(n_tsubs)}
            for j in range(n_ksubs):
                for ts_idx in ksub_tsub_indices[j]:
                    tsub_to_blocks[ts_idx].append(j)
            
            candidates = None
            for ts_idx in critical_ts:
                covering = set(tsub_to_blocks[ts_idx])
                if candidates is None:
                    candidates = covering
                else:
                    candidates &= covering
            
            if candidates:
                candidates -= block_set
                # Pick candidate with max coverage of non-critical tsubs it covers
                best_cand = None
                best_score = -1
                for cand_j in candidates:
                    if cand_j in block_set:
                        continue
                    score = 0
                    for ts_idx in ksub_tsub_indices[cand_j]:
                        if coverage[ts_idx] > 0:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_cand = cand_j
                
                if best_cand is not None:
                    # Swap b with best_cand
                    # Remove b
                    for ts_idx in ksub_tsub_indices[idx]:
                        coverage[ts_idx] -= 1
                    block_set.remove(b)
                    blocks.remove(b)
                    
                    # Add best_cand
                    new_b = all_ksubs[best_cand]
                    for ts_idx in ksub_tsub_indices[best_cand]:
                        coverage[ts_idx] += 1
                    block_set.add(new_b)
                    blocks.append(new_b)
                    improved = True

        if not improved:
            break

    # Final redundancy removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        candidates = list(blocks)
        for b in candidates:
            if b in block_set and is_redundant(b):
                idx = block_to_idx[b]
                for ts_idx in ksub_tsub_indices[idx]:
                    coverage[ts_idx] -= 1
                block_set.remove(b)
                blocks.remove(b)
                changed = True

    # Verify coverage
    for i in range(n_tsubs):
        if coverage[i] == 0:
            # Need to add blocks to cover this tsub
            # Find any block covering it
            for j in tsub_to_blocks[i]:
                b = all_ksubs[j]
                if b not in block_set:
                    for ts_idx in ksub_tsub_indices[j]:
                        coverage[ts_idx] += 1
                    block_set.add(b)
                    blocks.append(b)
                    break

    return blocks
