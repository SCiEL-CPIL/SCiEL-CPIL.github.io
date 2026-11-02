"""Covering design C(v,k,t) with WalkSAT-style stochastic local search."""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Precompute for each point the list of t-subsets containing it (for fast gain computation)
    # point_to_tsubs[point] = list of t-subset indices that contain this point
    point_to_tsubs = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubs):
        for p in ts:
            point_to_tsubs[p].append(idx)
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed, randomize_order=True):
        """Randomized greedy covering."""
        rng = random.Random(seed)
        uncovered_mask = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        
        tsub_order = np.arange(n_tsubs)
        if randomize_order:
            rng.shuffle(tsub_order.tolist())
            tsub_order = np.array(tsub_order)
        
        while uncovered_mask.any():
            first_uncovered = None
            for idx in tsub_order:
                if uncovered_mask[idx]:
                    first_uncovered = idx
                    break
            if first_uncovered is None:
                break
            
            block = list(all_tsubs[first_uncovered])
            
            while len(block) < k:
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                for x in candidates:
                    gain = 0
                    for combo in itertools.combinations(block, t-1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered_mask[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain:
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for idx in block_tsub_indices(block):
                uncovered_mask[idx] = 0
        
        return blocks
    
    def walksat_optimize(blocks, rng, max_steps=50000):
        """
        WalkSAT-style local search over block set.
        State: list of blocks. Objective: minimize number of blocks subject to all t-subs covered.
        Flips: remove a block (if still covered), add a block (to fix uncovered), or swap.
        Uses smooth penalty: soft-min of coverage counts as guide.
        """
        blocks_list = [list(b) for b in blocks]
        
        # Build coverage counts
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks_list:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        n_blocks = len(blocks_list)
        best_count = n_blocks
        best_blocks = [list(b) for b in blocks_list]
        
        # Precompute a pool of candidate blocks for additions
        # Sample random k-subsets
        candidate_pool = set()
        for _ in range(min(2000, n_tsubs)):
            cb = tuple(sorted(rng.sample(range(v), k)))
            candidate_pool.add(cb)
        
        # Also add blocks that cover currently low-coverage t-subs
        for _ in range(500):
            # Pick a random t-subset, extend to a k-block
            ts = rng.choice(all_tsubs)
            remaining = [x for x in range(v) if x not in ts]
            extra = rng.sample(remaining, k - t)
            cb = tuple(sorted(list(ts) + extra))
            candidate_pool.add(cb)
        
        candidate_pool = list(candidate_pool)
        
        step = 0
        no_improve = 0
        max_no_improve = 5000
        
        while step < max_steps and time.time() < deadline - 10:
            step += 1
            
            # Compute smooth penalty: for each t-subset, if cov_count[i] <= 1, it's at risk
            # We want to minimize total blocks while keeping all cov_count >= 1
            # Penalty = sum over t-subs of max(0, 1 - cov_count[i])  (hard constraint)
            # Plus lambda * n_blocks (soft objective)
            
            # Check if any t-sub is uncovered
            uncovered = np.where(cov_count < 1)[0]
            
            if len(uncovered) == 0:
                # Try to remove a block
                if n_blocks > 0:
                    # Pick a block to try removing - prefer blocks with low marginal coverage
                    # Compute marginal: how many t-subs would go from cov 2->1 if we remove
                    best_removal = -1
                    best_score = float('inf')
                    
                    # Sample a few blocks to try (not all, for speed)
                    sample_size = min(n_blocks, 20)
                    candidates_to_try = rng.sample(range(n_blocks), sample_size)
                    
                    for bi in candidates_to_try:
                        # Count how many t-subs in this block have cov_count == 1
                        critical = 0
                        for idx in block_tsubs[bi]:
                            if cov_count[idx] == 1:
                                critical += 1
                        
                        # Score: prefer removing blocks with few critical t-subs
                        # If critical == 0, removal is safe
                        score = critical * 1000 + rng.random()
                        
                        if score < best_score:
                            best_score = score
                            best_removal = bi
                    
                    if best_removal >= 0 and best_score < 1000:
                        # Safe removal
                        bi = best_removal
                        for idx in block_tsubs[bi]:
                            cov_count[idx] -= 1
                        blocks_list.pop(bi)
                        block_tsubs.pop(bi)
                        n_blocks -= 1
                        no_improve = 0
                        
                        if n_blocks < best_count:
                            best_count = n_blocks
                            best_blocks = [list(b) for b in blocks_list]
                        continue
                    else:
                        # Try to fix uncovered by adding a block
                        if len(uncovered) > 0:
                            # Pick an uncovered t-sub and find a block covering it
                            target_idx = rng.choice(uncovered)
                            target_ts = all_tsubs[target_idx]
                            
                            # Find candidate blocks that cover this t-sub
                            good_candidates = []
                            for cb in candidate_pool:
                                if all(p in cb for p in target_ts):
                                    good_candidates.append(cb)
                            
                            if good_candidates:
                                cb = rng.choice(good_candidates)
                                cb_list = list(cb)
                                indices = block_tsub_indices(cb_list)
                                
                                # Check if this block actually helps (covers some uncovered)
                                helps = 0
                                for idx in indices:
                                    if cov_count[idx] == 0:
                                        helps += 1
                                
                                if helps > 0:
                                    blocks_list.append(cb_list)
                                    block_tsubs.append(indices)
                                    for idx in indices:
                                        cov_count[idx] += 1
                                    n_blocks += 1
                                    no_improve += 1
                                    continue
            
            # If we couldn't make progress, do a random perturbation
            # Swap: remove one block, add another
            if n_blocks > 1:
                bi = rng.randrange(n_blocks)
                # Temporarily remove
                for idx in block_tsubs[bi]:
                    cov_count[idx] -= 1
                removed_block = blocks_list.pop(bi)
                removed_tsubs = block_tsubs.pop(bi)
                n_blocks -= 1
                
                # Add a random candidate
                if candidate_pool:
                    cb = rng.choice(candidate_pool)
                    cb_list = list(cb)
                    indices = block_tsub_indices(cb_list)
                    blocks_list.append(cb_list)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                    n_blocks += 1
                
                no_improve += 1
            
            if no_improve > max_no_improve:
                # Reset to best and restart
                blocks_list = [list(b) for b in best_blocks]
                cov_count = np.zeros(n_tsubs, dtype=np.int32)
                block_tsubs = []
                for block in blocks_list:
                    indices = block_tsub_indices(block)
                    block_tsubs.append(indices)
                    for idx in indices:
                        cov_count[idx] += 1
                n_blocks = len(blocks_list)
                no_improve = 0
                max_no_improve = int(max_no_improve * 0.8)
                if max_no_improve < 500:
                    break
        
        return best_blocks, best_count
    
    def deletion_pass(blocks, deadline):
        """Iteratively remove redundant blocks."""
        cov_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsubs = []
        for block in blocks:
            indices = block_tsub_indices(block)
            block_tsubs.append(indices)
            for idx in indices:
                cov_count[idx] += 1
        
        blocks_list = [list(b) for b in blocks]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n_blocks = len(blocks_list)
            for i in range(n_blocks):
                if time.time() > deadline - 5:
                    break
                test_cov = cov_count.copy()
                for idx in block_tsubs[i]:
                    test_cov[idx] -= 1
                if test_cov.min() >= 1:
                    blocks_list.pop(i)
                    block_tsubs.pop(i)
                    cov_count = test_cov
                    improved = True
                    break
        
        return blocks_list
    
    # Phase 1: Get initial coverings via randomized greedy
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy first
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Randomized greedy restarts
    seed = 1
    while time.time() < deadline - 40:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Deletion pass
    if time.time() < deadline - 30:
        best_blocks = deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: WalkSAT local search from multiple starting points
    rng = random.Random(42)
    
    # Start from best
    if time.time() < deadline - 20:
        ws_blocks, ws_count = walksat_optimize(best_blocks, rng, max_steps=30000)
        if ws_count < best_count:
            best_blocks = ws_blocks
            best_count = ws_count
    
    # Try WalkSAT from a few other starting points
    for start_seed in range(3):
        if time.time() > deadline - 15:
            break
        alt_blocks = greedy_build(100 + start_seed, randomize_order=True)
        alt_blocks = deletion_pass(alt_blocks, deadline)
        if len(alt_blocks) < best_count + 5:  # only if competitive
            ws_blocks, ws_count = walksat_optimize(alt_blocks, rng, max_steps=20000)
            if ws_count < best_count:
                best_blocks = ws_blocks
                best_count = ws_count
    
    # Final deletion pass
    if time.time() < deadline - 3:
        best_blocks = deletion_pass(best_blocks, deadline)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
