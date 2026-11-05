"""Simulated Annealing for C(19,7,4) with vectorized coverage matrix.

State: a set of blocks. Objective: minimize uncovered t-subsets.
Moves: add a block, remove a block, swap a block.
Coverage tracked via a boolean matrix; moves evaluated in O(1) via row sums.
"""
# EVOLVE-BLOCK-START
import itertools
import math
import time
import random

import numpy as np


def _precompute(v, k, t):
    """Precompute all t-subsets, block->subset matrix, and subset enumeration."""
    all_subs = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    # Precompute t-subsets of each possible k-block as a list of indices
    # For v=19,k=7,t=4: C(19,7)=50388 blocks, each with C(7,4)=35 subsets
    # We compute lazily but cache the subset indices for each block
    # Actually, let's precompute a mapping from block (as tuple) to list of subset indices
    # That's 50388 * 35 = ~1.76M entries, manageable.

    # Better: precompute for each block the list of t-subset indices
    # Use a dict: block_tuple -> np.array of indices
    block_to_subs = {}
    for block in itertools.combinations(range(v), k):
        idxs = []
        for sub in itertools.combinations(block, t):
            idxs.append(sub_to_idx[sub])
        block_to_subs[block] = np.array(idxs, dtype=np.int32)

    return all_subs, sub_to_idx, n_subsets, block_to_subs


def _coverage_matrix(blocks, block_to_subs, n_subsets):
    """Build coverage count array: how many blocks cover each t-subset."""
    cov = np.zeros(n_subsets, dtype=np.int32)
    for b in blocks:
        for idx in block_to_subs[b]:
            cov[idx] += 1
    return cov


def _uncovered_count(cov):
    """Number of t-subsets with zero coverage."""
    return int(np.sum(cov == 0))


def _simulated_annealing(v, k, t, deadline, seed=42):
    """
    Simulated Annealing to minimize uncovered t-subsets.
    
    State: list of blocks (tuples).
    Objective: minimize number of uncovered t-subsets (equivalently, maximize coverage).
    Moves:
      1. Add a random block (if not already present).
      2. Remove a random block (if its removal doesn't uncover any t-subset, or accept with prob).
      3. Swap: remove one block, add another.
    
    We want to END with a valid covering (0 uncovered). If we can't reach 0,
    we fall back to greedy repair.
    """
    all_subs, sub_to_idx, n_subsets, block_to_subs = _precompute(v, k, t)
    all_blocks_list = list(block_to_subs.keys())
    n_blocks_possible = len(all_blocks_list)
    
    rng = random.Random(seed)
    
    # Initial solution: random pool of blocks
    # Start with a moderately large pool
    initial_size = 200
    blocks = set(rng.sample(all_blocks_list, initial_size))
    cov = _coverage_matrix(list(blocks), block_to_subs, n_subsets)
    current_uncovered = _uncovered_count(cov)
    best_uncovered = current_uncovered
    best_blocks = list(blocks)
    
    # SA parameters
    T_start = 50.0
    T_end = 0.1
    alpha = 0.9995  # cooling rate per step
    max_steps = 200000
    step = 0
    
    start_time = time.time()
    
    while step < max_steps:
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (step / max_steps)
        
        # Choose a move type
        move_type = rng.random()
        if move_type < 0.4:
            # Add a random block
            new_block = rng.choice(all_blocks_list)
            if new_block in blocks:
                step += 1
                continue
            # Evaluate: how many new subsets does it cover?
            new_sub_idxs = block_to_subs[new_block]
            new_gain = int(np.sum(cov[new_sub_idxs] == 0))
            # Delta in uncovered: -new_gain
            delta = -new_gain
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                blocks.add(new_block)
                cov[new_sub_idxs] += 1
                current_uncovered += delta
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = list(blocks)
        elif move_type < 0.7:
            # Remove a random block
            if len(blocks) <= 50:
                step += 1
                continue
            block_to_remove = rng.choice(list(blocks))
            rem_sub_idxs = block_to_subs[block_to_remove]
            # How many subsets become uncovered?
            lost = int(np.sum(cov[rem_sub_idxs] == 1))
            delta = lost  # uncovered increases by 'lost'
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                blocks.discard(block_to_remove)
                cov[rem_sub_idxs] -= 1
                current_uncovered += delta
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = list(blocks)
        else:
            # Swap: remove one, add another
            if len(blocks) < 50:
                step += 1
                continue
            block_out = rng.choice(list(blocks))
            block_in = rng.choice(all_blocks_list)
            if block_in in blocks:
                step += 1
                continue
            out_sub_idxs = block_to_subs[block_out]
            in_sub_idxs = block_to_subs[block_in]
            # Compute delta: subsets that become uncovered minus subsets that become covered
            # After removal, cov decreases on out_sub_idxs; after addition, cov increases on in_sub_idxs
            # Net effect on uncovered:
            # - Subsets in out_sub_idxs with cov==1 become uncovered (bad)
            # - Subsets in in_sub_idxs with cov==0 become covered (good)
            # But if a subset is in both, the net is 0 for that subset.
            # Simpler: compute the change step by step
            lost = int(np.sum(cov[out_sub_idxs] == 1))
            gained = int(np.sum(cov[in_sub_idxs] == 0))
            # Subsets in intersection: if cov==1, removal makes it 0, addition makes it 1 -> net 0
            # if cov==0, removal doesn't change (stays 0), addition makes it 1 -> net -1
            # Let's just compute it properly
            # Temporarily simulate
            cov_temp = cov.copy()
            cov_temp[out_sub_idxs] -= 1
            lost_real = int(np.sum(cov_temp == 0)) - int(np.sum(cov == 0))
            cov_temp[in_sub_idxs] += 1
            gained_real = int(np.sum(cov_temp == 0)) - int(np.sum(cov_temp == 0))
            # Actually, let's just compute the final uncovered difference
            final_uncovered = int(np.sum(cov_temp == 0))
            delta = final_uncovered - current_uncovered
            
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-10)):
                blocks.discard(block_out)
                blocks.add(block_in)
                cov[out_sub_idxs] -= 1
                cov[in_sub_idxs] += 1
                current_uncovered = final_uncovered
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_blocks = list(blocks)
        
        step += 1
    
    # If we haven't achieved full coverage, repair with greedy
    if best_uncovered > 0:
        # Use best solution found and repair
        blocks = set(best_blocks)
        cov = _coverage_matrix(list(blocks), block_to_subs, n_subsets)
        current_uncovered = _uncovered_count(cov)
        
        # Greedy repair: add blocks that cover the most uncovered subsets
        while current_uncovered > 0 and time.time() < deadline:
            # Find the block that covers the most uncovered subsets
            best_gain = -1
            best_block = None
            # Sample a subset of blocks for speed
            sample_blocks = rng.sample(all_blocks_list, min(5000, n_blocks_possible))
            for b in sample_blocks:
                if b in blocks:
                    continue
                gain = int(np.sum(cov[block_to_subs[b]] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None or best_gain <= 0:
                # Try all blocks
                for b in all_blocks_list:
                    if b in blocks:
                        continue
                    gain = int(np.sum(cov[block_to_subs[b]] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
            if best_block is None or best_gain <= 0:
                break
            blocks.add(best_block)
            cov[block_to_subs[best_block]] += 1
            current_uncovered -= best_gain
        
        best_blocks = list(blocks)
        best_uncovered = current_uncovered
    
    # Final pruning: remove redundant blocks
    if time.time() < deadline:
        blocks = set(best_blocks)
        cov = _coverage_matrix(list(blocks), block_to_subs, n_subsets)
        # Remove blocks that are redundant
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for b in list(blocks):
                sub_idxs = block_to_subs[b]
                if np.all(cov[sub_idxs] > 1):
                    blocks.discard(b)
                    cov[sub_idxs] -= 1
                    changed = True
    
    return [list(b) for b in blocks]


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Run simulated annealing
    blocks = _simulated_annealing(v, k, t, deadline, seed=42)
    
    # Verify and return
    return blocks
# EVOLVE-BLOCK-END
