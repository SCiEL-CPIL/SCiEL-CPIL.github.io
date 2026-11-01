import itertools
import random
import time
import numpy as np
from collections import Counter

# Precomputation for v=24, k=7, t=3
V, K, T = 24, 7, 3
ALL_TSETS = list(itertools.combinations(range(V), T))
NUM_TSETS = len(ALL_TSETS)
TSET_TO_IDX = {ts: i for i, ts in enumerate(ALL_TSETS)}

# For each tset, list of blocks that contain it
ALL_BLOCKS = list(itertools.combinations(range(V), K))
NUM_BLOCKS = len(ALL_BLOCKS)
BLOCK_TO_IDX = {b: i for i, b in enumerate(ALL_BLOCKS)}
TSET_TO_BLOCKS = [[] for _ in range(NUM_TSETS)]

for b_idx, block in enumerate(ALL_BLOCKS):
    for ts in itertools.combinations(block, T):
        ts_idx = TSET_TO_IDX[ts]
        TSET_TO_BLOCKS[ts_idx].append(b_idx)

# Precompute for each block, the list of tset indices it contains.
BLOCK_TO_TSETS = []
for block in ALL_BLOCKS:
    block_to_tsets_idx = [TSET_TO_IDX[ts] for ts in itertools.combinations(block, T)]
    BLOCK_TO_TSETS.append(block_to_tsets_idx)

def _greedy_covering(v, k, t, rng, time_limit):
    """
    Fast greedy using precomputed TSET_TO_BLOCKS.
    """
    deadline = time.time() + time_limit
    uncovered = bytearray([1] * NUM_TSETS)
    uncovered_count = NUM_TSETS
    blocks = []
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick random uncovered tset
        while True:
            idx = rng.randint(0, NUM_TSETS - 1)
            if uncovered[idx]:
                break
        
        # Get candidate blocks
        candidates = TSET_TO_BLOCKS[idx]
        best_block_idx = -1
        best_gain = -1
        
        # Iterate candidates
        for b_idx in candidates:
            gain = 0
            for ts_idx in BLOCK_TO_TSETS[b_idx]:
                if uncovered[ts_idx]:
                    gain += 1
                    if gain > best_gain:
                        if gain == 35:
                            break
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        # Add block
        block_tuple = ALL_BLOCKS[best_block_idx]
        blocks.append(list(block_tuple))
        
        # Update uncovered
        for ts_idx in BLOCK_TO_TSETS[best_block_idx]:
            if uncovered[ts_idx]:
                uncovered[ts_idx] = 0
                uncovered_count -= 1
        
        if time.time() > deadline:
            break

    return blocks

def _local_search(blocks, v, k, t, rng, time_limit):
    """
    Local search: Remove redundant, then Swap.
    Uses precomputed structures for speed.
    """
    deadline = time.time() + time_limit
    blocks = [tuple(b) for b in blocks]
    
    # State
    tset_count = np.zeros(NUM_TSETS, dtype=np.int32)
    block_tsets = []
    
    def build_state(blocks):
        nonlocal tset_count, block_tsets
        tset_count = np.zeros(NUM_TSETS, dtype=np.int32)
        block_tsets = []
        for b in blocks:
            b_tsets = BLOCK_TO_TSETS[BLOCK_TO_IDX[b]]
            block_tsets.append(b_tsets)
            for ts_idx in b_tsets:
                tset_count[ts_idx] += 1
    
    build_state(blocks)
    
    def remove_redundant():
        nonlocal blocks, tset_count, block_tsets
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            new_block_tsets = []
            new_tset_count = np.zeros(NUM_TSETS, dtype=np.int32)
            
            for i, b in enumerate(blocks):
                b_tsets = block_tsets[i]
                is_critical = False
                for ts_idx in b_tsets:
                    if tset_count[ts_idx] == 1:
                        is_critical = True
                        break
                
                if is_critical:
                    new_blocks.append(b)
                    new_block_tsets.append(b_tsets)
                    for ts_idx in b_tsets:
                        new_tset_count[ts_idx] += 1
                else:
                    changed = True
            
            if changed:
                blocks = new_blocks
                block_tsets = new_block_tsets
                tset_count = new_tset_count

    def try_swap():
        nonlocal blocks, tset_count, block_tsets
        n = len(blocks)
        if n == 0:
            return False
        
        idx = rng.randint(0, n - 1)
        old_block = blocks[idx]
        old_tsets = block_tsets[idx]
        
        # Find newly uncovered
        newly_uncovered = []
        for ts_idx in old_tsets:
            if tset_count[ts_idx] == 1:
                newly_uncovered.append(ts_idx)
        
        if not newly_uncovered:
            # Redundant
            blocks.pop(idx)
            block_tsets.pop(idx)
            for ts_idx in old_tsets:
                tset_count[ts_idx] -= 1
            return True
        
        # Find replacement
        union_points = set()
        for ts_idx in newly_uncovered:
            union_points.update(ALL_TSETS[ts_idx])
        
        if len(union_points) > k:
            return False
        
        # Candidate blocks must contain union_points.
        target_tsets = newly_uncovered
        if len(target_tsets) == 1:
            candidates = TSET_TO_BLOCKS[target_tsets[0]]
        elif len(target_tsets) == 2:
            set1 = set(TSET_TO_BLOCKS[target_tsets[0]])
            candidates = [b_idx for b_idx in TSET_TO_BLOCKS[target_tsets[1]] if b_idx in set1]
        else:
            candidates = TSET_TO_BLOCKS[target_tsets[0]]
        
        # Filter candidates that cover all newly_uncovered
        found = -1
        old_block_idx = BLOCK_TO_IDX[old_block]
        for b_idx in candidates:
            if b_idx == old_block_idx:
                continue
            b_tsets = BLOCK_TO_TSETS[b_idx]
            # Check if all newly_uncovered are in b_tsets
            # Convert newly_uncovered to set for O(1) lookup
            nu_set = set(newly_uncovered)
            if all(ts_idx in b_tsets for ts_idx in nu_set):
                found = b_idx
                break
        
        if found == -1:
            return False
        
        # Perform swap
        new_block = ALL_BLOCKS[found]
        
        # Update state
        for ts_idx in old_tsets:
            tset_count[ts_idx] -= 1
        
        new_tsets_list = BLOCK_TO_TSETS[found]
        for ts_idx in new_tsets_list:
            tset_count[ts_idx] += 1
        
        blocks[idx] = new_block
        block_tsets[idx] = new_tsets_list
        
        return True
    
    # Main loop
    no_progress_count = 0
    while time.time() < deadline:
        # Try remove redundant
        remove_redundant()
        
        # Try swaps
        swaps_done = 0
        for _ in range(100):
            if time.time() > deadline:
                break
            if try_swap():
                swaps_done += 1
        
        if swaps_done == 0:
            no_progress_count += 1
            if no_progress_count > 10:
                break
        else:
            no_progress_count = 0
    
    return blocks

def construct(v, k, t):
    start_time = time.time()
    total_budget = 140.0 # 150s limit, leave 10s margin
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy + local search instances
    num_restarts = 8
    per_restart_time = (total_budget - 5.0) / num_restarts
    
    for i in range(num_restarts):
        remaining = total_budget - (time.time() - start_time) - 5.0
        if remaining < 1.0:
            break
        
        rng = random.Random(1000 + i)
        
        # Greedy
        greedy_time = min(1.0, remaining * 0.15)
        blocks = _greedy_covering(v, k, t, rng, greedy_time)
        
        # Local Search
        ls_time = remaining - greedy_time - 1.0
        if ls_time > 0.5:
            refined = _local_search(blocks, v, k, t, rng, ls_time)
            blocks = refined
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        best_blocks = _greedy_covering(v, k, t, rng, 2.0)
        
    return best_blocks
