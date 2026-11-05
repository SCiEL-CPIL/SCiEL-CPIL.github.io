import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

# Precompute for each block, the list of 4-subset indices it covers
# block_subsets[i] is a list of indices into subsets
block_subsets = []
# Precompute for each subset, the list of blocks containing it
subset_to_blocks = [[] for _ in range(N_SUBSETS)]

for b_idx, block in enumerate(all_blocks):
    subs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        subs.append(idx)
        subset_to_blocks[idx].append(b_idx)
    block_subsets.append(subs)

# Convert to numpy arrays for faster access if needed, but lists are fine for small sizes
# We will use a numpy array for coverage counts
# coverage[i] = number of blocks in current solution that cover subset i

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v, k)))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    # State
    coverage = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    in_solution = np.zeros(N_BLOCKS, dtype=bool)

    def add_block(idx):
        current_blocks.append(idx)
        in_solution[idx] = True
        for sub_idx in block_subsets[idx]:
            coverage[sub_idx] += 1

    def remove_block(idx):
        current_blocks.remove(idx)
        in_solution[idx] = False
        for sub_idx in block_subsets[idx]:
            coverage[sub_idx] -= 1

    def is_valid():
        return np.all(coverage > 0)

    # Fast Greedy Initialization
    # We want to pick blocks that cover the most uncovered subsets.
    # Gain of block i = sum(1 for sub in block_subsets[i] if coverage[sub] == 0)
    # We can compute this vectorized.
    
    # Precompute block_subsets as a 2D numpy array for vectorized operations?
    # Since block sizes are uniform (35 subsets per block), we can make a (N_BLOCKS, 35) array.
    block_subsets_np = np.array(block_subsets, dtype=np.int32)
    
    # Initialize coverage to 0
    coverage = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    in_solution = np.zeros(N_BLOCKS, dtype=bool)
    
    # Greedy Loop
    while time.time() < deadline - 5:
        # Compute gain for all blocks
        # block_subsets_np has shape (N_BLOCKS, 35)
        # coverage has shape (N_SUBSETS,)
        # We want sum of (coverage[block_subsets_np[i, j]] == 0) for each i
        
        # Vectorized check: coverage[block_subsets_np] gives shape (N_BLOCKS, 35)
        # Check where coverage is 0
        is_uncovered = (coverage[block_subsets_np] == 0)
        gains = is_uncovered.sum(axis=1)
        
        # Find block with max gain
        max_gain = np.max(gains)
        if max_gain == 0:
            # No block covers any new subset? This shouldn't happen if valid solution exists and we started empty.
            # But if we are stuck, break.
            break
        
        # Pick one of the blocks with max gain (randomize tie-breaking)
        candidates = np.where(gains == max_gain)[0]
        # Exclude blocks already in solution? 
        # Actually, adding a block already in solution is useless. 
        # But greedy usually adds new blocks. 
        # Let's ensure we don't pick a block that is already in solution.
        # If all max gain blocks are in solution, we might be stuck or need a different strategy.
        # However, standard greedy for set cover adds new blocks.
        # Let's filter out in_solution blocks.
        valid_candidates = candidates[~in_solution[candidates]]
        
        if len(valid_candidates) == 0:
             # All best candidates are already in solution. 
             # This implies we are not making progress on uncovered sets with new blocks.
             # We might need to restart or perturb.
             # For now, break to avoid infinite loop.
             break
             
        # Pick random from valid candidates
        idx = random.choice(valid_candidates)
        
        # Add block
        current_blocks.append(idx)
        in_solution[idx] = True
        for sub_idx in block_subsets[idx]:
            coverage[sub_idx] += 1
            
        # Check if we are done
        if np.all(coverage > 0):
            break

    # Local Search / Improvement
    # Strategy: Try to remove redundant blocks, or swap 1 for 1 to reduce count.
    # Since we are minimizing number of blocks, and greedy gives a valid cover,
    # we can try to prune.
    
    # Pruning Phase
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Randomize order of checks
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            idx = current_blocks[i]
            # Check if block idx is redundant
            # It is redundant if for every sub in block_subsets[idx], coverage[sub] > 1
            # (i.e., removing it leaves coverage[sub] >= 1)
            
            subs = block_subsets[idx]
            # If any subset in this block has coverage == 1, it's not redundant
            # We can check vectorized:
            # coverage[subs] gives the counts for these subsets
            # If min(coverage[subs]) > 1, it's redundant.
            if np.min(coverage[subs]) > 1:
                # Remove it
                for sub_idx in subs:
                    coverage[sub_idx] -= 1
                current_blocks.remove(idx)
                in_solution[idx] = False
                improved = True
                # Restart the loop to check for new redundancies
                break

    # Swap Phase: Try to replace a block with another that allows further pruning?
    # Or simply try 1-exchange: Remove block A, Add block B, if valid and allows pruning?
    # Given time constraints, simple pruning is often the most effective for greedy set cover.
    
    # Final Verification and Return
    # Ensure all subsets are covered
    if not np.all(coverage > 0):
        # Emergency fix: add blocks until valid
        while not np.all(coverage > 0) and time.time() < deadline:
             # Find uncovered subsets
             uncovered_idxs = np.where(coverage == 0)[0]
             if len(uncovered_idxs) == 0:
                 break
             # Pick a random uncovered subset
             target_sub = random.choice(uncovered_idxs)
             # Pick a random block containing it
             candidates = subset_to_blocks[target_sub]
             # Prefer blocks not in solution
             valid_cands = [c for c in candidates if not in_solution[c]]
             if not valid_cands:
                 valid_cands = candidates
             idx = random.choice(valid_cands)
             current_blocks.append(idx)
             in_solution[idx] = True
             for sub_idx in block_subsets[idx]:
                 coverage[sub_idx] += 1

    return [list(all_blocks[i]) for i in current_blocks]
