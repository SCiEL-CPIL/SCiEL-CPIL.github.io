import itertools
import random
import time

# Global Precomputation for v=19, k=7, t=4
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
# For each subset, list of blocks containing it
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
# For each block, list of subset indices it covers
all_block_subsets = []

for block in itertools.combinations(range(V), K):
    mask = 0
    subs = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        subs.append(idx)
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)
    all_block_subsets.append(subs)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
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
    deadline = start_time + 118  # Safety margin for 120s limit
    
    best_len = float('inf')
    best_blocks = []
    
    # ILS Parameters
    num_restarts = 20
    # We will dynamically allocate time, but let's start with a fixed number of restarts
    # and adjust if time allows more.
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(42 + restart * 137)
        
        current_blocks = []
        current_masks = []
        counts = [0] * N_SUBSETS
        in_set = set()
        
        def add_block(idx):
            current_blocks.append(idx)
            current_masks.append(all_masks[idx])
            in_set.add(idx)
            for bit_idx in all_block_subsets[idx]:
                counts[bit_idx] += 1

        def remove_block(idx):
            pos = current_blocks.index(idx)
            m = current_masks[pos]
            for bit_idx in all_block_subsets[idx]:
                counts[bit_idx] -= 1
            current_blocks.pop(pos)
            current_masks.pop(pos)
            in_set.discard(idx)

        def is_redundant(idx):
            for bit_idx in all_block_subsets[idx]:
                if counts[bit_idx] == 1:
                    return False
            return True

        def prune_all():
            """Remove all redundant blocks. Returns number removed."""
            removed = 0
            i = 0
            while i < len(current_blocks):
                if is_redundant(current_blocks[i]):
                    remove_block(current_blocks[i])
                    removed += 1
                    # Do not increment i, check the next block at the same index
                else:
                    i += 1
            return removed

        # 1. Fast Greedy Initialization
        uncovered_mask = (1 << N_SUBSETS) - 1
        while uncovered_mask:
            best_gain = 0
            candidates = []
            # Optimization: Only check blocks that cover at least one uncovered bit?
            # Hard to filter efficiently without precomputed indices. 
            # Given N_BLOCKS ~ 50388, a full scan is ~50k ops. 
            # 150 iterations * 50k = 7.5M ops. Very fast in Python if optimized.
            for i in range(N_BLOCKS):
                if i in in_set:
                    continue
                gain = (all_masks[i] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Randomized tie-breaking
            best_idx = random.choice(candidates)
            add_block(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
            
        # Initial Prune
        prune_all()
        
        current_len = len(current_blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = current_blocks.copy()
            
        # 2. Local Search Loop
        no_improve_count = 0
        max_no_improve = 30
        
        while time.time() < deadline:
            # Check time more frequently
            if time.time() > deadline:
                break
                
            # Phase 1: Prune
            if prune_all() > 0:
                no_improve_count = 0
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                continue
            
            if len(current_blocks) == 0:
                break
                
            # Phase 2: Try 1-out/1-in swap
            improved = False
            # Try a few random swaps before perturbing
            for _ in range(5):
                i = random.randint(0, len(current_blocks) - 1)
                idx_i = current_blocks[i]
                
                # Find subsets that would become uncovered if we remove idx_i
                uncovered_bits = []
                for bit_idx in all_block_subsets[idx_i]:
                    if counts[bit_idx] == 1:
                        uncovered_bits.append(bit_idx)
                        
                if not uncovered_bits:
                    # It was redundant, but prune_all should have caught it. 
                    # Just in case, remove it.
                    remove_block(idx_i)
                    improved = True
                    break
                    
                # If too many bits to cover, skip
                if len(uncovered_bits) > 5:
                    continue
                
                # Create mask of critical subsets
                critical_mask = 0
                for b in uncovered_bits:
                    critical_mask |= 1 << b
                    
                # Find most constrained subset
                min_len = float('inf')
                best_subset = uncovered_bits[0]
                for b in uncovered_bits:
                    l = len(blocks_containing_subset[b])
                    if l < min_len:
                        min_len = l
                        best_subset = b
                        
                # Search for a block that covers all critical subsets
                candidates = blocks_containing_subset[best_subset]
                found = False
                for j in candidates:
                    if j == idx_i or j in in_set:
                        continue
                    if (all_masks[j] & critical_mask) == critical_mask:
                        # Swap
                        remove_block(idx_i)
                        add_block(j)
                        improved = True
                        found = True
                        break
                if found:
                    break
                    
            if improved:
                no_improve_count = 0
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                continue
                
            # Phase 3: Perturbation
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Add random blocks to escape local minimum
                # Add 1-2 blocks that cover "tight" subsets (count==1 or 2)
                # Or just random blocks
                perturbation_size = random.randint(1, 2)
                for _ in range(perturbation_size):
                    # Prefer blocks that cover many currently singly-covered subsets
                    # This increases redundancy, allowing more pruning options
                    best_pert_idx = -1
                    best_pert_gain = -1
                    for _ in range(20): # Sample 20 random blocks
                        c = random.randint(0, N_BLOCKS - 1)
                        if c in in_set:
                            continue
                        gain = 0
                        for bit_idx in all_block_subsets[c]:
                            if counts[bit_idx] <= 2:
                                gain += 1
                        if gain > best_pert_gain:
                            best_pert_gain = gain
                            best_pert_idx = c
                    if best_pert_idx != -1:
                        add_block(best_pert_idx)
                    else:
                        c = random.randint(0, N_BLOCKS - 1)
                        if c not in in_set:
                            add_block(c)
                        
                no_improve_count = 0
                max_no_improve = min(100, max_no_improve * 1.5)

    # Final cleanup
    if not best_blocks:
        # Should not happen, but fallback
        return [list(all_blocks[0])]
        
    return [list(all_blocks[i]) for i in best_blocks]
