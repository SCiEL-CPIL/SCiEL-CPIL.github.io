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

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]

# Precompute coverage matrix as numpy array for fast vectorized greedy
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)


def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this specific task)
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
    deadline = start_time + 115  # Safety margin below 120s kill

    def hypergraph_greedy():
        """
        Hypergraph coloring formulation: select block maximizing gain * (1/degree)
        where degree = number of currently uncovered subsets the block intersects.
        This biases toward blocks that efficiently cover rare subsets.
        """
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        # Precompute: for each block, the number of subsets it covers (always 35)
        # But we need dynamic degree based on uncovered subsets
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            weighted_uncovered = weights_np * uncovered
            gains = coverage_matrix @ weighted_uncovered
            
            # Compute degree: number of uncovered subsets each block intersects
            # This is just the count of 1s in coverage_matrix[i] & uncovered
            degrees = coverage_matrix @ uncovered
            # Avoid division by zero
            safe_degrees = np.where(degrees > 0, degrees, 1)
            
            # Score = gain / degree (or gain * (1/degree))
            scores = gains / safe_degrees
            
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            
            if best_score <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen

    def is_valid_solution(block_indices):
        """Check if the set of blocks covers all subsets."""
        if not block_indices:
            return N_SUBSETS == 0
        # Use a bitset or count array for speed
        counts = np.zeros(N_SUBSETS, dtype=np.int8)
        for idx in block_indices:
            counts += coverage_matrix[idx]
        return np.all(counts > 0)

    def repair_greedy(block_indices):
        """Add blocks to block_indices until it is a valid solution."""
        counts = np.zeros(N_SUBSETS, dtype=np.int8)
        for idx in block_indices:
            counts += coverage_matrix[idx]
        
        uncovered_mask = (counts == 0).astype(np.int8)
        remaining = int(np.sum(uncovered_mask))
        
        while remaining > 0:
            # Find best block for remaining uncovered using hypergraph scoring
            weighted_uncovered = weights_np * uncovered_mask
            gains = coverage_matrix @ weighted_uncovered
            degrees = coverage_matrix @ uncovered_mask
            safe_degrees = np.where(degrees > 0, degrees, 1)
            scores = gains / safe_degrees
            
            best_idx = int(np.argmax(scores))
            if scores[best_idx] == 0:
                # Should not happen if all blocks are valid
                break
            
            block_indices.append(best_idx)
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered_mask & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered_mask = still_uncovered
            
        return block_indices

    def local_search_2swap(current_blocks, max_time=2.0):
        """
        Local search with 1-swaps and 2-for-1 swaps.
        Returns improved list of block indices.
        """
        start_ls = time.time()
        # Work on a copy
        sol = current_blocks.copy()
        
        # Data structures for fast updates
        # counts[s] = number of blocks in sol covering subset s
        counts = np.zeros(N_SUBSETS, dtype=np.int16)
        for idx in sol:
            counts += coverage_matrix[idx]
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() - start_ls < max_time:
            improved = False
            
            # Phase 1: Prune redundant blocks (1-for-0 swap)
            # A block is redundant if all its subsets have count > 1 (excluding itself)
            to_remove = []
            for pos, idx in enumerate(sol):
                bits = block_bit_indices[idx]
                redundant = True
                for s in bits:
                    if counts[s] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(pos)
            
            # Remove in reverse order to keep indices valid
            for pos in reversed(to_remove):
                idx = sol[pos]
                counts -= coverage_matrix[idx]
                sol.pop(pos)
                improved = True
                no_improve = 0
            
            if not sol:
                break

            # Phase 2: 1-Swap (Replace 1 block with 1 block)
            candidates_for_swap = []
            for pos, idx in enumerate(sol):
                bits = block_bit_indices[idx]
                unique_bits = [s for s in bits if counts[s] == 1]
                if not unique_bits:
                    continue
                if len(unique_bits) <= 5:  # Heuristic: only try if few unique subsets
                    candidates_for_swap.append((pos, idx, unique_bits))
            
            # Shuffle to avoid bias
            random.shuffle(candidates_for_swap)
            
            for pos, idx, unique_bits in candidates_for_swap[:10]:  # Limit attempts per iteration
                if idx not in sol:
                    continue
                    
                # Find a replacement block j that covers all unique_bits
                if not unique_bits:
                    continue
                    
                cands = set(blocks_containing_subset[unique_bits[0]])
                for s in unique_bits[1:]:
                    cands.intersection_update(blocks_containing_subset[s])
                    if not cands:
                        break
                
                if not cands:
                    continue
                
                # Exclude current block
                cands.discard(idx)
                
                if cands:
                    # Pick the best candidate using hypergraph scoring
                    # Score based on how many count=1 subsets it covers (beyond unique_bits)
                    best_j = None
                    best_score = -1
                    for j in list(cands)[:20]:  # Limit candidates
                        # Count how many subsets with count==1 this block covers
                        bits_j = block_bit_indices[j]
                        score = sum(1 for s in bits_j if counts[s] == 1)
                        if score > best_score:
                            best_score = score
                            best_j = j
                    
                    if best_j is not None:
                        # Perform swap
                        counts -= coverage_matrix[idx]
                        counts += coverage_matrix[best_j]
                        sol[pos] = best_j
                        improved = True
                        no_improve = 0
                        break  # Restart loop after a swap
            
            if improved:
                continue

            # Phase 3: 2-Swap (Replace 2 blocks with 1 block)
            blocks_with_uniques = []
            for pos, idx in enumerate(sol):
                bits = block_bit_indices[idx]
                unique_bits = [s for s in bits if counts[s] == 1]
                if unique_bits:
                    blocks_with_uniques.append((pos, idx, unique_bits))
            
            if len(blocks_with_uniques) < 2:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
                continue
            
            # Sample pairs
            n_pairs_to_try = 20
            pairs_tried = 0
            
            indices_in_bwu = list(range(len(blocks_with_uniques)))
            random.shuffle(indices_in_bwu)
            
            found_swap = False
            for p1 in indices_in_bwu:
                if pairs_tried >= n_pairs_to_try:
                    break
                pos1, idx1, unique1 = blocks_with_uniques[p1]
                
                for p2 in indices_in_bwu:
                    if p1 == p2:
                        continue
                    if pairs_tried >= n_pairs_to_try:
                        break
                    pairs_tried += 1
                    
                    pos2, idx2, unique2 = blocks_with_uniques[p2]
                    
                    # Check if idx1 and idx2 are still in sol
                    if idx1 not in sol or idx2 not in sol:
                        continue
                    
                    # Union of unique subsets
                    union_unique = set(unique1) | set(unique2)
                    
                    if len(union_unique) > 7:  # Heuristic: if too many, unlikely to be covered by 1 block
                        continue
                        
                    # Find a block covering union_unique
                    if not union_unique:
                        continue
                        
                    cands = set(blocks_containing_subset[list(union_unique)[0]])
                    for s in list(union_unique)[1:]:
                        cands.intersection_update(blocks_containing_subset[s])
                        if not cands:
                            break
                    
                    if not cands:
                        continue
                        
                    # Exclude idx1 and idx2
                    cands.discard(idx1)
                    cands.discard(idx2)
                    
                    if cands:
                        # Pick best candidate using hypergraph scoring
                        best_k = None
                        best_score = -1
                        for k in list(cands)[:20]:
                            bits_k = block_bit_indices[k]
                            score = sum(1 for s in bits_k if counts[s] == 1)
                            if score > best_score:
                                best_score = score
                                best_k = k
                        
                        if best_k is not None:
                            # Perform 2-for-1 swap
                            counts -= coverage_matrix[idx1]
                            counts -= coverage_matrix[idx2]
                            counts += coverage_matrix[best_k]
                            
                            # Remove idx1 and idx2 from sol
                            pos_remove1 = sol.index(idx1)
                            pos_remove2 = sol.index(idx2)
                            
                            # Remove higher index first
                            if pos_remove1 > pos_remove2:
                                sol.pop(pos_remove1)
                                sol.pop(pos_remove2)
                            else:
                                sol.pop(pos_remove2)
                                sol.pop(pos_remove1)
                            
                            sol.append(best_k)
                            
                            improved = True
                            no_improve = 0
                            found_swap = True
                            break
                        
                if found_swap:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return sol

    # 1. Hypergraph greedy initialization
    current_blocks = hypergraph_greedy()
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # Quick prune
    counts = np.zeros(N_SUBSETS, dtype=np.int16)
    for idx in best_blocks:
        counts += coverage_matrix[idx]
        
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for pos in range(len(best_blocks)):
            idx = best_blocks[pos]
            bits = block_bit_indices[idx]
            redundant = True
            for s in bits:
                if counts[s] == 1:
                    redundant = False
                    break
            if redundant:
                counts -= coverage_matrix[idx]
                best_blocks.pop(pos)
                changed = True
                if len(best_blocks) < best_len:
                    best_len = len(best_blocks)
                break

    # 2. Iterated Local Search
    current_sol = best_blocks.copy()
    
    while time.time() < deadline - 10:
        # Local Search
        improved_sol = local_search_2swap(current_sol, max_time=5.0)
        
        if len(improved_sol) < best_len:
            best_len = len(improved_sol)
            best_blocks = improved_sol.copy()
            current_sol = improved_sol.copy()
        else:
            # Perturbation
            # Remove 3-5 random blocks
            n_remove = random.randint(3, 5)
            if len(current_sol) > n_remove:
                to_remove = random.sample(current_sol, n_remove)
                # Remove them
                current_sol = [b for b in current_sol if b not in to_remove]
                # Repair using hypergraph greedy
                current_sol = repair_greedy(current_sol)
                
                if len(current_sol) < best_len:
                    best_len = len(current_sol)
                    best_blocks = current_sol.copy()

    # Final validation and formatting
    seen = set()
    final_blocks = []
    for idx in best_blocks:
        if idx not in seen:
            seen.add(idx)
            final_blocks.append(list(all_blocks[idx]))
    
    return final_blocks
