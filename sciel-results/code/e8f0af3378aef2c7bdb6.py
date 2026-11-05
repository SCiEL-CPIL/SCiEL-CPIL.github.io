import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.monotonic()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build coverage matrix M (num_kblocks x num_tsubs)
    # M[i, j] = 1 if block i covers tsub j
    M = np.zeros((num_kblocks, num_tsubs), dtype=np.uint8)
    
    for i, block in enumerate(all_kblocks):
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            M[i, idx] = 1

    # Precompute list of tsub indices for each block for faster local operations
    # This avoids repeated np.where calls
    block_tsubs = []
    for i in range(num_kblocks):
        row = M[i]
        idxs = np.where(row == 1)[0]
        block_tsubs.append(idxs)

    def get_blocks(indices):
        return [list(all_kblocks[i]) for i in indices]

    best_solution = None
    best_len = float('inf')
    
    rng = random.Random(42)

    # Phase 1: Vectorized Greedy Construction
    if time.monotonic() < deadline - 5.0:
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        counts = np.zeros(num_tsubs, dtype=np.int32)
        selected_mask = np.zeros(num_kblocks, dtype=bool)
        selected_indices = []
        
        while np.any(uncovered):
            if time.monotonic() > deadline - 10.0:
                break
            
            # Calculate gain for each block: number of uncovered tsubs it covers
            gains = M @ uncovered
            
            # Find block with max gain
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            
            # Update uncovered and counts
            row = M[best_idx]
            covered_indices = block_tsubs[best_idx]
            uncovered[covered_indices] = 0
            counts[covered_indices] += 1
        
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_solution = selected_indices[:]
            best_counts = counts.copy()
            best_selected_mask = selected_mask.copy()

    # Phase 2: Local Search Polish with ILS
    if best_solution is not None:
        current_indices = best_solution[:]
        current_counts = best_counts.copy()
        current_mask = best_selected_mask.copy()
        
        improved = True
        stagnation_counter = 0
        max_stagnation = 20 # Lower to perturb more often
        
        while time.monotonic() < deadline - 2.0:
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                
            # If stagnating, perform a perturbation (ILS)
            if stagnation_counter >= max_stagnation and len(current_indices) > 0:
                # Remove 5-10 random blocks
                n_remove = min(5 + rng.randint(0, 5), len(current_indices))
                to_remove = rng.sample(current_indices, n_remove)
                for b_idx in to_remove:
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    row = M[b_idx]
                    covered_indices = block_tsubs[b_idx]
                    current_counts[covered_indices] -= 1
                
                # Greedily re-add to cover gaps
                uncovered_mask = current_counts == 0
                while np.any(uncovered_mask):
                    if time.monotonic() > deadline - 5.0:
                        break
                    gains = M @ uncovered_mask
                    best_idx = np.argmax(gains)
                    if gains[best_idx] == 0:
                        break
                    current_mask[best_idx] = True
                    current_indices.append(best_idx)
                    row = M[best_idx]
                    covered_indices = block_tsubs[best_idx]
                    current_counts[covered_indices] += 1
                    uncovered_mask = current_counts == 0
                    
                # Reset stagnation
                stagnation_counter = 0
                improved = True
                continue

            improved = False
            current_indices_set = set(current_indices)
            
            # 1. Remove Redundant Blocks
            # A block is redundant if all tsubs it covers have count > 1
            redundant_found = False
            for b_idx in current_indices:
                if time.monotonic() > deadline - 3.0:
                    break
                row = M[b_idx]
                covered_indices = block_tsubs[b_idx]
                if len(covered_indices) == 0:
                    continue
                min_count = np.min(current_counts[covered_indices])
                if min_count > 1:
                    # Remove
                    current_mask[b_idx] = False
                    current_indices.remove(b_idx)
                    current_counts[covered_indices] -= 1
                    redundant_found = True
                    improved = True
                    break # Restart loop to check for new redundancies

            # 2. Swap / Replace
            if not improved:
                # Try to replace a block with a better one
                # Heuristic: Find a block that is "critical" (has count 1 tsubs)
                # and see if another block can cover those critical tsubs plus more
                # This is expensive, so we do a limited random search.
                
                # Pick a random block to try to replace
                if current_indices:
                    b_idx = rng.choice(current_indices)
                    row = M[b_idx]
                    covered_indices = block_tsubs[b_idx]
                    
                    # Find critical tsubs (count == 1)
                    critical_mask = (current_counts == 1)
                    critical_tsubs = covered_indices[critical_mask[covered_indices]]
                    
                    if len(critical_tsubs) == 0:
                        # Should have been removed as redundant
                        current_mask[b_idx] = False
                        current_indices.remove(b_idx)
                        current_counts[covered_indices] -= 1
                        improved = True
                        continue
                    
                    # We need a new block that covers all critical_tsubs
                    # AND provides more "slack" (covers tsubs with count 2)
                    # or at least covers the same number of tsubs
                    
                    # Find candidate blocks that cover all critical_tsubs
                    # This is an intersection of sets.
                    # Since critical_tsubs is small (usually < 5), we can filter.
                    
                    # Start with all blocks covering the first critical tsub
                    first_crit = critical_tsubs[0]
                    # Find blocks covering first_crit
                    # We don't have a direct index tsub->blocks, but we can scan M column?
                    # M is (num_blocks, num_tsubs). Column access is slow in NumPy if not contiguous.
                    # However, num_tsubs is 3876. M[:, first_crit] is a vector of length 50388.
                    # This is fast enough.
                    
                    candidates = np.where(M[:, first_crit] == 1)[0]
                    
                    # Filter candidates: must cover all other critical tsubs
                    if len(critical_tsubs) > 1:
                        for crit in critical_tsubs[1:]:
                            candidates = candidates[M[candidates, crit] == 1]
                            if len(candidates) == 0:
                                break
                    
                    if len(candidates) > 0:
                        # Exclude current block
                        candidates = candidates[candidates != b_idx]
                        
                        if len(candidates) > 0:
                            # Score candidates
                            # Score = number of tsubs covered by candidate that are currently count 2
                            # This increases redundancy, allowing future deletions.
                            
                            best_candidate = -1
                            best_score = -1
                            
                            # Limit candidates to top 50 by some heuristic to save time?
                            # Or just iterate if small.
                            # candidates can be large. Let's sample or limit.
                            if len(candidates) > 100:
                                candidates = rng.sample(list(candidates), 100)
                                candidates = np.array(candidates)
                                
                            for c in candidates:
                                if current_mask[c]:
                                    continue
                                
                                c_row = M[c]
                                c_covered = block_tsubs[c]
                                
                                # Calculate score: count of c_covered where current_counts == 2
                                # Also ensure it covers all criticals (already filtered)
                                score = np.sum(current_counts[c_covered] == 2)
                                
                                # Penalty if it doesn't cover as many total tsubs?
                                # No, we just want to increase redundancy.
                                
                                if score > best_score:
                                    best_score = score
                                    best_candidate = c
                            
                            if best_candidate != -1:
                                # Perform swap
                                # Remove b_idx
                                current_mask[b_idx] = False
                                current_indices.remove(b_idx)
                                current_counts[covered_indices] -= 1
                                
                                # Add best_candidate
                                current_mask[best_candidate] = True
                                current_indices.append(best_candidate)
                                c_covered = block_tsubs[best_candidate]
                                current_counts[c_covered] += 1
                                
                                improved = True

            # Update best
            if len(current_indices) < best_len:
                best_len = len(current_indices)
                best_solution = current_indices[:]
                best_counts = current_counts.copy()
                best_selected_mask = current_mask.copy()
                improved = True # Reset stagnation on new best? 
                # Actually, if we improved, stagnation was reset above.
                # If we found a new best, we definitely want to continue.

    # Final cleanup: one more pass of redundancy removal on best_solution
    if best_solution is not None:
        final_indices = best_solution[:]
        final_mask = best_selected_mask.copy()
        final_counts = best_counts.copy()
        
        # Repeat redundancy removal until no change
        changed = True
        while changed:
            changed = False
            for i in range(len(final_indices)):
                b_idx = final_indices[i]
                row = M[b_idx]
                covered_indices = block_tsubs[b_idx]
                if len(covered_indices) == 0:
                    continue
                min_count = np.min(final_counts[covered_indices])
                if min_count > 1:
                    final_mask[b_idx] = False
                    final_indices.pop(i)
                    final_counts[covered_indices] -= 1
                    changed = True
                    break

    return get_blocks(best_solution)

