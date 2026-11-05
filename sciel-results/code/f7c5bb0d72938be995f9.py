import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Pre-compute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block index to list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        subs = [t_sub_index[sub] for sub in itertools.combinations(b, t)]
        block_to_tsubs.append(subs)
        
    # Map t-subset index to list of block indices that cover it
    tsub_to_blocks = [[] for _ in range(n_t_subs)]
    for b_idx, subs in enumerate(block_to_tsubs):
        for s in subs:
            tsub_to_blocks[s].append(b_idx)
    
    # Precompute block sizes (number of t-subsets each block covers) for efficiency scoring
    block_sizes = [len(subs) for subs in block_to_tsubs]
    
    def local_search(solution_blocks, solution_set, cover_count):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            
            # Pass 1: Try to remove any redundant block
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break # Restart loop as indices changed
                    
            # Pass 2: If no removal, try to swap a block to create redundancy elsewhere
            if not improved and time.time() < deadline - 5:
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    # Find critical subsets (covered only by this block)
                    critical_subsets = []
                    for s in block_to_tsubs[b_idx]:
                        if cover_count[s] == 1:
                            critical_subsets.append(s)
                            
                    if not critical_subsets:
                        # Should have been removed in Pass 1, but just in case
                        for s in block_to_tsubs[b_idx]:
                            cover_count[s] -= 1
                        solution_blocks.pop(i)
                        solution_set.remove(b_idx)
                        improved = True
                        break
                        
                    # Find a candidate block that covers all critical subsets
                    candidate_set = set(tsub_to_blocks[critical_subsets[0]])
                    for s in critical_subsets[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                            
                    if candidate_set:
                        # Pick the best candidate using weighted scoring:
                        # Score = (number of newly covered subsets) / (1 + overlap with existing solution)
                        best_candidate = -1
                        best_score = -1.0
                        for c_idx in candidate_set:
                            if c_idx not in solution_set:
                                # Compute how many subsets this candidate covers that are currently uncovered
                                # or only covered by the block being replaced
                                new_cover_gain = 0
                                for s in block_to_tsubs[c_idx]:
                                    if cover_count[s] <= 1:
                                        new_cover_gain += 1
                                # Penalize blocks that overlap heavily with existing solution
                                # (approximate by total size)
                                score = new_cover_gain / (1.0 + block_sizes[c_idx] * 0.1)
                                if score > best_score:
                                    best_score = score
                                    best_candidate = c_idx
                                    
                        if best_candidate != -1:
                            # Swap: Remove b_idx, Add best_candidate
                            for s in block_to_tsubs[b_idx]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            solution_blocks[i] = best_candidate
                            solution_set.remove(b_idx)
                            solution_set.add(best_candidate)
                            improved = True
                            break
                            
            # Pass 3: If still no improvement, try a 2-for-1 swap (replace 2 blocks with 1)
            if not improved and time.time() < deadline - 5:
                criticals = []
                for i in range(len(solution_blocks)):
                    b_idx = solution_blocks[i]
                    crits = [s for s in block_to_tsubs[b_idx] if cover_count[s] == 1]
                    criticals.append(crits)
                    
                n_pairs_to_try = 50
                for _ in range(n_pairs_to_try):
                    if time.time() > deadline - 5:
                        break
                    i, j = random.sample(range(len(solution_blocks)), 2)
                    b_i = solution_blocks[i]
                    b_j = solution_blocks[j]
                    
                    all_crits = criticals[i] + criticals[j]
                    if not all_crits:
                        continue
                        
                    candidate_set = set(tsub_to_blocks[all_crits[0]])
                    for s in all_crits[1:]:
                        candidate_set.intersection_update(tsub_to_blocks[s])
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        valid_candidates = [c for c in candidate_set if c not in solution_set]
                        if valid_candidates:
                            best_candidate = valid_candidates[0]
                            for s in block_to_tsubs[b_i]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[b_j]:
                                cover_count[s] -= 1
                            for s in block_to_tsubs[best_candidate]:
                                cover_count[s] += 1
                                
                            if i < j:
                                solution_blocks.pop(j)
                                solution_blocks.pop(i)
                            else:
                                solution_blocks.pop(i)
                                solution_blocks.pop(j)
                            solution_set.remove(b_i)
                            solution_set.remove(b_j)
                            solution_set.add(best_candidate)
                            solution_blocks.append(best_candidate)
                            
                            improved = True
                            break
                            
    best_blocks = []
    best_len = float('inf')
    
    # Run multiple rounds of randomized greedy + local search
    # Use log-sum-exp based soft scoring for block selection in greedy phase
    
    while time.time() < deadline - 15:
        # --- Phase 1: Soft-Weighted Greedy Construction ---
        # Use dual weights with random perturbation, but select blocks using
        # a soft-max inspired scoring to smooth the selection landscape
        
        # Initialize dual weights
        duals = np.random.uniform(1.0, 1.5, size=n_t_subs)
        uncovered = np.ones(n_t_subs, dtype=bool)
        total_uncovered = n_t_subs
        solution_blocks = []
        solution_set = set()
        
        while total_uncovered > 0:
            if time.time() > deadline - 15:
                break
                
            # Compute gain for each block: sum of duals for uncovered subsets it covers
            # Then apply a soft normalization: score[b] = gain[b] / (1 + alpha * degree[b])
            # where degree[b] = number of uncovered subsets in block b
            # This creates a smooth surrogate that balances coverage vs. efficiency
            
            max_score = -1.0
            best_b_idx = -1
            
            for i in range(n_blocks):
                gain = 0.0
                deg = 0
                for s in block_to_tsubs[i]:
                    if uncovered[s]:
                        gain += duals[s]
                        deg += 1
                
                if gain <= 0:
                    continue
                    
                # Soft scoring: reward high gain, penalize high degree (overlap)
                # Using log-sum-exp inspired smoothing: score = gain / (1 + 0.5 * deg / block_sizes[i])
                # This is a smooth surrogate for the hard "max gain" selection
                score = gain / (1.0 + 0.5 * deg / max(block_sizes[i], 1))
                
                if score > max_score:
                    max_score = score
                    best_b_idx = i
            
            if best_b_idx == -1 or max_score <= 0:
                break
                
            solution_blocks.append(best_b_idx)
            solution_set.add(best_b_idx)
            
            for s in block_to_tsubs[best_b_idx]:
                if uncovered[s]:
                    uncovered[s] = False
                    total_uncovered -= 1
        
        # Check if we have a valid cover
        if total_uncovered > 0:
            continue
            
        # --- Phase 2: Local Search ---
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
                
        local_search(solution_blocks, solution_set, cover_count)
        
        # Update best if improved
        if len(solution_blocks) < best_len:
            best_len = len(solution_blocks)
            best_blocks = list(solution_blocks)
            
    # Final polish: if we have time, do one more intensive local search on the best solution
    if best_blocks and time.time() < deadline - 3:
        solution_blocks = list(best_blocks)
        solution_set = set(best_blocks)
        cover_count = [0] * n_t_subs
        for b_idx in solution_blocks:
            for s in block_to_tsubs[b_idx]:
                cover_count[s] += 1
        
        # Run local search with tighter time budget for final polish
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(len(solution_blocks)):
                b_idx = solution_blocks[i]
                redundant = True
                for s in block_to_tsubs[b_idx]:
                    if cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_to_tsubs[b_idx]:
                        cover_count[s] -= 1
                    solution_blocks.pop(i)
                    solution_set.remove(b_idx)
                    improved = True
                    break
        
        if len(solution_blocks) < best_len:
            best_len = len(solution_blocks)
            best_blocks = list(solution_blocks)
    
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[b_idx]) for b_idx in best_blocks]
    return final_blocks
