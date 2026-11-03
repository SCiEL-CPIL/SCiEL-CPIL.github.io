import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s if evaluator allows, but 120s is the hard stop in contract)
    
    # Fallback for non-target parameters to avoid hanging or crashing on large inputs
    if v != 19 or k != 7 or t != 4:
        # Simple greedy fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        if not uncovered:
            return blocks
        while uncovered:
            best_block = None
            best_gain = -1
            # Heuristic: sample if too large, else exhaustive
            # For small v, exhaustive is fine.
            candidates = list(itertools.combinations(range(v), k))
            if len(candidates) > 50000:
                candidates = random.sample(candidates, 5000)
            
            for b in candidates:
                gain = sum(1 for ts in itertools.combinations(b, t) if ts in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is None or best_gain == 0:
                # Should not happen if valid params
                break
                
            blocks.append(list(best_block))
            uncovered.difference_update(itertools.combinations(best_block, t))
            
            if time.time() > deadline:
                break
        
        return blocks

    # --- Optimized Path for v=19, k=7, t=4 ---
    
    points = range(v)
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_id = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_blocks_tuples = list(itertools.combinations(points, k))
    num_blocks_total = len(all_blocks_tuples)
    
    # Map block tuple to index
    block_tuple_to_idx = {bt: i for i, bt in enumerate(all_blocks_tuples)}
    
    # Precompute t-subset indices for each block
    # block_tsubs[bi] = list of t-subset indices covered by block bi
    block_tsubs = [None] * num_blocks_total
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(num_tsubs)]
    
    for bi, bt in enumerate(all_blocks_tuples):
        tsubs = [tsub_to_id[ts] for ts in itertools.combinations(bt, t)]
        block_tsubs[bi] = tsubs
        for t_idx in tsubs:
            tsub_blocks[t_idx].append(bi)
    
    # Convert to numpy arrays for faster operations where possible
    # But lists of lists are okay for 50k blocks * 35 items.
    
    # --- Phase 1: Primal-Dual Style Greedy Initialization ---
    # Instead of pure max-gain, we use a weighted gain to break ties and explore slightly different landscapes.
    # Weight = 1 / (number of blocks covering this t-subset) initially, updated as we go?
    # Actually, standard primal-dual increases duals for uncovered items.
    # Let's stick to a robust "Greedy with Randomized Tie-breaking" to get a good starting point.
    
    solution = []
    in_solution = [False] * num_blocks_total
    cover_count = np.zeros(num_tsubs, dtype=np.int32)
    uncovered_mask = np.ones(num_tsubs, dtype=bool)
    num_uncovered = num_tsubs
    
    rng = random.Random(42)
    
    while num_uncovered > 0:
        best_block_idx = -1
        best_gain = -1
        best_score = -1.0
        
        # Sample a subset of blocks if we are deep in the process to speed up?
        # No, 50k is small. Iterate all.
        for bi in range(num_blocks_total):
            if in_solution[bi]:
                continue
            
            gain = 0
            # Check how many uncovered tsubs this block covers
            # Using numpy slice for speed?
            # t_idxs = block_tsubs[bi]
            # gain = np.sum(uncovered_mask[t_idxs])
            
            # Python loop might be slower than numpy slice
            t_idxs = block_tsubs[bi]
            # Since t_idxs is a list, we can't directly index numpy array with list of ints efficiently without conversion?
            # Actually, numpy handles list indexing well.
            gain = np.sum(uncovered_mask[t_idxs])
            
            if gain > best_gain:
                best_gain = gain
                # Add a small random noise to break ties and avoid deterministic local minima
                best_score = gain + rng.random() * 0.1
                best_block_idx = bi
            elif gain == best_gain and best_block_idx != -1:
                score = gain + rng.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_block_idx = bi
        
        if best_block_idx == -1 or best_gain == 0:
            break
            
        # Add block
        solution.append(best_block_idx)
        in_solution[best_block_idx] = True
        
        # Update counts
        t_idxs = block_tsubs[best_block_idx]
        cover_count[t_idxs] += 1
        uncovered_mask[t_idxs] = False
        num_uncovered -= best_gain
        
        if time.time() > deadline - 20:
            break

    # --- Phase 2: Local Search (Remove, Swap, Add) ---
    # We will perform a series of passes.
    # 1. Remove redundant blocks.
    # 2. Try swaps (remove A, add B) where B covers all essential tsubs of A and has better "value" (e.g., covers more other uncovered or is just a valid swap).
    #    Since we want to minimize count, swaps keep count same. They help escape local optima for removals.
    # 3. Try to add a block if it allows removing multiple others? (Complex)
    #    Simpler: Just iterate Remove and Swap.
    
    best_solution = solution.copy()
    best_len = len(solution)
    no_improvement_count = 0
    
    while time.time() < deadline - 5:
        improved = False
        
        # Pass 1: Remove redundant blocks
        # Iterate in random order
        sol_copy = solution.copy()
        rng.shuffle(sol_copy)
        
        for bi in sol_copy:
            if bi not in solution:
                continue
            # Check if block bi is redundant
            # It is redundant if all its tsubs are covered by other blocks
            t_idxs = block_tsubs[bi]
            # cover_count includes bi itself. So we check if cover_count[tid] > 1 for all tid in t_idxs
            if np.all(cover_count[t_idxs] > 1):
                # Remove it
                solution.remove(bi)
                in_solution[bi] = False
                cover_count[t_idxs] -= 1
                improved = True
                no_improvement_count = 0
                if len(solution) < best_len:
                    best_len = len(solution)
                    best_solution = solution.copy()
                break # Restart loop after change
        
        if improved:
            continue
            
        # Pass 2: Try Swaps
        # For each block A in solution, find block B not in solution that can replace A.
        # B can replace A if B covers all tsubs that are uniquely covered by A (essential tsubs).
        # Essential tsubs of A: t_idx in block_tsubs[A] where cover_count[t_idx] == 1
        
        sol_copy = solution.copy()
        rng.shuffle(sol_copy)
        
        swap_done = False
        for A_idx in sol_copy:
            if A_idx not in solution:
                continue
                
            t_idxs_A = block_tsubs[A_idx]
            essential_mask = (cover_count[t_idxs_A] == 1)
            essential_t_idxs = t_idxs_A[essential_mask]
            
            if len(essential_t_idxs) == 0:
                # A is redundant, should have been removed in Pass 1.
                continue
            
            # Find candidates B that cover all essential_t_idxs
            # Candidate B must cover all t in essential_t_idxs.
            # We can find intersection of tsub_blocks for each essential t.
            candidates = None
            for t_idx in essential_t_idxs:
                blocks_covering = tsub_blocks[t_idx]
                if candidates is None:
                    candidates = set(blocks_covering)
                else:
                    candidates.intersection_update(blocks_covering)
                
                if not candidates:
                    break
            
            if not candidates:
                continue
            
            # Filter out blocks already in solution
            candidates = [b for b in candidates if not in_solution[b]]
            
            if not candidates:
                continue
            
            # Pick a random candidate to try swapping
            # We want a swap that might allow future removals.
            # Heuristic: Pick B that covers the most OTHER tsubs (not essential to A) that are currently covered exactly once?
            # Or just random.
            rng.shuffle(candidates)
            
            for B_idx in candidates[:5]:
                # Check if swap is valid (it is by construction of candidates)
                # Perform swap
                # Remove A
                solution.remove(A_idx)
                in_solution[A_idx] = False
                cover_count[t_idxs_A] -= 1
                
                # Add B
                t_idxs_B = block_tsubs[B_idx]
                cover_count[t_idxs_B] += 1
                in_solution[B_idx] = True
                solution.append(B_idx)
                
                improved = True
                swap_done = True
                break
            
            if swap_done:
                break
        
        if improved:
            no_improvement_count = 0
            # After a swap, we might have created redundancies, so loop back to Pass 1
            continue
        else:
            no_improvement_count += 1
            if no_improvement_count > 5:
                # Stagnant, break or randomize?
                # Let's try a few random restarts of the local search phase?
                # Or just break.
                break
    
    # Use the best solution found
    final_indices = best_solution
    final_blocks = [list(all_blocks_tuples[bi]) for bi in final_indices]
    
    # Final Safety Check: Ensure validity
    # If invalid, we must repair.
    # Check if all tsubs are covered.
    # Reconstruct cover count for final solution?
    # It's safer to just verify.
    covered = np.zeros(num_tsubs, dtype=np.int32)
    for bi in final_indices:
        covered[block_tsubs[bi]] += 1
    
    if np.any(covered == 0):
        # Repair
        missing_tsubs = np.where(covered == 0)[0]
        for t_idx in missing_tsubs:
            # Find a block covering t_idx
            candidates = tsub_blocks[t_idx]
            # Pick one not in solution? Or just pick one.
            # If we pick one in solution, it's already there, so cover count would be >0.
            # So pick any.
            bi = candidates[0]
            if bi not in final_indices:
                final_indices.append(bi)
                covered[block_tsubs[bi]] += 1
            else:
                # Should not happen if logic is correct, but force add if needed?
                # If it's in solution, covered[t_idx] should be > 0.
                # This branch is unlikely.
                pass
        
        # Re-verify
        covered = np.zeros(num_tsubs, dtype=np.int32)
        for bi in final_indices:
            covered[block_tsubs[bi]] += 1
        if np.any(covered == 0):
             # Fallback: return a known valid large covering?
             # Just add random blocks until covered.
             while np.any(covered == 0):
                 bi = rng.randint(0, num_blocks_total - 1)
                 if bi not in final_indices:
                     final_indices.append(bi)
                     covered[block_tsubs[bi]] += 1
                     
        final_blocks = [list(all_blocks_tuples[bi]) for bi in final_indices]
        
    return final_blocks
