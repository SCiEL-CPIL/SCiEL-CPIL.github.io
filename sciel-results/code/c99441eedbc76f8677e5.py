import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 140  # Safety margin
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute block subsets (indices)
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute blocks covering each subset
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # Precompute bitsets for fast coverage check
    # Each bitset is a Python int. Bit i is 1 if block covers subset i.
    block_bitsets = [0] * N_blocks
    for i, subs in enumerate(block_subsets):
        mask = 0
        for s in subs:
            mask |= (1 << s)
        block_bitsets[i] = mask

    def run_greedy(randomize_ties=False):
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)
        selected_blocks = []
        selected_set = set()
        
        while np.any(cover_count == 0):
            if time.time() - start_time > timeout:
                return None, None
                
            max_pot = np.max(potential)
            if max_pot == 0:
                break
                
            if randomize_ties:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential)
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1
                        
        # Redundancy removal
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
        return selected_blocks, cover_count

    def local_search(selected_blocks, cover_count):
        selected_set = set(selected_blocks)
        best_len = len(selected_blocks)
        best_blocks = list(selected_blocks)
        
        # Local search loop
        # We try to swap blocks to enable deletions.
        # To avoid infinite loops on neutral moves, we can limit steps or use a random walk.
        
        steps = 0
        max_steps = 50000  # Limit per local search phase
        no_improve_count = 0
        max_no_improve = 2000
        
        while steps < max_steps and (time.time() - start_time < timeout - 5):
            steps += 1
            if not selected_blocks:
                break
                
            # Pick a random block to try to remove/swap
            b_out_idx = random.randint(0, len(selected_blocks) - 1)
            b_out = selected_blocks[b_out_idx]
            
            # Find subsets covered ONLY by b_out
            critical_subsets = []
            for s in block_subsets[b_out]:
                if cover_count[s] == 1:
                    critical_subsets.append(s)
            
            if not critical_subsets:
                # Redundant, just remove
                for s in block_subsets[b_out]:
                    cover_count[s] -= 1
                selected_blocks.pop(b_out_idx)
                selected_set.remove(b_out)
                best_len = len(selected_blocks)
                best_blocks = list(selected_blocks)
                no_improve_count = 0
            else:
                # Need to find a block that covers all critical_subsets
                # Use bitset check
                critical_mask = 0
                for s in critical_subsets:
                    critical_mask |= (1 << s)
                
                # Candidate blocks must cover the first critical subset
                first_crit = critical_subsets[0]
                candidates = blocks_covering_subset[first_crit]
                
                found_swap = False
                # Shuffle candidates to randomize
                random.shuffle(candidates)
                
                for b_in in candidates:
                    if b_in in selected_set:
                        continue
                    
                    # Check if b_in covers all critical subsets
                    if (block_bitsets[b_in] & critical_mask) == critical_mask:
                        # Perform swap
                        # Remove b_out
                        for s in block_subsets[b_out]:
                            cover_count[s] -= 1
                        selected_blocks.pop(b_out_idx)
                        selected_set.remove(b_out)
                        
                        # Add b_in
                        selected_blocks.append(b_in)
                        selected_set.add(b_in)
                        for s in block_subsets[b_in]:
                            cover_count[s] += 1
                        
                        found_swap = True
                        break
                
                if found_swap:
                    # Try to remove redundant blocks
                    changed = True
                    while changed:
                        changed = False
                        for i in range(len(selected_blocks) - 1, -1, -1):
                            b_idx = selected_blocks[i]
                            is_redundant = True
                            for s in block_subsets[b_idx]:
                                if cover_count[s] == 1:
                                    is_redundant = False
                                    break
                            if is_redundant:
                                for s in block_subsets[b_idx]:
                                    cover_count[s] -= 1
                                selected_blocks.pop(i)
                                selected_set.remove(b_idx)
                                changed = True
                                if len(selected_blocks) < best_len:
                                    best_len = len(selected_blocks)
                                    best_blocks = list(selected_blocks)
                                    no_improve_count = 0
                    if len(selected_blocks) >= best_len:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                break
                
        return best_blocks, best_len

    # Main execution
    best_blocks_global = None
    best_len_global = float('inf')
    
    # Run a few greedy restarts to find a good starting point
    # Then do local search on the best one.
    
    # 1. Deterministic greedy
    blocks, counts = run_greedy(randomize_ties=False)
    if blocks:
        best_blocks_local, best_len_local = local_search(blocks, counts)
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local
            
    # 2. Randomized greedy restarts
    # Estimate time left
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
        # Do a few more restarts
        for i in range(5):
            if time.time() - start_time > timeout - 10:
                break
            blocks, counts = run_greedy(randomize_ties=True)
            if blocks:
                # Quick local search on this one?
                # Or just keep the best greedy?
                # Local search is expensive. Let's do a short local search here.
                # Or just check length after redundancy removal.
                # The local_search function inside handles the optimization.
                # To save time, we can do a shorter local search for intermediate runs.
                # But for simplicity, let's just run the full local search if time permits.
                # Given 150s, and greedy+redundancy is fast, we can afford a few.
                
                # Let's modify local_search to accept a time limit or step limit.
                # For now, the step limit in local_search is 50000.
                # Let's just run it.
                
                best_blocks_local, best_len_local = local_search(blocks, counts)
                if best_len_local < best_len_global:
                    best_len_global = best_len_local
                    best_blocks_global = best_blocks_local

    # If we still have time, do more local search on the best global solution
    # This requires re-running local search on the best found so far?
    # The local_search function modifies the input lists.
    # We should probably save the state and re-run if we want to continue.
    # However, the local_search function returns the best found *during* that run.
    # If we want to continue searching, we need to restart local search from the best state.
    
    # Let's assume the local_search within the loop is sufficient.
    # To be safe, let's run one final long local search on the best global blocks if time remains.
    
    if best_blocks_global and time.time() - start_time < timeout - 5:
        # Reconstruct cover_count for the best global blocks
        cover_count_final = np.zeros(N_subsets, dtype=np.int32)
        for b in best_blocks_global:
            for s in block_subsets[b]:
                cover_count_final[s] += 1
        
        # Run local search again, potentially with higher step limit
        # We need a version of local_search that can run for a specific duration
        # For now, we just call it again. It will stop if no improvement or time runs out.
        # To avoid infinite loops if it's stuck, the no_improve_count handles it.
        
        # Let's just call it.
        best_blocks_final, best_len_final = local_search(list(best_blocks_global), cover_count_final)
        if best_len_final < best_len_global:
            best_len_global = best_len_final
            best_blocks_global = best_blocks_final

    if not best_blocks_global:
        # Fallback: return the first greedy result if something went wrong
        blocks, _ = run_greedy(randomize_ties=False)
        best_blocks_global = blocks

    return [list(all_blocks[i]) for i in best_blocks_global]
