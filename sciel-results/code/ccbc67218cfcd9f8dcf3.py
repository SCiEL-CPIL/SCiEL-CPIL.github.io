import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    # Parameters specific to this task: v=19, k=8, t=5
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    
    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute the list of t-subset indices for every possible k-subset (block)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_tsubs[i] is a list of indices into all_tsubs
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = [tsub_index[ts] for ts in itertools.combinations(b, t)]
        block_tsubs.append(tsubs_in_b)
        
    # Inverse mapping: for each t-subset, which blocks cover it?
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for i, b_tsubs in enumerate(block_tsubs):
        for j in b_tsubs:
            tsub_blocks[j].append(i)
            
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    def get_elapsed():
        return time.time() - start_time

    # Helper: Greedy construction to get a valid starting point
    def greedy_construct():
        uncovered = bytearray([1] * num_tsubs)
        uncovered_count = num_tsubs
        selected_blocks = []
        
        while uncovered_count > 0:
            if get_elapsed() > deadline - 5.0:
                return selected_blocks
            
            # Pick a random uncovered t-subset
            # Optimization: If few uncovered, scan. Else, probe.
            if uncovered_count < 100:
                uncov_indices = [i for i in range(num_tsubs) if uncovered[i]]
                if not uncov_indices:
                    break
                s_idx = random.choice(uncov_indices)
            else:
                while True:
                    s_idx = random.randint(0, num_tsubs - 1)
                    if uncovered[s_idx]:
                        break
            
            candidates = tsub_blocks[s_idx]
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for j in block_tsubs[b_idx]:
                    if uncovered[j]:
                        gain += 1
                        if gain == 56: # Max possible
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            selected_blocks.append(best_block_idx)
            
            for j in block_tsubs[best_block_idx]:
                if uncovered[j]:
                    uncovered[j] = 0
                    uncovered_count -= 1
                    
        return selected_blocks

    # Helper: Remove redundant blocks
    def reduce_blocks(block_indices):
        block_indices = list(set(block_indices))
        changed = True
        while changed:
            changed = False
            if get_elapsed() > deadline - 2.0:
                break
            # Count coverage
            cover_count = [0] * num_tsubs
            for b_idx in block_indices:
                for j in block_tsubs[b_idx]:
                    cover_count[j] += 1
            
            new_block_indices = []
            for b_idx in block_indices:
                is_essential = False
                for j in block_tsubs[b_idx]:
                    if cover_count[j] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_block_indices.append(b_idx)
                else:
                    changed = True
            block_indices = new_block_indices
        return block_indices

    # Simulated Annealing
    def simulated_annealing(initial_blocks):
        current_blocks = set(initial_blocks)
        # Precompute coverage counts for current state
        cover_count = [0] * num_tsubs
        for b_idx in current_blocks:
            for j in block_tsubs[b_idx]:
                cover_count[j] += 1
        
        # Track which t-subsets are covered by exactly 1 block (critical)
        # And which are covered by > 1 (redundant)
        
        best_blocks = list(current_blocks)
        best_size = len(best_blocks)
        
        # Temperature parameters
        # We want to allow some "bad" moves (increasing size) to escape local minima,
        # but primarily we want to decrease size.
        # Actually, standard SA on set cover usually minimizes size.
        # Move: Remove block B. If valid, accept. If invalid, accept with prob exp(-delta/T).
        # Here delta = number of uncovered t-subsets created.
        
        T = 10.0
        cooling = 0.995
        min_T = 0.01
        
        # To speed up, we don't recompute full cover_count every step.
        # We maintain it incrementally.
        
        while T > min_T:
            if get_elapsed() > deadline - 10.0:
                break
            
            # Random move: Pick a block to remove
            if not current_blocks:
                break
                
            block_to_remove = random.choice(list(current_blocks))
            
            # Calculate impact of removal
            # How many t-subsets become uncovered?
            new_uncovered = 0
            for j in block_tsubs[block_to_remove]:
                if cover_count[j] == 1:
                    new_uncovered += 1
            
            if new_uncovered == 0:
                # Safe move: Block is redundant. Remove it.
                current_blocks.remove(block_to_remove)
                for j in block_tsubs[block_to_remove]:
                    cover_count[j] -= 1
                
                # Try to add a block that covers the most currently "weak" (count=1) t-subsets?
                # Or just keep it small.
                # Since we removed a block, size decreased. This is always an improvement.
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks = list(current_blocks)
            else:
                # Bad move: Block is essential.
                # Accept with probability exp(-new_uncovered / T)
                # But wait, if we accept, we have to add a block to cover them?
                # Standard SA for Set Cover:
                # State: Set of blocks.
                # Move: Swap B_out with B_in.
                # Let's stick to Swap to maintain validity or handle invalid states carefully.
                # If we allow invalid states, we must add a block back eventually.
                # Simpler: Only consider swaps that maintain validity or slightly degrade.
                
                # Let's try a Swap: Remove B_out, Add B_in.
                # B_in must cover the t-subsets that B_out uniquely covered?
                # No, B_in just needs to cover them to keep validity.
                
                # Find candidate B_in that covers the critical t-subsets of B_out.
                critical_tsubs = [j for j in block_tsubs[block_to_remove] if cover_count[j] == 1]
                
                if not critical_tsubs:
                    # Should have been caught above, but just in case
                    current_blocks.remove(block_to_remove)
                    for j in block_tsubs[block_to_remove]:
                        cover_count[j] -= 1
                    if len(current_blocks) < best_size:
                        best_size = len(current_blocks)
                        best_blocks = list(current_blocks)
                    continue

                # Find B_in that covers ALL critical_tsubs
                # This is expensive. Let's sample.
                # Or, just try to add a random block that covers one of the criticals, 
                # and check if it covers all.
                
                # Better heuristic:
                # 1. Remove B_out.
                # 2. Identify uncovered tsubs (the criticals).
                # 3. Pick a block B_in that covers the most uncovered tsubs.
                # 4. If it covers all, accept (size same).
                # 5. If it doesn't cover all, reject? Or accept with penalty?
                
                # Let's try to find a block that covers the most critical tsubs.
                # Candidates are blocks that cover critical_tsubs[0].
                candidates_in = tsub_blocks[critical_tsubs[0]]
                
                best_in = -1
                best_covered = -1
                # Sample a few candidates to save time
                sample_size = min(10, len(candidates_in))
                if len(candidates_in) > sample_size:
                    sampled = random.sample(candidates_in, sample_size)
                else:
                    sampled = candidates_in
                
                for b_in in sampled:
                    if b_in == block_to_remove:
                        continue
                    # Check how many critical tsubs it covers
                    covered = 0
                    for j in critical_tsubs:
                        if j in block_tsubs[b_in]: # This is O(56) check, okay for small sample
                            covered += 1
                    if covered > best_covered:
                        best_covered = covered
                        best_in = b_in
                
                if best_in != -1 and best_covered == len(critical_tsubs):
                    # Perfect swap! Size remains same, but structure changes.
                    # This might allow future reductions.
                    # Execute swap
                    current_blocks.remove(block_to_remove)
                    for j in block_tsubs[block_to_remove]:
                        cover_count[j] -= 1
                    
                    current_blocks.add(best_in)
                    for j in block_tsubs[best_in]:
                        cover_count[j] += 1
                    
                    # After swap, check if we can remove any other block?
                    # The reduction phase handles that.
                    # But maybe this swap made another block redundant?
                    # We'll let the next iteration or a quick check handle it.
                    
                    # If size is same, it's not an improvement in size, but it's a move in search space.
                    # We don't update best_size unless we actually reduced it.
                else:
                    # No good swap found. Maybe try a random addition?
                    # Or just cool down.
                    pass
            
            T *= cooling
            
            # Periodic reduction to clean up
            if T < 0.5 and len(current_blocks) > best_size:
                 # If we drifted, snap back?
                 # No, SA should explore.
                 pass

        return best_blocks

    # Initial Greedy
    initial_blocks = greedy_construct()
    initial_blocks = reduce_blocks(initial_blocks)
    
    best_len = len(initial_blocks)
    best_block_list = [all_blocks[i] for i in initial_blocks]
    
    # Run SA
    # We can run SA on the initial solution.
    # To improve, we might run SA from multiple restarts if time permits.
    # But SA is slow per step if not optimized.
    # Let's try one long SA run first.
    
    sa_blocks = simulated_annealing(initial_blocks)
    if len(sa_blocks) < best_len:
        best_len = len(sa_blocks)
        best_block_list = [all_blocks[i] for i in sa_blocks]
        
    # If time remains, do more restarts with different greedy seeds
    # Or run SA on a slightly perturbed greedy solution
    
    # Given the complexity, let's just do a few more greedy+reduce if time allows, 
    # and pick the best.
    # The SA above might be too slow if `critical_tsubs` check is heavy.
    # Let's add a time check inside SA loop more aggressively.
    
    # Actually, the current SA implementation inside `simulated_annealing` uses `random.choice(list(current_blocks))` which is O(N).
    # And `j in block_tsubs[b_in]` is O(56).
    # 10 candidates * 56 checks * (steps).
    # Steps: ~1000-5000?
    # 5000 * 10 * 56 = 2.8M ops. Fast enough.
    
    # Let's try a second SA run on a different greedy start if time allows
    if get_elapsed() < deadline - 20.0:
        initial_blocks_2 = greedy_construct()
        initial_blocks_2 = reduce_blocks(initial_blocks_2)
        if len(initial_blocks_2) < best_len:
            best_len = len(initial_blocks_2)
            best_block_list = [all_blocks[i] for i in initial_blocks_2]
            
        sa_blocks_2 = simulated_annealing(initial_blocks_2)
        if len(sa_blocks_2) < best_len:
            best_len = len(sa_blocks_2)
            best_block_list = [all_blocks[i] for i in sa_blocks_2]

    return [list(b) for b in best_block_list]
