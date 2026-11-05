import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)
    
    # Precompute coverage of each block as a set of t-subset indices
    block_coverages = []
    for block in all_k_blocks:
        covers = [t_subset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_coverages.append(np.array(covers, dtype=np.int32))
        
    # Reverse index: t_subset_idx -> list of block indices covering it
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, cov in enumerate(block_coverages):
        for t_idx in cov:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    def greedy_initialization():
        """Fast greedy to get a valid starting solution."""
        covered = np.zeros(num_t_subsets, dtype=bool)
        current_blocks = []
        
        # Use a simple greedy: pick block with max new coverage
        # For speed, we don't use a heap here, just scan. 
        # Since we only need a decent start, this is fine.
        while not np.all(covered):
            best_block = -1
            best_gain = 0
            for b_idx in range(num_k_blocks):
                if b_idx in current_blocks:
                    continue
                cov = block_coverages[b_idx]
                gain = np.count_nonzero(~covered[cov])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block == -1:
                break
            current_blocks.append(best_block)
            covered[block_coverages[best_block]] = True
            if time.time() > deadline - 10.0:
                break
        return current_blocks

    def simulated_annealing():
        # Start with a greedy solution
        current_blocks = greedy_initialization()
        if not current_blocks:
            return []
            
        # Maintain counts
        counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b in current_blocks:
            counts[block_coverages[b]] += 1
            
        current_size = len(current_blocks)
        best_blocks = current_blocks.copy()
        best_size = current_size
        
        # Temperature schedule
        T_start = 1.0
        T_end = 0.01
        cooling_rate = 0.995
        
        T = T_start
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0 and T > T_end:
            # Generate a random move
            move_type = random.random()
            improved = False
            new_size = current_size
            
            if move_type < 0.4:
                # Try to remove a block
                if current_size <= 1:
                    T *= cooling_rate
                    continue
                b_idx = random.choice(current_blocks)
                cov = block_coverages[b_idx]
                # Check if any t-subset has count 1 in this block
                if np.any(counts[cov] == 1):
                    # Cannot remove
                    T *= cooling_rate
                    continue
                # Remove
                counts[cov] -= 1
                current_blocks.remove(b_idx)
                new_size = current_size - 1
                # This is always an improvement if valid
                best_blocks = current_blocks.copy()
                best_size = new_size
                no_improve_count = 0
                improved = True
            elif move_type < 0.8:
                # Try to add a block (might enable future removals, but increases size now)
                # This is a "worsening" move usually, but helps escape local minima
                # We accept it based on probability
                # Pick a random block not in current set
                # Optimization: Pick a random block index
                b_idx = random.randint(0, num_k_blocks - 1)
                if b_idx in current_blocks:
                    T *= cooling_rate
                    continue
                cov = block_coverages[b_idx]
                counts[cov] += 1
                current_blocks.append(b_idx)
                new_size = current_size + 1
                # Cost increase = 1
                # Acceptance: always accept if it leads to immediate deletion? No, we don't check that here.
                # Just accept based on SA
                # Delta = 1 (worse)
                if random.random() < np.exp(-1.0 / T):
                    # Accepted
                    pass
                else:
                    # Reject
                    counts[cov] -= 1
                    current_blocks.remove(b_idx)
                    T *= cooling_rate
                    continue
                # If accepted, we might have a chance to delete something else later
                # But for now, size increased.
                # We don't update best here.
                no_improve_count += 1
            else:
                # 2-for-1 Exchange: Remove A, B, Add C
                if current_size < 2:
                    T *= cooling_rate
                    continue
                # Pick A and B
                A = random.choice(current_blocks)
                B = random.choice([b for b in current_blocks if b != A])
                cov_A = block_coverages[A]
                cov_B = block_coverages[B]
                
                # Determine must-cover set for C
                # Must cover:
                # 1. Unique to A (count 1 in A)
                # 2. Unique to B (count 1 in B)
                # 3. Shared by A and B with count 2
                
                # Efficient calculation of must_cover
                # We need a mask
                must_cover = np.zeros(num_t_subsets, dtype=bool)
                
                # Unique to A
                mask_A = counts[cov_A] == 1
                must_cover[cov_A[mask_A]] = True
                
                # Unique to B
                mask_B = counts[cov_B] == 1
                must_cover[cov_B[mask_B]] = True
                
                # Shared with count 2
                # Intersection of A and B
                # We can use the fact that if t is in A and B, and counts[t] == 2, it must be covered by C
                # Iterate through cov_A and check if in cov_B?
                # Or use set logic?
                # Let's use a temporary boolean array for B's coverage
                in_B = np.zeros(num_t_subsets, dtype=bool)
                in_B[cov_B] = True
                # Intersection
                shared = cov_A[in_B[cov_A]]
                # Filter those with count 2
                shared_count_2 = shared[counts[shared] == 2]
                if len(shared_count_2) > 0:
                    must_cover[shared_count_2] = True
                    
                if not np.any(must_cover):
                    # No constraints? Then any C works? 
                    # If A and B are not critical, we could just remove them.
                    # But we are in the exchange branch.
                    # If must_cover is empty, it means removing A and B doesn't uncover anything.
                    # Then we can just remove them (size -2). 
                    # But we are doing a 2-for-1, so we add C.
                    # Actually, if must_cover is empty, we can just remove A and B and NOT add C.
                    # That would be a size -2 move. 
                    # Let's handle this as a deletion of two blocks.
                    counts[cov_A] -= 1
                    counts[cov_B] -= 1
                    current_blocks.remove(A)
                    current_blocks.remove(B)
                    new_size = current_size - 2
                    best_blocks = current_blocks.copy()
                    best_size = new_size
                    no_improve_count = 0
                    improved = True
                    T *= cooling_rate
                    continue

                # Find candidates C
                must_cover_list = np.where(must_cover)[0]
                if len(must_cover_list) == 0:
                    T *= cooling_rate
                    continue
                    
                # Sort by candidate list size to speed up intersection
                must_cover_list = must_cover_list[np.argsort([len(t_sub_to_blocks[t]) for t in must_cover_list])]
                
                candidates_set = set(t_sub_to_blocks[must_cover_list[0]])
                candidates_set.discard(A)
                candidates_set.discard(B)
                
                for t_idx in must_cover_list[1:]:
                    if not candidates_set:
                        break
                    candidates_set.intersection_update(t_sub_to_blocks[t_idx])
                    candidates_set.discard(A)
                    candidates_set.discard(B)
                    
                if not candidates_set:
                    T *= cooling_rate
                    continue
                
                C = random.choice(list(candidates_set))
                cov_C = block_coverages[C]
                
                # Apply move: Remove A, B; Add C
                # Size change: -1
                # Is it valid? Yes, by construction C covers all must_cover.
                # Are there other t-subsets affected?
                # T-subsets in A or B that are NOT in must_cover:
                # - Unique to A (count 1): Covered by must_cover.
                # - Unique to B (count 1): Covered by must_cover.
                # - Shared count 2: Covered by must_cover.
                # - Shared count > 2: Count decreases by 2, still >= 1. OK.
                # - Unique to A but count > 1? Impossible, unique means count 1.
                # So the move is valid and reduces size by 1.
                
                counts[cov_A] -= 1
                counts[cov_B] -= 1
                counts[cov_C] += 1
                
                current_blocks.remove(A)
                current_blocks.remove(B)
                current_blocks.append(C)
                
                new_size = current_size - 1
                best_blocks = current_blocks.copy()
                best_size = new_size
                no_improve_count = 0
                improved = True
                
            if not improved and not (move_type >= 0.8 and move_type < 0.9): # Approx logic for rejection
                # If we didn't improve and didn't accept a worsening move, count no-improve
                if not (move_type >= 0.8 and move_type < 0.9): 
                    pass # Logic above handled rejection
                    
            # Update temperature
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                # Reset temperature or break?
                # Let's just cool down faster
                T *= 0.5
                no_improve_count = 0
                
        return best_blocks

    # Run SA
    best_solution = simulated_annealing()
    
    if not best_solution:
        # Fallback
        best_solution = greedy_initialization()
        
    return [list(all_k_blocks[b_idx]) for b_idx in best_solution]
