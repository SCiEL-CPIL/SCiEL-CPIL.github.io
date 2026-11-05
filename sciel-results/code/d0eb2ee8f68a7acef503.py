import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s kill

    # 1. Precomputation
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    id_to_triplet = [None] * n_triplets
    for i, trip in enumerate(all_triplets):
        id_to_triplet[i] = trip

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute triplet IDs for each block
    block_triplet_ids = []
    for block in all_blocks:
        ids = [triplet_to_id[trip] for trip in itertools.combinations(block, t)]
        block_triplet_ids.append(ids)
        
    # Precompute point sets for each block for faster intersection
    block_point_sets = [set(block) for block in all_blocks]

    # Index: point -> list of block indices
    point_to_blocks = defaultdict(list)
    for bi, block in enumerate(all_blocks):
        for p in block:
            point_to_blocks[p].append(bi)

    def get_blocks_containing(points):
        """Return list of block indices containing all points in the set."""
        pts = list(points)
        if len(pts) == 0:
            return list(range(n_blocks))
        if len(pts) > k:
            return []
        first = pts[0]
        candidate_set = set(point_to_blocks[first])
        for p in pts[1:]:
            candidate_set &= set(point_to_blocks[p])
            if not candidate_set:
                return []
        return list(candidate_set)

    # 2. Greedy Construction
    def greedy_construct(seed):
        random.seed(seed)
        # Use bitmask for uncovered triplets
        uncovered = (1 << n_triplets) - 1
        selected_indices = []
        used_blocks = set()
        
        while uncovered:
            # Pick a random uncovered triplet
            # Find a set bit
            # Efficient way: pick random index, check bit. If not set, try again.
            # Or use a list of uncovered indices, but updating it is expensive.
            # Given n_triplets is ~2000, a few random tries are fine.
            idx = -1
            for _ in range(10):
                cand = random.randint(0, n_triplets - 1)
                if (uncovered >> cand) & 1:
                    idx = cand
                    break
            if idx == -1:
                # Fallback: find first set bit
                lsb = uncovered & -uncovered
                idx = lsb.bit_length() - 1
            
            start_triplet = id_to_triplet[idx]
            
            # Find best block containing start_triplet
            # Candidates: blocks containing all 3 points of start_triplet
            candidate_indices = get_blocks_containing(start_triplet)
            
            best_block_idx = -1
            best_gain = 0
            best_mask = 0
            
            # Precompute mask for candidates? No, just calculate gain on the fly.
            # Gain = number of bits in (block_mask & uncovered)
            
            for bi in candidate_indices:
                if bi in used_blocks:
                    # We can reuse blocks? The problem says "blocks must be distinct".
                    # So we cannot pick the same block twice.
                    continue
                
                # Calculate gain
                # We need the mask of this block.
                # We didn't precompute masks as integers because 2^2024 is huge.
                # But we can calculate gain by iterating through its triplets.
                gain = 0
                for tid in block_triplet_ids[bi]:
                    if (uncovered >> tid) & 1:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = bi
                    best_mask = None # We don't need the mask, just the IDs to update uncovered
            
            if best_block_idx == -1:
                # No new block found? This shouldn't happen if covering is possible and we haven't used all blocks.
                # If it happens, break to avoid infinite loop.
                break

            selected_indices.append(best_block_idx)
            used_blocks.add(best_block_idx)
            
            # Update uncovered
            for tid in block_triplet_ids[best_block_idx]:
                uncovered &= ~(1 << tid)

        return [all_blocks[i] for i in selected_indices]

    # 3. Simulated Annealing Optimizer
    def optimize_sa(blocks, time_limit):
        n = len(blocks)
        # Convert to indices for faster lookup
        block_indices = [all_blocks.index(b) for b in blocks]
        
        # Initialize counts
        counts = np.zeros(n_triplets, dtype=np.int32)
        for bi in block_indices:
            for tid in block_triplet_ids[bi]:
                counts[tid] += 1
        
        best_blocks = list(block_indices)
        best_len = len(block_indices)
        
        T = 1.0
        T_end = 0.01
        cooling_rate = 0.995
        
        current_len = len(block_indices)
        
        while time.time() < time_limit and T > T_end:
            # Pick a move
            move_type = random.random()
            if move_type < 0.2:
                move = 'remove'
            elif move_type < 0.7:
                move = 'swap'
            else:
                move = '2for1'
                
            old_len = current_len
            accepted = False
            delta_e = 0
            
            if move == 'remove' and current_len > 0:
                i = random.randint(0, current_len - 1)
                bi = block_indices[i]
                # Check if redundant
                redundant = True
                for tid in block_triplet_ids[bi]:
                    if counts[tid] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove
                    for tid in block_triplet_ids[bi]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    current_len -= 1
                    delta_e = -1 # Improvement
                    accepted = True
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(block_indices)
                        
            elif move == 'swap' and current_len > 0:
                i = random.randint(0, current_len - 1)
                bi = block_indices[i]
                # Find unique triplets of bi
                unique_tids = []
                for tid in block_triplet_ids[bi]:
                    if counts[tid] == 1:
                        unique_tids.append(tid)
                
                if not unique_tids:
                    # Redundant, treat as remove
                    for tid in block_triplet_ids[bi]:
                        counts[tid] -= 1
                    block_indices.pop(i)
                    current_len -= 1
                    delta_e = -1
                    accepted = True
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(block_indices)
                else:
                    # Find points in unique triplets
                    points_needed = set()
                    for tid in unique_tids:
                        points_needed.update(id_to_triplet[tid])
                    
                    if len(points_needed) <= k:
                        candidates = get_blocks_containing(points_needed)
                        # Filter out current block and check if it covers all unique_tids
                        valid_candidates = []
                        for cand_bi in candidates:
                            if cand_bi == bi:
                                continue
                            # Check if cand covers all unique_tids
                            cand_set = set(block_triplet_ids[cand_bi])
                            if all(tid in cand_set for tid in unique_tids):
                                valid_candidates.append(cand_bi)
                                
                        if valid_candidates:
                            new_bi = random.choice(valid_candidates)
                            # Swap
                            # Remove old bi
                            for tid in block_triplet_ids[bi]:
                                counts[tid] -= 1
                            # Add new bi
                            for tid in block_triplet_ids[new_bi]:
                                counts[tid] += 1
                            block_indices[i] = new_bi
                            # Length doesn't change
                            delta_e = 0
                            accepted = True
                            # We don't update best_len here unless we have a tie-breaker, 
                            # but since len is same, we just keep exploring.
                            # However, if we find a better arrangement, we might want to track it.
                            # For simplicity, we only update best if len decreases.
                            # But maybe we should update best if we find a "more robust" solution?
                            # No, the objective is just min blocks.
                            
            elif move == '2for1' and current_len >= 2:
                i, j = random.sample(range(current_len), 2)
                if i > j: i, j = j, i
                bi1 = block_indices[i]
                bi2 = block_indices[j]
                
                # Find triplets covered ONLY by bi1 or bi2
                # A triplet is "unique to the pair" if count==1 and it is in bi1 or bi2
                # Actually, we need to find triplets that will lose coverage if we remove bi1 and bi2.
                # These are triplets where count==1 AND (tid in bi1 OR tid in bi2).
                # Wait, if count==1, it is covered by exactly one block. 
                # If that block is bi1 or bi2, then removing both would uncover it.
                
                unique_tids = []
                for tid in block_triplet_ids[bi1]:
                    if counts[tid] == 1:
                        unique_tids.append(tid)
                for tid in block_triplet_ids[bi2]:
                    if counts[tid] == 1:
                        unique_tids.append(tid)
                        
                # Note: A triplet cannot be in both bi1 and bi2 if count==1? 
                # Yes, if it were in both, count would be at least 2.
                # So the lists are disjoint.
                
                if not unique_tids:
                    # Both redundant
                    for tid in block_triplet_ids[bi1]:
                        counts[tid] -= 1
                    for tid in block_triplet_ids[bi2]:
                        counts[tid] -= 1
                    block_indices.pop(j)
                    block_indices.pop(i)
                    current_len -= 2
                    delta_e = -2
                    accepted = True
                    if current_len < best_len:
                        best_len = current_len
                        best_blocks = list(block_indices)
                else:
                    points_needed = set()
                    for tid in unique_tids:
                        points_needed.update(id_to_triplet[tid])
                    
                    if len(points_needed) <= k:
                        candidates = get_blocks_containing(points_needed)
                        valid_candidates = []
                        for cand_bi in candidates:
                            if cand_bi == bi1 or cand_bi == bi2:
                                continue
                            cand_set = set(block_triplet_ids[cand_bi])
                            if all(tid in cand_set for tid in unique_tids):
                                valid_candidates.append(cand_bi)
                                
                        if valid_candidates:
                            new_bi = random.choice(valid_candidates)
                            # Remove bi1, bi2
                            for tid in block_triplet_ids[bi1]:
                                counts[tid] -= 1
                            for tid in block_triplet_ids[bi2]:
                                counts[tid] -= 1
                            # Add new_bi
                            for tid in block_triplet_ids[new_bi]:
                                counts[tid] += 1
                            
                            # Update indices
                            # Remove j first (higher index)
                            block_indices.pop(j)
                            block_indices.pop(i)
                            block_indices.append(new_bi)
                            current_len -= 1
                            delta_e = -1
                            accepted = True
                            if current_len < best_len:
                                best_len = current_len
                                best_blocks = list(block_indices)
            
            if not accepted:
                # No move made, or move rejected? 
                # In this implementation, we only attempt moves that are structurally valid.
                # We don't explicitly reject based on T yet. 
                # We should implement acceptance probability.
                pass
                
            # Simulated Annealing Acceptance
            # If delta_e < 0, always accept.
            # If delta_e > 0, accept with prob exp(-delta_e / T).
            # If delta_e == 0, accept (to move around).
            
            # In the code above, I only implemented "improving" moves (delta_e <= 0).
            # To escape local minima, I need to allow worsening moves.
            # Let's refactor the move execution to calculate delta_e and then decide.
            
            # Actually, the logic above modifies the state immediately.
            # I need to undo if rejected.
            # This is complex. 
            # Alternative: Only do improving moves? That's just Local Search.
            # To do SA, I must allow worsening moves.
            
            # Let's add a "worsening" move: Add a random block.
            # This increases length by 1.
            
            if move == 'add' or (random.random() < 0.1 and current_len < n_blocks):
                 # Add a random block
                 # This is a worsening move (delta_e = +1)
                 # Accept with prob exp(-1/T)
                 if random.random() < np.exp(-1.0 / T):
                     # Add random unused block? Or any block?
                     # Adding a duplicate is not allowed.
                     # Find an unused block.
                     used_set = set(block_indices)
                     available = [i for i in range(n_blocks) if i not in used_set]
                     if available:
                         new_bi = random.choice(available)
                         for tid in block_triplet_ids[new_bi]:
                             counts[tid] += 1
                         block_indices.append(new_bi)
                         current_len += 1
                         delta_e = 1
                         # Don't update best
                         
            # Cooling
            T *= cooling_rate
            
        # Final check and return best
        return [all_blocks[i] for i in best_blocks]

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    n_restarts = 5
    for seed in range(n_restarts):
        if time.time() > deadline - 20:
            break
            
        init_blocks = greedy_construct(seed)
        if len(init_blocks) < best_len:
            best_len = len(init_blocks)
            best_blocks = init_blocks
            
        # Optimize
        opt_blocks = optimize_sa(init_blocks, deadline - 5)
        
        # Verify validity
        counts = np.zeros(n_triplets, dtype=np.int32)
        for b in opt_blocks:
            for triplet in itertools.combinations(b, t):
                counts[triplet_to_id[triplet]] += 1
        if np.min(counts) > 0:
            if len(opt_blocks) < best_len:
                best_len = len(opt_blocks)
                best_blocks = opt_blocks

    if best_blocks is None:
        best_blocks = greedy_construct(0)

    return list(best_blocks)

if __name__ == "__main__":
    pass
