import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin

    # Precompute all blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subsets for each block, and map t-subsets to blocks
    # For efficiency, represent t-subsets as sorted tuples and use a dictionary
    # But for speed, let's use a different approach: bitmasks or just direct sets
    # Since v=19, k=7, t=4, C(7,4)=35 t-subsets per block, C(19,4)=3876 total t-subsets
    
    # Map each point to a bitmask or just use tuples
    block_tsubs = []
    for b in all_blocks:
        tsubs = list(itertools.combinations(b, t))
        block_tsubs.append(tsubs)
    
    # Build reverse map: t-subset -> list of block indices containing it
    tsub_to_blocks = {}
    for i, tsubs in enumerate(block_tsubs):
        for ts in tsubs:
            if ts not in tsub_to_blocks:
                tsub_to_blocks[ts] = []
            tsub_to_blocks[ts].append(i)
    
    # Phase 1: Standard Greedy Set Cover
    # gains[i] = number of currently uncovered t-subsets in block i
    # Initially all t-subsets are uncovered
    initial_gain = len(block_tsubs[0])  # C(k, t)
    gains = np.full(num_blocks, initial_gain, dtype=np.int32)
    
    coverage_count = {}  # t-subset -> count of selected blocks covering it
    selected_indices = []
    
    while time.time() < deadline - 60:  # Reserve 60s for local search
        max_gain = np.max(gains)
        if max_gain <= 0:
            break
        best_idx = int(np.argmax(gains))
        selected_indices.append(best_idx)
        
        for ts in block_tsubs[best_idx]:
            if ts not in coverage_count:
                coverage_count[ts] = 0
            coverage_count[ts] += 1
            
            if coverage_count[ts] == 1:
                for b_idx in tsub_to_blocks[ts]:
                    gains[b_idx] -= 1
    
    # If greedy didn't finish, try to complete with a faster method
    # But for v=19, greedy should finish quickly
    
    # Phase 2: Local Search with Simulated Annealing
    # Multi-move neighborhood: remove one block, optionally add one or two blocks
    
    selected_set = set(selected_indices)
    best_set = set(selected_indices)
    best_size = len(selected_set)
    
    # Compute current coverage for quick updates
    cur_coverage = {}
    for idx in selected_set:
        for ts in block_tsubs[idx]:
            cur_coverage[ts] = cur_coverage.get(ts, 0) + 1
    
    def is_valid_coverage(cov):
        for ts in tsub_to_blocks:
            if cov.get(ts, 0) < 1:
                return False
        return True
    
    # Simulated Annealing
    T = 5.0
    min_T = 0.1
    cooling_rate = 0.995
    
    iteration = 0
    while time.time() < deadline - 5:
        iteration += 1
        
        # Try to remove a block
        candidates = list(selected_set)
        if not candidates:
            break
        rem_block = random.choice(candidates)
        
        # Check if removing this block breaks any t-subset
        uncovered = []
        for ts in block_tsubs[rem_block]:
            if cur_coverage[ts] <= 1:
                uncovered.append(ts)
        
        if not uncovered:
            # Safe removal
            selected_set.remove(rem_block)
            for ts in block_tsubs[rem_block]:
                cur_coverage[ts] -= 1
                if cur_coverage[ts] == 0:
                    del cur_coverage[ts]
            
            if len(selected_set) < best_size:
                best_set = set(selected_set)
                best_size = len(selected_set)
            
            # Continue with next iteration (accept the move)
            continue
        
        # Need to add blocks to cover uncovered t-subsets
        # Try to find a single block that covers all uncovered
        # If not, try two blocks
        
        # Find blocks that cover at least one uncovered t-subset
        candidate_adds = set()
        for ts in uncovered:
            for b_idx in tsub_to_blocks[ts]:
                if b_idx not in selected_set:
                    candidate_adds.add(b_idx)
        
        if not candidate_adds:
            # Can't fix, skip
            T = max(min_T, T * cooling_rate)
            continue
        
        # Try single add
        found_single = False
        for add_idx in candidate_adds:
            # Check if this block covers all uncovered
            covers_all = True
            for ts in uncovered:
                if ts not in block_tsubs[add_idx]:
                    covers_all = False
                    break
            if covers_all:
                # Accept swap: remove rem_block, add add_idx
                selected_set.remove(rem_block)
                for ts in block_tsubs[rem_block]:
                    cur_coverage[ts] -= 1
                    if cur_coverage[ts] == 0:
                        del cur_coverage[ts]
                
                selected_set.add(add_idx)
                for ts in block_tsubs[add_idx]:
                    cur_coverage[ts] = cur_coverage.get(ts, 0) + 1
                
                if len(selected_set) < best_size:
                    best_set = set(selected_set)
                    best_size = len(selected_set)
                
                found_single = True
                break
        
        if found_single:
            T = max(min_T, T * cooling_rate)
            continue
        
        # Try two-block add: find two blocks whose union covers all uncovered
        # This is more expensive, so only do it occasionally
        if iteration % 10 == 0:
            cand_list = list(candidate_adds)
            random.shuffle(cand_list)
            # Limit candidates for speed
            cand_list = cand_list[:50]
            
            found_pair = False
            for i in range(len(cand_list)):
                for j in range(i+1, len(cand_list)):
                    block_i = set(block_tsubs[cand_list[i]])
                    block_j = set(block_tsubs[cand_list[j]])
                    union = block_i | block_j
                    if all(ts in union for ts in uncovered):
                        # Accept: remove rem_block, add two blocks
                        # This increases size by 1, so only accept with SA probability
                        delta_size = 1  # Net increase
                        if delta_size < 0 or random.random() < np.exp(-delta_size / T):
                            selected_set.remove(rem_block)
                            for ts in block_tsubs[rem_block]:
                                cur_coverage[ts] -= 1
                                if cur_coverage[ts] == 0:
                                    del cur_coverage[ts]
                            
                            for add_idx in [cand_list[i], cand_list[j]]:
                                selected_set.add(add_idx)
                                for ts in block_tsubs[add_idx]:
                                    cur_coverage[ts] = cur_coverage.get(ts, 0) + 1
                            
                            if len(selected_set) < best_size:
                                best_set = set(selected_set)
                                best_size = len(selected_set)
                            
                            found_pair = True
                            break
                if found_pair:
                    break
        
        if not found_pair:
            T = max(min_T, T * cooling_rate)
    
    # Final greedy reduction pass
    final_set = set(best_set)
    final_coverage = {}
    for idx in final_set:
        for ts in block_tsubs[idx]:
            final_coverage[ts] = final_coverage.get(ts, 0) + 1
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for idx in list(final_set):
            can_remove = True
            for ts in block_tsubs[idx]:
                if final_coverage.get(ts, 0) <= 1:
                    can_remove = False
                    break
            if can_remove:
                final_set.remove(idx)
                for ts in block_tsubs[idx]:
                    final_coverage[ts] -= 1
                    if final_coverage[ts] == 0:
                        del final_coverage[ts]
                changed = True
    
    return [all_blocks[i] for i in final_set]
