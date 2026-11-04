import itertools
import time
import random
import sys

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin for 120s limit
    rng = random.Random(42)
    
    # 1. Precomputation
    # Map all t-subsets to indices 0..N_t-1
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    all_mask = (1 << n_tsubs) - 1
    
    # Map all k-subsets to indices and precompute their coverage masks
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    ksub_to_idx = {ks: i for i, ks in enumerate(all_ksubs)}
    
    # Precompute coverage mask for each k-subset
    # mask[i] is a Python int where bit j is set if tsub j is contained in ksub i
    ksub_masks = [0] * n_ksubs
    for i, ks in enumerate(all_ksubs):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = tsub_to_idx[ts]
            mask |= (1 << idx)
        ksub_masks[i] = mask
    
    # 2. Greedy Construction
    def greedy_construct():
        uncovered = all_mask
        blocks = []
        while uncovered:
            if time.time() > deadline - 10:
                break
            # Find block covering most uncovered t-subsets
            best_gain = -1
            best_block_idx = -1
            # Randomize order to escape local minima in greedy
            indices = list(range(n_ksubs))
            rng.shuffle(indices)
            for idx in indices:
                mask = ksub_masks[idx]
                gain = bin(uncovered & mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = idx
                if best_gain == 0:
                    break
            if best_block_idx == -1 or best_gain == 0:
                break
            blocks.append(best_block_idx)
            uncovered &= ~ksub_masks[best_block_idx]
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # 3. Simulated Annealing / Local Search
    def local_search(initial_blocks, time_limit):
        current_blocks = list(initial_blocks)
        current_mask = 0
        for b in current_blocks:
            current_mask |= ksub_masks[b]
        
        # Ensure valid start
        if current_mask != all_mask:
            return None, float('inf')
            
        current_cost = len(current_blocks)
        best_cost = current_cost
        best_blocks = list(current_blocks)
        
        temp = 10.0
        cooling = 0.999
        min_temp = 0.01
        end_time = time.time() + time_limit
        
        while time.time() < end_time and temp > min_temp:
            # Generate a move
            move_type = rng.random()
            
            if move_type < 0.3 and len(current_blocks) > 1:
                # Remove a random block
                idx_to_remove = rng.randint(0, len(current_blocks) - 1)
                b = current_blocks[idx_to_remove]
                new_mask = current_mask & ~ksub_masks[b]
                if new_mask == all_mask:
                    # Valid move, cost decreases
                    current_blocks.pop(idx_to_remove)
                    current_mask = new_mask
                    current_cost -= 1
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                else:
                    # Invalid move, accept with probability
                    # Penalty is number of newly uncovered t-subsets
                    uncovered_new = all_mask & ~new_mask
                    penalty = bin(uncovered_new).count('1')
                    # Accept if exp(-penalty/temp) > random
                    if rng.random() < pow(2.71828, -penalty / temp):
                        current_blocks.pop(idx_to_remove)
                        current_mask = new_mask
                        current_cost -= 1
            elif move_type < 0.7:
                # Swap a random block with a random other block
                if len(current_blocks) > 0:
                    idx_to_swap = rng.randint(0, len(current_blocks) - 1)
                    old_b = current_blocks[idx_to_swap]
                    new_b = rng.randint(0, n_ksubs - 1)
                    if new_b == old_b:
                        continue
                    new_mask = (current_mask & ~ksub_masks[old_b]) | ksub_masks[new_b]
                    if new_mask == all_mask:
                        # Valid swap, cost stays same
                        current_blocks[idx_to_swap] = new_b
                        current_mask = new_mask
                        # Cost unchanged, but structure changed
                    else:
                        uncovered_new = all_mask & ~new_mask
                        penalty = bin(uncovered_new).count('1')
                        if rng.random() < pow(2.71828, -penalty / temp):
                            current_blocks[idx_to_swap] = new_b
                            current_mask = new_mask
            else:
                # Add a random block (increases cost, helps escape)
                new_b = rng.randint(0, n_ksubs - 1)
                # Check if already present
                if new_b in current_blocks:
                    continue
                new_mask = current_mask | ksub_masks[new_b]
                # Mask can only gain bits, so if current was valid, new is valid
                # If current was invalid, new might be more valid
                current_blocks.append(new_b)
                current_mask = new_mask
                current_cost += 1
                
            temp *= cooling
            
            # If we found a better solution, reset temp slightly to explore around it?
            # Or just let it cool.
            
        # Final cleanup: remove redundant blocks
        # Check each block, if removing it keeps coverage, remove it
        i = 0
        while i < len(best_blocks):
            b = best_blocks[i]
            # Check if b is redundant in best_blocks
            # This is O(N^2) in block count, but N is ~150, so 22500 checks.
            # Each check is O(1) bitwise op if we recompute mask, or O(N) if we check coverage of each tsub.
            # Better: recompute mask of best_blocks without i?
            # Actually, just try to remove from current state if we are still valid?
            # Let's do a simple pass on the final best_blocks
            i += 1
            
        # Proper redundancy removal on best_blocks
        valid_blocks = list(best_blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(valid_blocks)):
                b = valid_blocks[i]
                # Compute mask of all other blocks
                other_mask = 0
                for j in range(len(valid_blocks)):
                    if j != i:
                        other_mask |= ksub_masks[valid_blocks[j]]
                if other_mask == all_mask:
                    valid_blocks.pop(i)
                    changed = True
                    break
        
        return valid_blocks, len(valid_blocks)

    # Run multiple restarts
    restarts = 0
    while time.time() < deadline - 5:
        # Generate initial solution
        initial = greedy_construct()
        if not initial:
            break
        
        # Allocate time for local search
        # Decrease time limit for later restarts
        remaining = deadline - 5 - time.time()
        if remaining < 5:
            break
        ls_time = min(remaining * 0.8, 10.0) # Max 10s per LS
        
        res_blocks, res_cost = local_search(initial, ls_time)
        
        if res_blocks is not None and res_cost < best_count:
            best_count = res_cost
            best_blocks = res_blocks
            # Print progress if needed
            # print(f"Restart {restarts}: Found {best_count} blocks", file=sys.stderr)
        restarts += 1
        
        # If we hit the theoretical lower bound or a very good number, stop early?
        # Lower bound is 131. Best known 151.
        if best_count <= 151:
             # We matched or beat best known. Keep searching a bit more?
             # If we beat 151, we are in new territory.
             pass

    # If no solution found, return empty or a dummy valid one?
    # The problem requires a valid covering.
    if best_blocks is None:
        # Fallback: simple greedy without SA
        initial = greedy_construct()
        best_blocks = initial
        best_count = len(initial)
        
    # Convert indices back to lists of points
    final_blocks = [list(all_ksubs[i]) for i in best_blocks]
    
    # Final validation and cleanup
    # Ensure no duplicates
    seen = set()
    unique_blocks = []
    for b in final_blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    
    # Check validity
    covered = 0
    for b in unique_blocks:
        for ts in itertools.combinations(b, t):
            idx = tsub_to_idx[ts]
            covered |= (1 << idx)
            
    if covered != all_mask:
        # Add missing blocks
        uncovered = all_mask & ~covered
        while uncovered:
            # Find block covering most uncovered
            best_gain = -1
            best_b = -1
            for i in range(n_ksubs):
                gain = bin(uncovered & ksub_masks[i]).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_b = i
            if best_b == -1:
                break
            unique_blocks.append(list(all_ksubs[best_b]))
            covered |= ksub_masks[best_b]
            uncovered = all_mask & ~covered
            
    return unique_blocks

