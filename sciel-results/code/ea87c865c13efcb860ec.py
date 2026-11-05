"""Covering design C(24,7,3) using VNS on Block Permutations."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer for safety

    # Precompute all t-subsets as indices for bitset representation
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    full_mask = (1 << num_triples) - 1
    
    def get_block_mask(block):
        """Compute the bitset of triples covered by a block."""
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(sorted(triple))]
            mask |= (1 << idx)
        return mask

    # 1. Initial Greedy Construction
    def greedy_construct():
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() - start_time > timeout:
                break
            
            # Pick a random uncovered triple
            anchor_idx = -1
            for _ in range(100):
                idx = random.randint(0, num_triples - 1)
                if (uncovered >> idx) & 1:
                    anchor_idx = idx
                    break
            
            if anchor_idx == -1:
                temp = uncovered
                while temp:
                    lsb = temp & -temp
                    anchor_idx = lsb.bit_length() - 1
                    break
            
            if anchor_idx == -1:
                break

            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            
            # Iterate over combinations of k-t points from available_points
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                mask = get_block_mask(candidate_block)
                gain = (mask & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                break
            
            blocks.append(list(best_block))
            best_mask = get_block_mask(best_block)
            uncovered &= ~best_mask
            
        return blocks

    initial_blocks = greedy_construct()
    
    # If greedy didn't finish, force completion (shouldn't happen with 115s limit for this size)
    if len(initial_blocks) == 0 or (1 << num_triples) - 1 & ~reduce(lambda x, y: x | get_block_mask(y), initial_blocks, 0):
         # Fallback: just return what we have or try simple completion
         pass 

    # 2. VNS on Permutations
    
    # Precompute masks for initial blocks
    block_masks = [get_block_mask(b) for b in initial_blocks]
    n_blocks = len(initial_blocks)
    
    # Current best state
    best_perm = list(range(n_blocks))
    best_essential_count = 0
    
    def evaluate_permutation(perm):
        """
        Returns the number of essential blocks in the given permutation.
        Essential block: covers at least one triple not covered by any preceding block.
        """
        covered_so_far = 0
        essential_count = 0
        for idx in perm:
            mask = block_masks[idx]
            # Check if this block adds any new coverage
            if mask & ~covered_so_far:
                essential_count += 1
                covered_so_far |= mask
        return essential_count

    # Initial evaluation
    best_essential_count = evaluate_permutation(best_perm)
    
    # Helper: Get the set of essential block indices for a permutation
    def get_essential_indices(perm):
        covered_so_far = 0
        essential_indices = []
        for idx in perm:
            mask = block_masks[idx]
            if mask & ~covered_so_far:
                essential_indices.append(idx)
                covered_so_far |= mask
        return essential_indices

    # Local Search: Improve permutation by swapping adjacent elements or small segments
    def local_search(perm):
        improved = True
        while improved:
            improved = False
            n = len(perm)
            for i in range(n - 1):
                # Try swapping i and i+1
                perm_swap = perm[:]
                perm_swap[i], perm_swap[i+1] = perm_swap[i+1], perm_swap[i]
                if evaluate_permutation(perm_swap) < evaluate_permutation(perm):
                    perm = perm_swap
                    improved = True
                    break # Restart scan
            # Also try inserting element i at j (shift)
            # This is more expensive, maybe just swaps and reversals are enough for VNS
            # Let's add a simple "move" neighborhood
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    perm_move = perm[:]
                    elem = perm_move.pop(i)
                    perm_move.insert(j, elem)
                    if evaluate_permutation(perm_move) < evaluate_permutation(perm):
                        perm = perm_move
                        improved = True
                        break
                if improved: break
        return perm

    # VNS Main Loop
    current_perm = best_perm[:]
    current_count = best_essential_count
    
    max_iter = 5000 # Limit iterations to fit in time
    iter_count = 0
    
    while iter_count < max_iter:
        if time.time() - start_time > timeout - 5: # Reserve 5s for final steps
            break
        iter_count += 1
        
        # Local Search
        current_perm = local_search(current_perm)
        current_count = evaluate_permutation(current_perm)
        
        if current_count < best_essential_count:
            best_perm = current_perm[:]
            best_essential_count = current_count
            # Reset neighborhood size implicitly by restarting VNS logic or just continuing
            # In standard VNS, we might restart from best. Here we just continue.
        
        # Perturbation
        # N1: Swap two random elements
        # N2: Reverse a random segment
        # N3: Move one element to another position
        
        n = len(current_perm)
        if n < 2:
            break
            
        if iter_count % 3 == 0:
            # N1: Swap
            i, j = random.sample(range(n), 2)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        elif iter_count % 3 == 1:
            # N2: Reverse segment
            i, j = random.sample(range(n), 2)
            if i > j: i, j = j, i
            current_perm[i:j+1] = reversed(current_perm[i:j+1])
        else:
            # N3: Move
            i, j = random.sample(range(n), 2)
            elem = current_perm.pop(i)
            current_perm.insert(j, elem)

    # Extract essential blocks from best_perm
    essential_indices = get_essential_indices(best_perm)
    reduced_blocks = [initial_blocks[i] for i in essential_indices]
    reduced_masks = [block_masks[i] for i in essential_indices]
    
    # 3. Refinement: Remove & Re-greedy
    # Now we have a smaller set of blocks. 
    # We try to remove some blocks from this set and re-greedy the gaps.
    # This is effectively a VNS on the *set* of blocks, using the permutation insight to get a good start.
    
    best_final_blocks = reduced_blocks
    best_final_size = len(reduced_blocks)
    
    # Precompute masks for reduced blocks
    # We need to be careful with indices if we modify the list
    
    def try_regreedy(blocks_to_test):
        """
        Takes a list of blocks, removes some randomly, re-greeds the gaps.
        Returns the new list if it's valid and smaller, else None.
        """
        if not blocks_to_test:
            return None
        
        current_blocks = list(blocks_to_test)
        current_masks = [get_block_mask(b) for b in current_blocks]
        
        # Calculate uncovered
        covered_mask = 0
        for m in current_masks:
            covered_mask |= m
        uncovered = full_mask & ~covered_mask
        
        # Randomly remove some blocks
        n_rem = random.randint(1, min(3, len(current_blocks)//4))
        if n_rem == 0:
            return None
            
        indices_to_remove = random.sample(range(len(current_blocks)), n_rem)
        
        # Remove them
        new_blocks = []
        new_masks = []
        for i in range(len(current_blocks)):
            if i not in indices_to_remove:
                new_blocks.append(current_blocks[i])
                new_masks.append(current_masks[i])
                
        # Recalculate uncovered based on new_blocks
        covered_mask = 0
        for m in new_masks:
            covered_mask |= m
        uncovered = full_mask & ~covered_mask
        
        # Re-greedy
        while uncovered:
            if time.time() - start_time > timeout:
                return None # Timeout
            
            anchor_idx = -1
            for _ in range(50):
                idx = random.randint(0, num_triples - 1)
                if (uncovered >> idx) & 1:
                    anchor_idx = idx
                    break
            
            if anchor_idx == -1:
                temp = uncovered
                while temp:
                    lsb = temp & -temp
                    anchor_idx = lsb.bit_length() - 1
                    break
            
            if anchor_idx == -1:
                break

            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            
            for comb in itertools.combinations(available_points, k - t):
                candidate_block = tuple(sorted(anchor_triple + comb))
                mask = get_block_mask(candidate_block)
                gain = (mask & uncovered).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
            
            if best_block is None:
                return None
            
            new_blocks.append(list(best_block))
            best_mask = get_block_mask(best_block)
            new_masks.append(best_mask)
            uncovered &= ~best_mask
            
        if not uncovered and len(new_blocks) < best_final_size:
            return new_blocks
            
        return None

    # Run refinement loop
    ref_iter = 0
    while time.time() - start_time < timeout - 2:
        ref_iter += 1
        # Occasionally restart from the very best found
        if ref_iter % 100 == 0:
             pass # Just continue
        
        result = try_regreedy(best_final_blocks)
        if result:
            best_final_blocks = result
            best_final_size = len(result)
            
    return best_final_blocks

# Test locally
if __name__ == "__main__":
    # Quick test
    blocks = construct(24, 7, 3)
    print(f"Number of blocks: {len(blocks)}")
    
    # Verify
    covered = set()
    for b in blocks:
        for c in itertools.combinations(b, 3):
            covered.add(tuple(sorted(c)))
    
    all_subsets = set(itertools.combinations(range(24), 3))
    if covered == all_subsets:
        print("Valid covering")
    else:
        print("Invalid covering")
        print(f"Missing: {len(all_subsets - covered)}")
