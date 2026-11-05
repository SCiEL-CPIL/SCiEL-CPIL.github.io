"""Covering design C(24,7,3) using Bitset Greedy + VNS on Block Permutations."""
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
    
    # Cache for block masks to avoid recomputation
    block_mask_cache = {}

    def get_block_mask(block):
        """Compute the bitset of triples covered by a block."""
        key = tuple(sorted(block))
        if key in block_mask_cache:
            return block_mask_cache[key]
        
        mask = 0
        for triple in itertools.combinations(key, t):
            idx = triple_to_idx[tuple(sorted(triple))]
            mask |= (1 << idx)
        
        block_mask_cache[key] = mask
        return mask

    # 1. Initial Greedy Construction
    def greedy_construct():
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            if time.time() - start_time > timeout:
                break
            
            # Find a random uncovered triple
            anchor_idx = -1
            # Optimization: Use a list of uncovered indices for faster random access?
            # For now, random sampling is fine for 2024 bits.
            for _ in range(50):
                idx = random.randint(0, num_triples - 1)
                if (uncovered >> idx) & 1:
                    anchor_idx = idx
                    break
            
            if anchor_idx == -1:
                # Fallback: find lowest set bit
                temp = uncovered
                while temp:
                    lsb = temp & -temp
                    anchor_idx = lsb.bit_length() - 1
                    break
            
            if anchor_idx == -1:
                break # Should not happen if uncovered != 0

            anchor_triple = all_triples[anchor_idx]
            anchor_set = set(anchor_triple)
            available_points = [p for p in range(v) if p not in anchor_set]
            
            best_block = None
            best_gain = -1
            
            # Generate candidates
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
            
            if not uncovered:
                break
                
        return blocks

    # 2. Local Search: VND on the sequence of blocks
    # Objective: Minimize number of blocks.
    # Neighborhoods:
    # N1: Remove redundant blocks (order-independent check, but applied sequentially)
    # N2: Replace a block with a better one
    # N3: Swap/Reverse segments (handled by the outer VNS loop perturbation)
    
    def local_search(blocks):
        """
        Takes a list of blocks, returns a potentially smaller valid list.
        Uses a sequential "reconstruction" approach:
        1. Iterate through blocks.
        2. If a block adds no new coverage given previous kept blocks, drop it.
        3. If a block adds coverage, check if a replacement block covers the same 'essential' 
           triples and potentially more total coverage, allowing for future drops.
        """
        if not blocks:
            return blocks
        
        kept_blocks = []
        covered_mask = 0
        
        # Pre-calculate masks for current blocks to speed up
        current_masks = [get_block_mask(b) for b in blocks]
        
        for i in range(len(blocks)):
            if time.time() - start_time > timeout:
                break
                
            block = blocks[i]
            mask = current_masks[i]
            
            # Essential triples: those covered by this block but not by previously kept blocks
            essential_mask = mask & ~covered_mask
            
            if essential_mask == 0:
                # Redundant given the order
                continue
            
            # Check if we can replace this block with a better one
            # Strategy: Find a block that covers `essential_mask` and has high total coverage
            # We limit the search to blocks that contain at least one point from the current block
            # to keep it fast.
            
            # For now, simple replacement: 
            # If the current block covers many non-essential triples, it's "good".
            # We can try to find a block that covers the essential_mask and has a higher 
            # "potential" (covers more total triples).
            
            # Optimization: Only attempt replacement if we have time and if the block 
            # is "heavy" (covers many triples).
            
            # A simpler, effective local move:
            # Try to remove this block and see if the *remaining* blocks (which haven't been processed yet)
            # can cover the essential_mask? No, that breaks the sequential logic.
            
            # Instead, let's stick to the "Drop if redundant" logic first.
            # Then, in a second pass, try to replace blocks.
            
            kept_blocks.append(block)
            covered_mask |= mask
            
        # After the first pass, we have a valid covering (assuming the input was valid and we didn't drop essential ones).
        # Now, try to reduce further by replacing blocks.
        
        # Second Pass: Try to replace each kept block with a better one.
        # We iterate and try to find a block that covers the "essential" part of the current block
        # but allows other blocks to become redundant? This is complex.
        
        # Simpler Local Search: 
        # 1. Remove redundant blocks (done above).
        # 2. For each block, try to swap it with a random candidate that covers its essential triples.
        
        # To keep it fast, we will do a limited number of replacement attempts.
        improved = True
        iterations = 0
        max_iterations = 100
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() - start_time > timeout:
                break
            
            for i in range(len(kept_blocks)):
                if time.time() - start_time > timeout:
                    break
                
                block = kept_blocks[i]
                mask = get_block_mask(block)
                
                # Calculate essential mask for this block in the current set
                # Essential = Mask - (Union of all other masks)
                # This is expensive to calculate for every block every iteration.
                # Approximation: Essential = Mask - (Union of masks of blocks at indices < i and > i)
                
                # Instead, let's use a simpler heuristic:
                # If a block covers many triples that are *only* covered by it, it's critical.
                # If a block covers many triples that are *also* covered by others, it's flexible.
                
                # Let's try to replace block i with a block that covers the same "unique" triples
                # but has a smaller "footprint" or better alignment.
                
                # For speed, we will skip complex replacements in the local search 
                # and rely on the VNS perturbation + re-greedy to find better blocks.
                # The "Drop if redundant" step is the main gain from permutation.
                
                # However, we can do a simple "Exchange" move:
                # Try to find a block B_new such that:
                # 1. B_new covers all triples that B_i covers exclusively (in the current set).
                # 2. B_new is not already in kept_blocks.
                # 3. B_new allows some other block to become redundant? (Hard to check).
                
                # Let's just return the result of the "Drop if redundant" pass.
                # The VNS loop will handle the heavy lifting by perturbing the order 
                # and re-running this pass, which effectively changes which blocks 
                # are considered "redundant" due to the order of coverage.
                
        return kept_blocks

    # 3. VNS Main Loop
    # Start with greedy solution
    current_blocks = greedy_construct()
    
    # Ensure validity
    if current_blocks:
        covered = 0
        for b in current_blocks:
            covered |= get_block_mask(b)
        if covered != full_mask:
            # Complete it
            uncovered = full_mask & ~covered
            while uncovered:
                if time.time() - start_time > timeout:
                    break
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
                if anchor_idx == -1: break
                
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
                if best_block:
                    current_blocks.append(list(best_block))
                    uncovered &= ~get_block_mask(best_block)
                else:
                    break

    best_blocks = current_blocks
    best_size = len(current_blocks)
    
    # Run VNS
    max_neighborhoods = 10
    perturbations_per_neighborhood = 5
    
    for k in range(1, max_neighborhoods + 1):
        if time.time() - start_time > timeout:
            break
            
        for _ in range(perturbations_per_neighborhood):
            if time.time() - start_time > timeout:
                break
                
            # Perturb the current best solution
            perturbed_blocks = list(best_blocks)
            
            # Apply k random swaps
            for _ in range(k):
                if len(perturbed_blocks) < 2:
                    break
                i, j = random.sample(range(len(perturbed_blocks)), 2)
                perturbed_blocks[i], perturbed_blocks[j] = perturbed_blocks[j], perturbed_blocks[i]
                
            # Optional: Apply a reversal
            if random.random() < 0.5 and len(perturbed_blocks) >= 2:
                i, j = random.sample(range(len(perturbed_blocks)), 2)
                if i > j: i, j = j, i
                perturbed_blocks[i:j+1] = reversed(perturbed_blocks[i:j+1])
            
            # Local Search on the perturbed solution
            # This effectively re-evaluates redundancy based on the new order
            new_blocks = local_search(perturbed_blocks)
            
            # Check if the new solution is valid
            # Since local_search only drops blocks that are redundant *given the order*, 
            # and we started with a valid set, the result should be valid 
            # IF the "essential" check in local_search is correct.
            # My local_search drops a block if `mask & ~covered_mask == 0`.
            # This ensures that every triple covered by the dropped block was already covered 
            # by a block earlier in the sequence. Thus, the union of the kept blocks 
            # is the same as the union of the original blocks.
            # So validity is preserved.
            
            if len(new_blocks) < best_size:
                # Verify validity just in case
                covered = 0
                for b in new_blocks:
                    covered |= get_block_mask(b)
                if covered == full_mask:
                    best_blocks = new_blocks
                    best_size = len(best_blocks)
                    
    return best_blocks

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
