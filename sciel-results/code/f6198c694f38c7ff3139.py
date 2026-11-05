"""Covering design C(24,7,3) using Directed Local Search (1-Exchange) and Greedy Initialization."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0

    # Precompute all t-subsets
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Map point to list of triples containing it? 
    # Not strictly needed for mask calculation, but useful for candidate generation.
    
    def get_block_mask(block):
        """Compute the bitset of triples covered by a block."""
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(sorted(triple))]
            mask |= (1 << idx)
        return mask

    # 1. Greedy Initialization
    uncovered = (1 << num_triples) - 1
    blocks = []
    
    # Precompute masks for all possible blocks? 
    # C(24,7) = 346104. 
    # Each mask is 2024 bits. 
    # 346104 * (2024/8) bytes ~ 87 MB. 
    # This might be too much memory or slow to generate initially.
    # Let's stick to on-the-fly generation but cache if we see repeated candidates?
    # In greedy, we see many candidates. 
    # In local search, we see specific candidates.
    
    # Greedy Phase
    while uncovered:
        if time.time() - start_time > timeout:
            break
        
        # Pick a random uncovered triple
        anchor_idx = -1
        # Optimization: Instead of random sampling, just pick the first uncovered one?
        # Random is better for diversity.
        # To speed up finding an uncovered triple, we can maintain a list of uncovered indices.
        # But updating it is O(N). 
        # Given N=2024, scanning is fast.
        
        # Let's just scan for the first uncovered one to be deterministic and fast?
        # Or random. Let's use random to avoid bias.
        for _ in range(20):
            idx = random.randint(0, num_triples - 1)
            if (uncovered >> idx) & 1:
                anchor_idx = idx
                break
        
        if anchor_idx == -1:
            # Fallback: find first
            for idx in range(num_triples):
                if (uncovered >> idx) & 1:
                    anchor_idx = idx
                    break
            
        if anchor_idx == -1:
            break # Should not happen if uncovered is not 0

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
            # Early stop if we cover all uncovered? Unlikely.
            
        if best_block is None:
            break
            
        blocks.append(list(best_block))
        best_mask = get_block_mask(best_block)
        uncovered &= ~best_mask

    # If we timed out during greedy, we have an invalid covering.
    # We must complete it to have a valid baseline, otherwise score is 0.
    if uncovered:
        # Simple completion
        while uncovered:
            if time.time() - start_time > timeout:
                break
            # Pick first uncovered
            idx = 0
            while idx < num_triples and not ((uncovered >> idx) & 1):
                idx += 1
            if idx == num_triples:
                break
            anchor_triple = all_triples[idx]
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
                blocks.append(list(best_block))
                best_mask = get_block_mask(best_block)
                uncovered &= ~best_mask
            else:
                break

    if uncovered:
        # Still invalid. Return what we have.
        return blocks

    # 2. Local Search: Directed 1-Exchange
    
    # Helper to check validity and get coverage counts
    def analyze_covering(b_list):
        """
        Returns:
        - is_valid: bool
        - coverage_counts: list of counts for each triple
        - private_masks: list of masks for triples covered only by block i
        """
        coverage_counts = [0] * num_triples
        masks = [0] * len(b_list)
        
        for i, b in enumerate(b_list):
            m = get_block_mask(b)
            masks[i] = m
            # Update counts
            # Iterating over set bits in m
            temp = m
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                coverage_counts[idx] += 1
                temp ^= lsb
        
        is_valid = all(c > 0 for c in coverage_counts)
        
        # Find private triples for each block
        private_masks = [0] * len(b_list)
        for i in range(len(b_list)):
            m = masks[i]
            temp = m
            pm = 0
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    pm |= lsb
                temp ^= lsb
            private_masks[i] = pm
            
        return is_valid, coverage_counts, private_masks, masks

    # Initial analysis
    is_valid, coverage_counts, private_masks, masks = analyze_covering(blocks)
    
    # Phase 2a: Remove redundant blocks (if any, though greedy shouldn't produce many)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if time.time() - start_time > timeout:
                return blocks
            if private_masks[i] == 0:
                # Block i is redundant
                blocks.pop(i)
                masks.pop(i)
                private_masks.pop(i)
                changed = True
                # Re-analyze
                is_valid, coverage_counts, private_masks, masks = analyze_covering(blocks)
                break
        if time.time() - start_time > timeout:
            return blocks

    # Phase 2b: 1-Exchange Search
    # Goal: Find a block Bi with private triples Pi.
    # Find a candidate block Bnew that covers Pi.
    # If replacing Bi with Bnew keeps the covering valid, and allows removal of another block, accept.
    
    max_iterations = 1000
    iter_count = 0
    
    while iter_count < max_iterations:
        if time.time() - start_time > timeout:
            break
        iter_count += 1
        
        found_improvement = False
        
        # Pick a random block to try to exchange
        # Focus on blocks with private triples
        candidates_for_exchange = [i for i in range(len(blocks)) if private_masks[i] != 0]
        
        if not candidates_for_exchange:
            # No blocks have private triples? Then all blocks are redundant?
            # Or we are at a local minimum where every block has at least one private triple.
            # In this case, 1-exchange cannot reduce size directly unless the new block
            # makes another block redundant.
            # We need to search for Bi, Bnew such that:
            # 1. Bnew covers Private(Bi)
            # 2. New covering is valid.
            # 3. Some block Bj becomes redundant.
            
            pass # We will just pick a random block with private triples if any, else break or shake.

        if not candidates_for_exchange:
            # If no private triples, we might be stuck or valid with no redundancies.
            # Actually, if private_masks[i] is 0 for all i, then every triple is covered by >= 2 blocks.
            # We can remove one block and check validity.
            # This is handled by the "Remove redundant" loop above?
            # Wait, the "Remove redundant" loop removes if private_masks[i] == 0.
            # So if we exit that loop, all blocks have private_masks[i] != 0.
            # So candidates_for_exchange should not be empty if len(blocks) > 0.
            
            # If it is empty, it means len(blocks) == 0? Or logic error.
            # Let's break to avoid infinite loop.
            break

        idx_i = random.choice(candidates_for_exchange)
        Pi = private_masks[idx_i]
        
        # Determine points that MUST be in Bnew
        # Bnew must cover all triples in Pi.
        # This means Bnew must contain the union of points of all triples in Pi.
        union_points = set()
        temp = Pi
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            union_points.update(all_triples[idx])
            temp ^= lsb
            
        if len(union_points) > k:
            # Cannot cover all private triples with a single block of size k
            continue
            
        # Available points for the rest of Bnew
        available = [p for p in range(v) if p not in union_points]
        
        # Generate candidates Bnew containing union_points
        # Need k - len(union_points) more points
        num_to_add = k - len(union_points)
        
        if num_to_add < 0:
            continue
            
        best_candidate = None
        best_candidate_mask = 0
        
        # Heuristic: Try to find a Bnew that also covers many triples from other blocks
        # to potentially make them redundant.
        # We can score candidates by how many "shared" triples they cover.
        
        for comb in itertools.combinations(available, num_to_add):
            candidate_block = tuple(sorted(tuple(union_points) + comb))
            candidate_mask = get_block_mask(candidate_block)
            
            # Check if candidate covers Pi (it must, by construction)
            # Check if replacing blocks[idx_i] with candidate_block results in a valid covering
            
            # New coverage counts
            # We can simulate:
            # old_counts = coverage_counts
            # new_counts = old_counts - mask_i + candidate_mask
            
            # But we need to check if any block becomes redundant.
            # A block j becomes redundant if all its private triples are covered by other blocks.
            # Actually, simpler:
            # 1. Construct new_blocks list (conceptually)
            # 2. Check validity.
            # 3. Check if any block is redundant.
            
            # Optimization: 
            # If we replace Bi with Bnew, the only triples that change coverage are:
            # - Triples in mask_i (old)
            # - Triples in candidate_mask (new)
            # 
            # Let's just build the new list and analyze? 
            # That is O(N * K) which is slow.
            
            # Faster check:
            # 1. Calculate new coverage counts for affected triples.
            # 2. Check if any triple drops to 0.
            # 3. Check if any block j (j != i) has all its private triples covered by blocks other than j.
            #    Actually, if we just check if the new covering is valid, and then check for redundancies,
            #    we can do it efficiently if we update counts incrementally.
            
            # Let's try a simpler approach for now: 
            # Only accept if the new block allows removal of another block.
            # How to check this fast?
            # 
            # Let's just generate a few candidates and test them fully.
            # Given the constraint, we can't test all.
            # Let's pick the candidate that maximizes overlap with other blocks?
            
            # Overlap = (candidate_mask & (sum of other masks)).bit_count()?
            # This is expensive.
            
            # Let's just take the first valid candidate that allows a removal?
            # Or just the first valid candidate?
            
            # Let's check validity of replacement.
            # We need to ensure no triple is uncovered.
            # A triple is uncovered if:
            # - It was covered only by Bi (in Pi) and NOT in candidate_mask. (Impossible by construction)
            # - It was covered by Bi and others, but NOT in candidate_mask, and others don't cover it?
            #   No, if it was covered by others, it's still covered.
            #   So the only risk is triples that were covered by Bi and NOT by candidate_mask 
            #   AND were NOT covered by any other block.
            #   These are exactly the triples in (mask_i & ~candidate_mask) that have count 1 in old covering?
            #   No. If count is 1, it's in Pi. And Pi is covered by candidate_mask.
            #   So if Pi is covered, and candidate_mask covers Pi, then any triple in mask_i 
            #   that is NOT in candidate_mask must have count >= 2 in old covering (so covered by others).
            #   Thus, the replacement is ALWAYS valid if candidate_mask covers Pi!
            
            # Wait, is that true?
            # Triple x in mask_i.
            # If x in candidate_mask: covered by new block.
            # If x not in candidate_mask:
            #   If x in Pi (count 1): Contradiction, because Pi subset candidate_mask.
            #   If x not in Pi (count >= 2): Covered by at least one other block.
            # So YES, replacing Bi with any Bnew that covers Pi results in a VALID covering.
            
            # So we just need to check if any block becomes redundant.
            # A block j (j != i) becomes redundant if all its private triples are covered by blocks in the new set.
            # Since the new set is same as old except Bi -> Bnew, the only change in coverage for block j
            # is:
            # - Triples in Bnew that were not in Bi.
            # - Triples in Bi that were not in Bnew.
            
            # Let's just pick a random candidate from the generated ones and test it fully.
            # To save time, we can limit the number of candidates we test per iteration.
            
            # Let's just take the first candidate.
            best_candidate = candidate_block
            best_candidate_mask = candidate_mask
            break # Take the first one to save time
            
        if best_candidate is None:
            continue
            
        # Perform the exchange
        new_blocks = blocks.copy()
        new_blocks[idx_i] = list(best_candidate)
        
        # Check for redundancies in new_blocks
        # We need to see if any block j is now redundant.
        # We can do a quick check:
        # For each j != i, check if private_masks[j] (from old state) is still "private"?
        # No, private masks change.
        
        # Let's just run the reduction phase on new_blocks.
        # But that is O(N^2).
        
        # Optimization: 
        # Only check blocks that might be affected.
        # Blocks that share triples with Bi or Bnew.
        
        # For now, let's just accept the exchange if it reduces size after a quick reduction check.
        # To avoid O(N^2) per candidate, we will just do the exchange and then run the "Remove redundant" loop.
        
        # But we only do this for ONE candidate per main loop iteration to keep it fast.
        
        # Apply exchange
        blocks = new_blocks
        is_valid, coverage_counts, private_masks, masks = analyze_covering(blocks)
        
        # Try to remove redundant blocks
        local_changed = True
        while local_changed:
            local_changed = False
            for j in range(len(blocks)):
                if private_masks[j] == 0:
                    blocks.pop(j)
                    private_masks.pop(j)
                    masks.pop(j)
                    local_changed = True
                    # Re-analyze
                    is_valid, coverage_counts, private_masks, masks = analyze_covering(blocks)
                    break
            if time.time() - start_time > timeout:
                return blocks
        
        if len(blocks) < len(new_blocks) - 1: # If we removed at least 1 block, it was an improvement
            found_improvement = True
            # Restart search from new state
            iter_count = 0 
            break
        
        # If no improvement, we could try another candidate, but for simplicity, we just move on.
        # To avoid getting stuck, we might want to accept non-improving moves (Simulated Annealing),
        # but given the time limit, let's stick to strict improvement for now.
        # If we don't find an improvement in many iterations, we should Shake.

    # If we are stuck, do a Shake (Random Perturbation)
    if time.time() - start_time < timeout - 10:
        # Simple shake: remove 2 random blocks, re-greedy
        if len(blocks) > 10:
            shake_blocks = blocks.copy()
            remove_count = random.randint(1, 3)
            indices = random.sample(range(len(shake_blocks)), remove_count)
            for idx in sorted(indices, reverse=True):
                shake_blocks.pop(idx)
            
            # Re-greedy fill
            # Calculate uncovered
            covered_mask = 0
            for b in shake_blocks:
                covered_mask |= get_block_mask(b)
            uncovered = ((1 << num_triples) - 1) & ~covered_mask
            
            while uncovered:
                if time.time() - start_time > timeout:
                    break
                anchor_idx = -1
                for _ in range(20):
                    idx = random.randint(0, num_triples - 1)
                    if (uncovered >> idx) & 1:
                        anchor_idx = idx
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
                
                if best_block:
                    shake_blocks.append(list(best_block))
                    best_mask = get_block_mask(best_block)
                    uncovered &= ~best_mask
                else:
                    break
            
            if not uncovered and len(shake_blocks) < len(blocks):
                blocks = shake_blocks
                # Re-run local search on new best
                # (Loop continues, so it will re-analyze and search)

    return blocks

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
