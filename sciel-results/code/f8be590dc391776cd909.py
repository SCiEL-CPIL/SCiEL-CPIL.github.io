import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the specific task instance
    # v=19, k=7, t=4
    
    # 1. Precomputation
    # Map all t-subsets to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    # Precompute bitsets for each k-block
    # block_bitsets[i] is a Python int where bit j is 1 if k_block i covers t_subset j
    block_bitsets = [0] * n_k_blocks
    # Also precompute for each t_subset, which k_blocks cover it
    # subset_to_blocks[j] is a list of block indices
    subset_to_blocks = [[] for _ in range(n_t_subsets)]
    
    for i, block in enumerate(k_blocks):
        block_set = set(block)
        # Find all t-subsets within this block
        for t_sub in itertools.combinations(block, t):
            idx = subset_to_idx[t_sub]
            block_bitsets[i] |= (1 << idx)
            subset_to_blocks[idx].append(i)
            
    # 2. Greedy Construction with Randomization
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
        
        uncovered = (1 << n_t_subsets) - 1
        blocks = []
        used_blocks = set()
        
        while uncovered:
            best_block = -1
            best_count = -1
            best_bitset = 0
            
            # Iterate through all blocks to find the one covering most uncovered subsets
            # Optimization: We can skip blocks already used, though greedy usually doesn't repeat.
            # We must pick from all blocks.
            for i in range(n_k_blocks):
                if i in used_blocks:
                    continue
                covered = block_bitsets[i] & uncovered
                count = covered.bit_count()
                if count > best_count:
                    best_count = count
                    best_block = i
                    best_bitset = covered
                    if count == 0:
                        break # No block covers any more? Should not happen if uncovered != 0
            
            if best_block == -1:
                # Fallback: just pick any block covering at least 1
                # This shouldn't happen if the problem is solvable
                for i in range(n_k_blocks):
                    if i in used_blocks:
                        continue
                    if block_bitsets[i] & uncovered:
                        best_block = i
                        best_bitset = block_bitsets[i] & uncovered
                        break
            
            if best_block == -1:
                break # Error, no coverage possible?
                
            blocks.append(best_block)
            used_blocks.add(best_block)
            uncovered &= ~best_bitset
            
        return blocks

    # 3. Local Search Improvement
    def local_search(blocks):
        # Convert list of block indices to set for O(1) lookup
        blocks_set = set(blocks)
        
        # Calculate coverage counts
        coverage_counts = [0] * n_t_subsets
        for b in blocks:
            bs = block_bitsets[b]
            # Iterate set bits
            while bs:
                lsb = bs & -bs
                idx = lsb.bit_length() - 1
                coverage_counts[idx] += 1
                bs ^= lsb
        
        # Function to check if removing block b is safe (no count drops to 0)
        def can_remove(b):
            bs = block_bitsets[b]
            while bs:
                lsb = bs & -bs
                idx = lsb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    return False
                bs ^= lsb
            return True

        # Function to get lost subsets if we remove block b
        def get_lost(b):
            lost = []
            bs = block_bitsets[b]
            while bs:
                lsb = bs & -bs
                idx = lsb.bit_length() - 1
                if coverage_counts[idx] == 1:
                    lost.append(idx)
                bs ^= lsb
            return lost

        # Function to apply removal
        def apply_removal(b):
            bs = block_bitsets[b]
            while bs:
                lsb = bs & -bs
                idx = lsb.bit_length() - 1
                coverage_counts[idx] -= 1
                bs ^= lsb

        # Function to apply addition
        def apply_addition(b):
            bs = block_bitsets[b]
            while bs:
                lsb = bs & -bs
                idx = lsb.bit_length() - 1
                coverage_counts[idx] += 1
                bs ^= lsb

        improved = True
        while improved:
            improved = False
            # Try to remove any redundant block
            for i in range(len(blocks)):
                if can_remove(blocks[i]):
                    apply_removal(blocks[i])
                    blocks_set.remove(blocks[i])
                    blocks.pop(i)
                    improved = True
                    break # Restart loop as indices changed
            
            if improved:
                continue
            
            # Try to swap
            # To make it faster, we can limit the number of swaps or randomize
            for i in range(len(blocks)):
                b = blocks[i]
                lost = get_lost(b)
                if not lost:
                    # Should have been caught by can_remove, but just in case
                    apply_removal(b)
                    blocks_set.remove(b)
                    blocks.pop(i)
                    improved = True
                    break
                
                # Find a replacement block that covers all 'lost' subsets
                # Use the first lost subset to filter candidates
                first_lost = lost[0]
                candidates = subset_to_blocks[first_lost]
                
                best_candidate = -1
                best_gain = -1
                
                for cand in candidates:
                    if cand in blocks_set:
                        continue
                    
                    # Check if cand covers all other lost subsets
                    covers_all = True
                    for l in lost[1:]:
                        if not (block_bitsets[cand] >> l) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    
                    # If it covers all lost, calculate "gain" (how many other subsets it helps)
                    # Gain = number of subsets with count 0 (none, since we only consider valid states)
                    # Actually, we just want to swap if it maintains validity.
                    # To improve further, we could prefer blocks that allow future removals.
                    # Simple heuristic: just swap.
                    best_candidate = cand
                    break # Take the first valid one
                    
                if best_candidate != -1:
                    # Swap
                    apply_removal(b)
                    blocks_set.remove(b)
                    apply_addition(best_candidate)
                    blocks_set.add(best_candidate)
                    blocks[i] = best_candidate
                    improved = True
                    break # Restart loop
                    
        return blocks

    # 4. Main Loop: Multiple Restart
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        seed += 1
        # Randomize greedy by shuffling block order? 
        # Or just use different random seeds for tie-breaking if we added that.
        # Current greedy is deterministic. Let's add random tie-breaking.
        
        # Modified Greedy with Random Tie-Breaking
        uncovered = (1 << n_t_subsets) - 1
        blocks = []
        used_blocks = set()
        
        while uncovered:
            best_count = -1
            candidates = []
            
            for i in range(n_k_blocks):
                if i in used_blocks:
                    continue
                covered = block_bitsets[i] & uncovered
                count = covered.bit_count()
                if count > best_count:
                    best_count = count
                    candidates = [i]
                elif count == best_count:
                    candidates.append(i)
            
            if not candidates:
                break
                
            # Pick randomly from best candidates
            chosen = random.choice(candidates)
            blocks.append(chosen)
            used_blocks.add(chosen)
            uncovered &= ~block_bitsets[chosen]
            
        # Local Search
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
            # If we hit the known best or lower bound, we can stop early?
            # Lower bound is 111. Known best 151.
            if best_len <= 140: # Arbitrary threshold for "good enough" to stop searching harder?
                # Maybe keep going to see if we can go lower
                pass

    # Convert block indices back to lists of points
    final_blocks = [list(k_blocks[i]) for i in best_blocks]
    
    # Ensure distinct blocks (they should be by construction)
    # Sort for consistency? Not required but nice.
    # final_blocks.sort()
    
    return final_blocks

