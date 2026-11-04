import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s kill

    # Map every t-subset to an integer id
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # Precompute masks for fast gain calculation
    # masks[x] is a dict: key is a (t-1)-tuple of points, value is a bitmask of t-subsets
    # that contain x and that (t-1)-tuple.
    masks = [dict() for _ in range(v)]
    
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i in range(t):
            others = s[:i] + s[i+1:]
            key = others 
            x = s[i]
            masks[x][key] = masks[x].get(key, 0) | bit

    covered = 0  # Bitset of covered t-subsets
    blocks = []
    block_set = set()  # To avoid duplicate blocks

    # Helper: Count bits in an integer
    def popcount(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    def gain_of_add(x, B_tuple):
        """
        Calculate the number of new t-subsets covered by adding x to the current block B_tuple.
        B_tuple is a tuple of points currently in the block.
        """
        g_mask = 0
        # Iterate over all (t-1) subsets of B_tuple
        for S in itertools.combinations(B_tuple, t - 1):
            key = S
            m = masks[x].get(key)
            if m is not None:
                g_mask |= m
        # Intersect with currently uncovered subsets
        return popcount(g_mask & ~covered)

    def build_block_from(seed_points):
        """
        Grow a block from initial set (seed_points) to size k by max-gain greedy.
        Returns the block as a list of ints.
        """
        B = list(seed_points)
        B_set = set(B)
        available = [x for x in range(v) if x not in B_set]
        
        while len(B) < k:
            best_x = -1
            best_gain = -1
            # Evaluate all available points
            for x in available:
                g = gain_of_add(x, tuple(B))
                if g > best_gain:
                    best_gain = g
                    best_x = x
            
            if best_x == -1:
                break
            
            B.append(best_x)
            B_set.add(best_x)
            available.remove(best_x)
            
        return B

    def total_new_coverage(block):
        """
        Count how many t-subsets in the block are newly covered.
        """
        total = 0
        for s in itertools.combinations(sorted(block), t):
            idx = subset_to_index[s]
            if not (covered >> idx) & 1:
                total += 1
        return total

    def random_seed_block():
        """
        Construct a block by picking random points, then greedily filling.
        This explores the space differently than seeding from uncovered triples.
        """
        B = random.sample(range(v), t)
        return build_block_from(B)

    # Main Loop: Hybrid Strong Greedy with Randomized Seeds
    all_points = list(range(v))
    
    # Adaptive seed count: start high, decrease if time is tight
    # But we want to keep quality high. Let's use a fixed robust number.
    SEEDS_PER_ROUND = 80
    
    while True:
        if time.time() > deadline:
            break

        # Check if all covered
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices
        uncovered_indices = []
        m = uncov_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            if len(uncovered_indices) >= 500:  # Cap to avoid long list generation
                break
        
        if not uncovered_indices:
            break

        best_block = None
        best_gain_total = -1
        
        # Strategy: Mix of seeds from uncovered triples and random seeds
        # This helps escape local optima where greedy-from-uncovered might converge to same block structure
        
        # 1. Sample from uncovered triples
        n_uncov_seeds = SEEDS_PER_ROUND // 2
        # 2. Sample random seeds
        n_rand_seeds = SEEDS_PER_ROUND - n_uncov_seeds
        
        # Shuffle uncovered to randomize selection
        random.shuffle(uncovered_indices)
        selected_uncov = uncovered_indices[:n_uncov_seeds]
        
        for _ in range(n_rand_seeds):
            if time.time() > deadline:
                break
            # Random seed: pick t random points
            seed = random.sample(range(v), t)
            B = build_block_from(seed)
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        for idx in selected_uncov:
            if time.time() > deadline:
                break
            seed = all_subsets[idx]
            B = build_block_from(list(seed))
            bl = tuple(sorted(B))
            if bl in block_set:
                continue
            total = total_new_coverage(B)
            if total > best_gain_total:
                best_gain_total = total
                best_block = B
        
        if best_block is None:
            # If no new block found with gain > 0, we might be stuck or done.
            # If uncov_mask != 0, we are stuck. 
            # Try a pure random restart with more seeds? 
            # For now, break to avoid infinite loop if logic fails.
            # However, if best_gain_total == 0, it means all generated blocks were redundant.
            # This can happen if we are very late in the process.
            break

        bl = sorted(best_block)
        
        # Check if this block is already in our list
        if tuple(bl) in block_set:
            # This shouldn't happen if we checked above, but safe guard
            # If it does, we need to find another block. 
            # To keep it simple, we break if we can't find a new one.
            # A more robust approach would be to continue searching for a non-duplicate block.
            # Given time constraints, we assume the gain check filters out most duplicates 
            # effectively (gain 0 for duplicate).
            # Wait, if gain > 0, it cannot be a duplicate of a previous block that already covered those subsets?
            # No, a duplicate block covers the same subsets. If those subsets are covered, gain is 0.
            # If gain > 0, it covers new subsets, so it's not a duplicate of a block that only covered old subsets.
            # It *could* be a duplicate of a block that covered *some* new subsets? No, if it's the same block, it covers the same subsets.
            # If it was added before, those subsets are covered. So gain would be 0.
            # So gain > 0 implies it's a new block.
            pass

        # Mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
            
        blocks.append(bl)
        block_set.add(tuple(bl))

    return blocks
