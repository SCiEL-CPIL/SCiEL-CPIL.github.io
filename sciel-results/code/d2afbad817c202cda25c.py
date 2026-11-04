"""Covering design C(v,k,t) via strong-greedy max-gain block selection (bitset accelerated)."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s / 150s kill
    
    # Set random seed for reproducibility if needed, or just use system random
    # random.seed(42) 

    # Map every t-subset to an integer id; use a Python int as a bitset for covered.
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_index = {s: i for i, s in enumerate(all_subsets)}

    # pair_mask[x][a][b] = bitmask of t-subsets containing x, a, b (for t=3).
    # Generalize: for t-subsets, precompute mask of t-subsets containing a given (t-1)-tuple.
    # We'll build, for each point x, a dict from frozenset(other t-1 points) -> bitmask of
    # t-subsets containing x and those others.
    # For efficiency with t=3: mask3[x][a][b].
    # Build generic: masks[x] is a dict keyed by tuple of (t-1) other points -> bitmask.
    masks = [dict() for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        bit = 1 << idx
        for i, x in enumerate(s):
            others = tuple(s[:i] + s[i+1:])
            masks[x][others] = masks[x].get(others, 0) | bit

    covered = 0  # bitset of covered t-subsets
    blocks = []

    # Helper to count bits, using bit_count if available (Python 3.10+)
    def count_bits(n):
        return n.bit_count() if hasattr(n, 'bit_count') else bin(n).count('1')

    # For a partial block (list/set B) and candidate x not in B, compute gain =
    # number of uncovered t-subsets that would become covered by adding x.
    # A t-subset {x} ∪ S (|S|=t-1, S ⊆ B) is newly covered iff it contains x and S ⊆ B.
    def gain_of_add(x, B_set):
        g = 0
        # iterate over (t-1)-subsets of B
        B_list = list(B_set)
        for S in itertools.combinations(B_list, t - 1):
            key = tuple(S)
            m = masks[x].get(key)
            if m is not None:
                g += count_bits(m & ~covered)
        return g

    def build_block_from(seed_points):
        # Grow a block from an initial set to size k by max-gain greedy.
        B = set(seed_points)
        # List of points not in B
        available = [x for x in range(v) if x not in B]
        while len(B) < k:
            best_x = -1
            best_gain = -1
            for x in available:
                g = gain_of_add(x, B)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x == -1:
                break
            B.add(best_x)
            available.remove(best_x)
        return B

    while True:
        if time.time() > deadline:
            break

        # Find the best block overall: try seeding from each uncovered t-subset,
        # but to limit cost, evaluate a representative set of seeds.
        # Strong-greedy: pick the block with max total new coverage.
        # We enumerate candidate seeds = each uncovered t-subset (as the "forced" core),
        # grow each, and keep the best. To bound runtime, sample seeds.

        # Determine uncovered subsets
        uncov_mask = ((1 << n_subsets) - 1) & ~covered
        if uncov_mask == 0:
            break

        # Collect uncovered subset indices (bounded sample)
        # Find them by scanning bits
        uncovered_indices = []
        m = uncov_mask
        while m:
            # lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            m ^= lsb
            # No hard cap here yet, we will sample from this list

        if not uncovered_indices:
            break

        # Randomly sample seeds to explore different regions of the search space
        # and avoid getting stuck in a deterministic local optimum.
        # Sample size: trade-off between exploration and computation time.
        sample_size = min(100, len(uncovered_indices))
        sampled_indices = random.sample(uncovered_indices, sample_size)

        best_block = None
        best_gain_total = -1

        for idx in sampled_indices:
            seed = all_subsets[idx]
            B = build_block_from(set(seed))
            # compute total new coverage of B
            total = 0
            for s in itertools.combinations(sorted(B), t):
                total += 1 if not (covered >> subset_to_index[s]) & 1 else 0
            if total > best_gain_total:
                best_gain_total = total
                best_block = B

        if best_block is None:
            break

        bl = sorted(best_block)
        # mark covered
        for s in itertools.combinations(bl, t):
            covered |= 1 << subset_to_index[s]
        blocks.append(bl)
        
        # Optional: If time is running out, break early?
        # The deadline check at the top of the loop handles this.

    # Post-processing: Remove redundant blocks
    # A block is redundant if every triplet it covers is also covered by at least one other block.
    if blocks:
        # Create a list of (block_id, list_of_triplet_indices)
        block_triplets = []
        for b in blocks:
            tps = [subset_to_index[s] for s in itertools.combinations(b, t)]
            block_triplets.append(tps)
        
        # For each block, check if it has a "unique" triplet
        # To do this efficiently, we can count how many blocks cover each triplet.
        # triplet_count[i] = number of blocks covering triplet i
        triplet_count = [0] * n_subsets
        for tps in block_triplets:
            for idx in tps:
                triplet_count[idx] += 1
        
        # Now, a block is redundant if for all its triplets idx, triplet_count[idx] > 1
        # AND removing it doesn't uncover any triplet.
        # Wait, if triplet_count[idx] > 1, it means at least one *other* block covers it.
        # So if ALL triplets in a block have count > 1, the block is redundant.
        
        new_blocks = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for idx in block_triplets[i]:
                if triplet_count[idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(b)
        
        # If we removed blocks, we might be able to remove more?
        # One pass is usually sufficient for simple redundancy.
        # We could loop this removal process if time permits.
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            # Recalculate covered? No, the set of covered triplets hasn't changed.
            # But we should update the block list.

    return blocks
