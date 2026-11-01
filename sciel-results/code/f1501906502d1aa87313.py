"""Covering design C(24,7,3) using randomized greedy set cover with post-optimization."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys


def _generate_all_blocks(v, k):
    """Generate all C(v,k) blocks as tuples."""
    return list(itertools.combinations(range(v), k))


def _triple_index(triple, v):
    """Map a 3-tuple to a unique integer index in [0, C(v,3))."""
    a, b, c = triple
    # a < b < c
    # index = C(a,3) + C(b,3) ... actually use combinatorial number system
    # Simpler: use a dictionary or a formula
    # For v=24, C(24,3)=2024, small enough
    # Formula: index = C(c,3) - C(b,3) - C(a,3) ... not standard
    # Let's just use a dict mapping for safety and speed
    pass


def construct(v, k, t):
    """
    Construct a covering design: choose as few k-subsets of v points as possible
    such that every t-subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit

    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)

    # Map each t-subset to an index
    t_sub_index = {sub: i for i, sub in enumerate(all_t_subs)}

    # Precompute for each k-block, which t-subsets it covers
    # Use bitsets for efficiency
    import numpy as np

    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    # Build coverage matrix: for each k-block, a bitset of which t-subsets it covers
    # For v=24, k=7, t=3: C(24,7)=346104, C(24,3)=2024
    # That's 346104 * 2024 bits ~ 346104 * 253 bytes ~ 87 MB. Might be too much.
    # Alternative: for each t-subset, list of k-blocks covering it.
    # Or: for each k-block, compute the list of t-subset indices it covers.

    # Let's use a different approach: for each t-subset, precompute candidate blocks
    # Actually, let's use numpy bitsets if memory allows
    # 346104 blocks * 2024 bits = 346104 * 253 bytes ≈ 87.5 MB. Should be OK.

    # Build: for each k-block, a numpy array of uint64 representing which t-subsets it covers
    # 2024 bits = 32 uint64s

    num_words = (num_t_subs + 63) // 64

    # coverage[block_idx] = numpy array of num_words uint64s
    coverage = np.zeros((num_k_blocks, num_words), dtype=np.uint64)

    for bi, block in enumerate(all_k_blocks):
        for triple in itertools.combinations(block, t):
            idx = t_sub_index[triple]
            coverage[bi, idx // 64] |= (1 << (idx % 64))

    # Now greedy set cover with randomization
    best_blocks = None
    best_count = num_k_blocks + 1
    best_bitset = None

    num_restarts = 50  # will adjust based on time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Randomly shuffle the order of t-subsets for tie-breaking diversity
        order = list(range(num_t_subs))
        random.shuffle(order)

        # Greedy: repeatedly pick the block that covers the most uncovered t-subsets
        # Track uncovered as a bitset
        uncovered_mask = np.ones(num_words, dtype=np.uint64)
        # Set all bits to 1 for valid t-subsets
        for w in range(num_words):
            uncovered_mask[w] = 0xFFFFFFFFFFFFFFFF
        # Clear bits beyond num_t_subs
        if num_t_subs % 64 != 0:
            leftover = num_t_subs % 64
            uncovered_mask[-1] = (1 << leftover) - 1

        selected_blocks = []
        count = 0

        while True:
            # Check if all covered
            if not np.any(uncovered_mask):
                break

            # Find the block that covers the most uncovered t-subsets
            # Compute for each block: popcount(coverage[bi] & uncovered_mask)
            # This is expensive: 346104 blocks * 32 words
            # Optimize: use numpy vectorized operations

            # For each block, compute intersection count
            # coverage & uncovered_mask, then sum popcounts per row
            # numpy doesn't have built-in popcount for uint64, so use lookup or bit tricks

            # Alternative: use a smaller approach. Since this is slow, let's limit restarts
            # and use a smarter initialization.

            # Compute gains
            # For speed, we'll use a precomputed approach or just do it in a loop for now
            # Given time constraints, let's use a simpler greedy with random tie-breaking

            best_gain = -1
            best_bi = -1
            # To speed up, only consider a random subset of blocks or all
            # For now, iterate all but break early if time is tight

            # Vectorized popcount using numpy
            # Use a lookup table for popcount of 16-bit chunks
            # Simpler: use int.bit_count() per word in a loop per block - too slow

            # Let's use a different strategy: instead of full greedy, use a heuristic
            # that picks blocks more efficiently.

            # For this problem size, let's just do the full greedy but with numpy
            # popcount via unpacking

            # Actually, let's precompute popcount for 16-bit values
            # And use numpy to process 4 uint16 per uint64

            # This is getting complex. Let's use a simpler approach:
            # Use Python's int.bit_count() on the combined value per block
            # But that requires converting numpy arrays to Python ints, which is slow.

            # Given the time constraint, let's use a reduced set of restarts and
            # a faster greedy that uses randomization to pick among top candidates.

            # For now, do a full scan but with early termination if time is up
            if time.time() > deadline - 5:
                break

            # Compute gain for each block
            # Use numpy: for each block, sum of (coverage[bi] & uncovered_mask) popcounts
            # We'll use a vectorized approach with a lookup table

            # Precompute popcount for bytes
            popcount_table = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

            # For each block, compute the intersection and popcount
            # This is still O(num_blocks * num_words) which is 346104 * 32 = 11M ops
            # In numpy, this should be fast enough per iteration if we vectorize

            # Let's compute all gains at once
            # intersection = coverage & uncovered_mask[None, :]
            # Then popcount per row

            # Use a trick: unpack uint64 into 8 bytes, look up popcount, sum
            # This is complex in numpy. Let's just do it in a loop for now, but limit
            # the number of blocks we consider.

            # Alternative: use a random subset of blocks for speed
            sample_size = min(num_k_blocks, 5000)
            candidates = random.sample(range(num_k_blocks), sample_size)

            best_gain = -1
            best_bi = candidates[0]
            for bi in candidates:
                # Compute popcount of coverage[bi] & uncovered_mask
                inter = coverage[bi] & uncovered_mask
                # Popcount
                gain = sum(bin(x).count('1') for x in inter)
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi

            if best_gain <= 0:
                # No block covers any uncovered t-subset - shouldn't happen
                break

            # Add the block
            selected_blocks.append(all_k_blocks[best_bi])
            count += 1
            uncovered_mask &= ~coverage[best_bi]

            # Check if complete
            if not np.any(uncovered_mask):
                break

        # End of greedy for this restart
        if best_blocks is None or count < best_count:
            best_blocks = selected_blocks
            best_count = count
            best_bitset = uncovered_mask.copy()

    # Post-optimization: try to remove blocks
    if best_blocks is None:
        # Fallback: return empty (shouldn't happen)
        return []

    # Try to remove redundant blocks
    if time.time() < deadline - 5:
        for i in range(len(best_blocks)):
            if time.time() > deadline - 3:
                break
            # Temporarily remove block i
            remaining = best_blocks[:i] + best_blocks[i+1:]
            # Check if all t-subsets are still covered
            covered = set()
            for block in remaining:
                for triple in itertools.combinations(block, t):
                    covered.add(triple)
            if len(covered) == num_t_subs:
                best_blocks = remaining
                best_count -= 1

    # Convert to lists of ints
    result = [list(block) for block in best_blocks]
    return result
# EVOLVE-BLOCK-END
