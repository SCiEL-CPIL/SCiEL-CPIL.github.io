# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    # Precompute all t-subsets as bitmasks per point
    t_subs = list(itertools.combinations(range(v), t))
    n = len(t_subs)
    sub_to_id = {s: i for i, s in enumerate(t_subs)}

    # point_masks[p] = bitmask of t-subsets containing p
    point_masks = [0] * v
    for i, s in enumerate(t_subs):
        bit = 1 << i
        for p in s:
            point_masks[p] |= bit

    # all_mask = bitmask of all t-subsets
    all_mask = (1 << n) - 1

    # Precompute for each point p and each (t-1)-subset, the bitmask of t-subsets
    # that contain p plus that (t-1)-subset. This lets us compute marginal gain fast.
    t_minus_1 = t - 1
    t1_subs = list(itertools.combinations(range(v), t_minus_1))
    # For a candidate point x and a partial block B (size < k), the new t-subsets
    # covered are those t-subsets S such that x in S and S\{x} is a (t-1)-subset of B.
    # gain_mask(B, x) = OR over (t-1)-subsets Q of B of (bitmask of t-subs = Q U {x})
    # Precompute: for each x and each (t-1)-subset Q, mask_x_Q
    mask_x_Q = [[0] * len(t1_subs) for _ in range(v)]
    q_to_idx = {q: i for i, q in enumerate(t1_subs)}
    for x in range(v):
        for i, q in enumerate(t1_subs):
            m = 0
            # t-subsets containing x and q: add one point y not in q, y != x, to q
            for y in range(v):
                if y in q or y == x:
                    continue
                s = tuple(sorted(q + (y,)))
                m |= 1 << sub_to_id[s]
            mask_x_Q[x][i] = m

    def build_greedy(seed, uncovered_mask):
        """Greedy: repeatedly open a block on a random uncovered t-subset, fill to k."""
        rng = random.Random(seed)
        blocks = []
        unc = uncovered_mask
        while unc:
            # pick a random set bit from unc
            # get list of set bits (could be large); instead pick random by scanning
            # Use a trick: find a random set bit
            # Convert to find any set bit; for randomness, use rng to pick position
            # We'll get the list of set bit positions lazily
            # For speed, pick the lowest set bit but randomize by shuffling via seed
            # Simpler: pick a random set bit by generating random indices
            # Number of set bits:
            # We'll just pick a random position and check; fallback to lowest
            # To keep it simple and fast, pick lowest set bit but add randomness
            # by permuting the search: use rng to choose a random start
            # Find lowest set bit
            low = unc & (-unc)
            # For diversity, with some probability pick a different random set bit
            if rng.random() < 0.5:
                # pick a random set bit
                # count bits
                # We'll do a few random probes
                for _ in range(8):
                    pos = rng.randrange(n)
                    if (unc >> pos) & 1:
                        low = 1 << pos
                        break
            # The seed t-subset:
            seed_pos = low.bit_length() - 1
            seed_sub = t_subs[seed_pos]
            block = list(seed_sub)
            # Fill block to size k
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    # compute gain mask: t-subs containing x and a (t-1)-subset of block
                    # (t-1)-subsets of block: block has size len(block), need t-1 of them
                    gm = 0
                    for Q in itertools.combinations(block, t_minus_1):
                        gm |= mask_x_Q[x][q_to_idx[Q]]
                    gain = (gm & unc).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_gain <= 0:
                    # no point adds coverage; just pick a random unused point
                    used = set(block)
                    avail = [x for x in range(v) if x not in used]
                    if not avail:
                        break
                    best_x = rng.choice(avail)
                block.append(best_x)
            block.sort()
            blocks.append(block)
            # update uncovered
            bm = 0
            for s in itertools.combinations(block, t):
                bm |= 1 << sub_to_id[s]
            unc &= ~bm
        return blocks

    def count_blocks(blocks):
        return len(blocks)

    best_blocks = None
    best_count = None

    # Initial deterministic run
    blocks0 = build_greedy(0, all_mask)
    best_blocks = blocks0
    best_count = count_blocks(blocks0)

    # Randomized restarts
    seed = 1
    while True:
        if time.time() > deadline - 5.0:
            break
        blocks = build_greedy(seed, all_mask)
        c = count_blocks(blocks)
        if c < best_count:
            best_count = c
            best_blocks = blocks
        seed += 1

    return best_blocks
# EVOLVE-BLOCK-END
