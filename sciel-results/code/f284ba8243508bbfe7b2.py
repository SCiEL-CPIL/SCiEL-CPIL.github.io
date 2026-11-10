"""Covering design C(v,k,t) via greedy set cover with deadline guard."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    all_tset = set(all_tsubs)
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # For each point, precompute which t-subsets contain it
    point_tsubs = defaultdict(set)
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsubs[p].add(i)

    # Greedy set cover: at each step, pick the k-subset that covers the most uncovered t-subsets
    uncovered = set(range(n_tsubs))  # set of indices of uncovered t-subsets

    blocks = []

    def build_block_greedy(start_points=None):
        """Build a single block greedily, maximizing coverage of uncovered t-subsets."""
        if start_points is None:
            # Pick the t-subset with the most "bottleneck" - i.e., the one that is hardest to cover
            # Use the first uncovered one for simplicity, or pick randomly
            first_uncovered = min(uncovered) if uncovered else None
            if first_uncovered is None:
                return []
            block = list(all_tsubs[first_uncovered])
        else:
            block = list(start_points)

        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)

        block.sort()
        return block

    def build_block_random(start_points=None):
        """Build a block with some randomness."""
        if start_points is None:
            first_uncovered = min(uncovered) if uncovered else None
            if first_uncovered is None:
                return []
            block = list(all_tsubs[first_uncovered])
        else:
            block = list(start_points)

        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            # Compute gains
            gains = []
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and idx in uncovered:
                        gain += 1
                gains.append((gain, x))
            # Pick from top candidates with some randomness
            gains.sort(reverse=True)
            top_k = min(3, len(gains))
            _, best_point = gains[rng.randint(0, top_k - 1)]
            block.append(best_point)

        block.sort()
        return block

    def add_block(block):
        """Add a block and update uncovered set."""
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                uncovered.discard(idx)
        blocks.append(block)

    # Main loop: iteratively build blocks
    while uncovered and time.time() < deadline:
        # Try greedy first
        block = build_block_greedy()
        if not block:
            break
        add_block(block)

        # Occasionally try random variants to escape local optima
        if len(blocks) % 10 == 0 and time.time() < deadline:
            # Try a random restart of the current block construction
            pass  # Keep it simple for now

    return blocks
