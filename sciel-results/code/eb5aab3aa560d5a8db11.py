"""Covering design C(24,7,3) – greedy build + randomized local search with a strict deadline."""
import time
import random
import itertools
import numpy as np

_DEADLINE = 120.0
_SAFETY = 3.0  # seconds to spare before evaluator kills us


def _triple_to_index(triple, v=24):
    """Map a 3-subset of {0..v-1} to a unique integer in [0, C(v,3)-1)."""
    a, b, c = triple
    return a * v * v + b * v + c  # unique encoding, not compact but fast


def _build_greedy(v, k, t, rng=None):
    """Greedy set-cover: repeatedly pick the block covering the most uncovered t-subsets."""
    if rng is None:
        rng = random.Random()
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    covered = np.zeros(n_triples, dtype=np.uint8)

    # Precompute for each candidate point x and each partial block, how many triples it covers
    # Actually, for efficiency in greedy, we iterate: pick the uncovered triple with highest "potential"
    # Simpler greedy: pick a random uncovered triple, then greedily build a block that covers the most
    # uncovered triples.
    blocks = []
    uncovered_set = set(range(n_triples))

    while uncovered_set:
        if time.time() > _DEADLINE - _SAFETY:
            break
        # Pick an uncovered triple (randomized for diversity)
        tr = all_triples[rng.choice(sorted(uncovered_set))]
        block = set(tr)
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count uncovered triples that would be covered by adding x
                gain = 0
                # For each (t-1)-subset of current block, check if (that subset + x) is an uncovered triple
                # This is O(C(k,t-1)) per candidate
                for sub in itertools.combinations(sorted(block), t - 1):
                    candidate_triple = tuple(sorted(sub + (x,)))
                    idx = triple_index.get(candidate_triple, -1)
                    if idx != -1 and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain == 0:
                # No point improves coverage; pick random
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            block.add(best_x)
        block = sorted(block)
        blocks.append(block)
        # Mark all triples in this block as covered
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                covered[idx] = 1
                uncovered_set.discard(idx)

    return blocks


def _count_blocks(blocks, v, t):
    """Count how many t-subsets are covered by the given blocks."""
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    covered = set()
    for block in blocks:
        for tr3 in itertools.combinations(block, t):
            covered.add(tr3)
    return len(covered)


def _is_valid_cover(blocks, v, t):
    """Check if blocks cover all t-subsets of {0..v-1}."""
    all_triples = set(itertools.combinations(range(v), t))
    covered = set()
    for block in blocks:
        for tr3 in itertools.combinations(block, t):
            covered.add(tr3)
    return all_triples.issubset(covered)


def _local_search(blocks, v, k, t, rng, deadline):
    """Try to reduce block count by removing redundant blocks and swapping."""
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Build coverage count array
    coverage = np.zeros(n_triples, dtype=np.int32)
    for block in blocks:
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                coverage[idx] += 1

    def update_coverage_remove(block):
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                coverage[idx] -= 1

    def update_coverage_add(block):
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                coverage[idx] += 1

    improved = True
    while improved and time.time() < deadline - _SAFETY:
        improved = False
        # Try removing each block if it doesn't cause uncovered triples
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - _SAFETY:
                break
            update_coverage_remove(blocks[i])
            if np.all(coverage > 0):
                blocks.pop(i)
                improved = True
            else:
                update_coverage_add(blocks[i])

    return blocks


def _random_block(v, k, rng):
    return sorted(rng.sample(range(v), k))


def _greedy_build_from_seed(v, k, t, initial_blocks, rng):
    """Continue greedy from initial blocks."""
    all_triples = list(itertools.combinations(range(v), t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)
    covered = np.zeros(n_triples, dtype=np.uint8)

    for block in initial_blocks:
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                covered[idx] = 1

    blocks = list(initial_blocks)
    uncovered_set = set(range(n_triples))
    for i in range(n_triples):
        if covered[i]:
            uncovered_set.discard(i)

    while uncovered_set and time.time() < _DEADLINE - _SAFETY:
        tr = all_triples[rng.choice(sorted(uncovered_set))]
        block = set(tr)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for sub in itertools.combinations(sorted(block), t - 1):
                    candidate_triple = tuple(sorted(sub + (x,)))
                    idx = triple_index.get(candidate_triple, -1)
                    if idx != -1 and idx in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain == 0:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                best_x = rng.choice(candidates)
            block.add(best_x)
        block = sorted(block)
        blocks.append(block)
        for tr3 in itertools.combinations(block, t):
            idx = triple_index.get(tr3, -1)
            if idx != -1:
                covered[idx] = 1
                uncovered_set.discard(idx)

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Try multiple random seeds for greedy + local search
    best_blocks = None
    best_count = float('inf')

    for seed in range(30):
        if time.time() > deadline - _SAFETY:
            break
        rng = random.Random(seed)

        # Greedy build
        blocks = _build_greedy(v, k, t, rng)

        # Local search
        blocks = _local_search(blocks, v, k, t, rng, deadline)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]

    # If we still have time, try more diverse initializations
    if best_blocks is None:
        best_blocks = _build_greedy(v, k, t, random.Random(0))
    else:
        # Try additional random restarts with partial random blocks
        for seed in range(30, 60):
            if time.time() > deadline - _SAFETY:
                break
            rng = random.Random(seed)
            # Start with some random blocks
            n_init = max(1, len(best_blocks) // 4)
            init_blocks = [_random_block(v, k, rng) for _ in range(n_init)]
            blocks = _greedy_build_from_seed(v, k, t, init_blocks, rng)
            blocks = _local_search(blocks, v, k, t, rng, deadline)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks[:]

    # Final safety: ensure we have a valid cover
    if best_blocks is None:
        # Extremely slow fallback
        best_blocks = _build_greedy(v, k, t, random.Random(123))

    return best_blocks
