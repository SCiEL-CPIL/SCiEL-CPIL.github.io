"""Direct structural construction for covering design C(24,7,3)."""
import itertools
import random
import time
import math


def _precompute_triples(v, t):
    """Precompute all t-subsets and map them to indices."""
    triples = []
    triple_to_idx = {}
    for combo in itertools.combinations(range(v), t):
        idx = len(triples)
        triple_to_idx[combo] = idx
        triples.append(combo)
    return triples, triple_to_idx


def _block_triple_indices(block, triple_to_idx):
    """Get the set of triple indices covered by a block."""
    idxs = set()
    for combo in itertools.combinations(block, 3):
        if combo in triple_to_idx:
            idxs.add(triple_to_idx[combo])
    return idxs


def _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx, max_triples):
    """
    Bitmask-based greedy construction.
    Returns a list of blocks (tuples).
    """
    n_triples = len(triples)
    # Use a bit array for uncovered triples
    uncovered = bytearray([1] * n_triples)
    uncovered_count = n_triples
    blocks = []
    seen_blocks = set()

    while uncovered_count > 0:
        # Find the triple with highest "need" - for now, just pick the first uncovered
        # Better: pick the uncovered triple that is hardest to cover
        # For speed, just pick the first uncovered one
        seed_idx = -1
        for i in range(n_triples):
            if uncovered[i]:
                seed_idx = i
                break

        if seed_idx < 0:
            break

        seed_triple = triples[seed_idx]
        block = list(seed_triple)

        # Greedily extend the block
        while len(block) < k:
            best_x = None
            best_gain = -1
            best_tie = -1

            for x in range(v):
                if x in block:
                    continue

                # Calculate gain: how many new triples would be covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(rest + (x,)))
                    idx = triple_to_idx.get(new_triple, -1)
                    if idx >= 0 and uncovered[idx]:
                        gain += 1

                # Tie-breaker: prefer points that appear in more uncovered triples
                if gain > best_gain or (gain == best_gain and best_x is not None and gain > 0):
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # Add random point to fill
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    best_x = rng.choice(candidates)
                else:
                    break

            block.append(best_x)

        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in seen_blocks:
            seen_blocks.add(b_tuple)
            blocks.append(b_tuple)
            # Update uncovered
            for combo in itertools.combinations(b_tuple, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and uncovered[idx]:
                    uncovered[idx] = 0
                    uncovered_count -= 1

    return blocks


def _local_search_sa(blocks, v, k, t, triples, triple_to_idx, time_limit=100.0):
    """
    Simulated annealing local search to minimize block count.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    rng = random.Random(42)

    best_blocks = list(blocks)
    best_n = len(best_blocks)
    current_blocks = list(blocks)
    current_n = len(current_blocks)

    # Precompute coverage counts for current solution
    def compute_counts(blk_list):
        counts = [0] * len(triples)
        for b in blk_list:
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0:
                    counts[idx] += 1
        return counts

    def is_valid_counts(counts):
        return all(c >= 1 for c in counts)

    current_counts = compute_counts(current_blocks)

    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    cooling = 0.9995

    iteration = 0
    while time.time() < deadline:
        iteration += 1
        T = T_start * (cooling ** iteration)
        if T < T_end:
            T = T_end

        # Choose a random move
        move_type = rng.random()
        if move_type < 0.4:
            # Try to remove a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            b = current_blocks[i]
            # Check if removal keeps validity
            removable = True
            for combo in itertools.combinations(b, t):
                idx = triple_to_idx.get(combo, -1)
                if idx >= 0 and current_counts[idx] <= 1:
                    removable = False
                    break
            if removable:
                new_blocks = current_blocks[:i] + current_blocks[i+1:]
                new_counts = compute_counts(new_blocks)
                if is_valid_counts(new_counts):
                    current_blocks = new_blocks
                    current_counts = new_counts
                    current_n = len(current_blocks)
                    if current_n < best_n:
                        best_n = current_n
                        best_blocks = list(current_blocks)
                    continue
        elif move_type < 0.7:
            # Try to swap a block
            if len(current_blocks) <= 1:
                continue
            i = rng.randrange(len(current_blocks))
            old_b = current_blocks[i]
            old_set = set(old_b)

            # Generate a new block by perturbing the old one
            new_set = set()
            for _ in range(k):
                candidates = [x for x in range(v) if x not in new_set]
                old_cands = [x for x in old_set if x not in new_set]
                if old_cands and rng.random() < 0.6:
                    x = rng.choice(old_cands)
                else:
                    x = rng.choice(candidates)
                new_set.add(x)

            new_b = tuple(sorted(new_set))
            if new_b == old_b:
                continue

            # Check if new_b is already in the list
            if new_b in current_blocks:
                continue

            new_blocks = current_blocks[:]
            new_blocks[i] = new_b
            new_counts = compute_counts(new_blocks)

            if is_valid_counts(new_counts):
                current_blocks = new_blocks
                current_counts = new_counts
                # Block count unchanged, but we might enable future removals
                if current_n < best_n:
                    best_n = current_n
                    best_blocks = list(current_blocks)
        else:
            # Try to add a block (to escape local minima)
            if current_n > 100:  # Don't add too many
                continue
            # Generate a random block
            new_b = tuple(sorted(rng.sample(range(v), k)))
            if new_b in current_blocks:
                continue

            new_blocks = current_blocks + [new_b]
            new_counts = compute_counts(new_blocks)
            # This is always valid since we're adding coverage
            current_blocks = new_blocks
            current_counts = new_counts
            current_n = len(current_blocks)

        # Occasional restart to best if stuck
        if iteration % 5000 == 0 and current_n > best_n + 5:
            current_blocks = list(best_blocks)
            current_counts = compute_counts(current_blocks)
            current_n = best_n

    return best_blocks, best_n


def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for 120s limit

    rng = random.Random(12345)

    # Precompute triples
    triples, triple_to_idx = _precompute_triples(v, t)
    max_triples = len(triples)

    # Step 1: Multiple greedy restarts to find a good initial covering
    best_greedy = None
    best_greedy_n = float('inf')

    greedy_time = 20.0  # Spend up to 20s on greedy restarts
    greedy_deadline = start_time + greedy_time

    for restart in range(15):
        if time.time() > greedy_deadline:
            break
        r = random.Random(12345 + restart * 7919)
        blocks = _greedy_cover_bitmask(v, k, t, r, triples, triple_to_idx, max_triples)
        if len(blocks) < best_greedy_n:
            best_greedy_n = len(blocks)
            best_greedy = blocks

    if best_greedy is None:
        best_greedy = _greedy_cover_bitmask(v, k, t, rng, triples, triple_to_idx, max_triples)

    # Step 2: Simulated annealing local search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        best_blocks, best_n = _local_search_sa(best_greedy, v, k, t, triples, triple_to_idx, time_limit=remaining_time - 5.0)
    else:
        best_blocks = best_greedy
        best_n = len(best_greedy)

    return [list(b) for b in best_blocks]


if __name__ == "__main__":
    import time
    start = time.time()
    result = construct(24, 7, 3)
    elapsed = time.time() - start
    print(f"Blocks: {len(result)}, Time: {elapsed:.2f}s")
    # Verify
    covered = set()
    for b in result:
        for c in itertools.combinations(b, 3):
            covered.add(frozenset(c))
    total = len(list(itertools.combinations(range(24), 3)))
    print(f"Covered: {len(covered)}/{total}")
    assert len(covered) == total, "Invalid covering!"
