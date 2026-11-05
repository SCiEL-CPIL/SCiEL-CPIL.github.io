import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin under 120s
    rng = random.Random(12345)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # count[i] = number of blocks covering t-subset i
    count = [0] * n_tsubs
    uncovered_count = n_tsubs

    def block_tsubs(block):
        """Return list of t-subset indices covered by this block."""
        return [tsub_index[ts] for ts in itertools.combinations(block, t)]

    # ---- Phase 1: Greedy construction ----
    blocks = []
    uncovered = set(range(n_tsubs))

    while uncovered and time.time() < deadline - 30:
        # Pick the "hardest" uncovered t-subset: one with fewest candidate blocks
        # For simplicity, pick one at random from uncovered
        start_ts = next(iter(uncovered))
        start_points = list(all_tsubs[start_ts])
        block = list(start_points)

        # Greedily add points to maximize marginal gain (number of newly covered t-subsets)
        while len(block) < k and time.time() < deadline - 30:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of uncovered t-subsets that include x and
                # (block minus one point) such that the t-subset is in uncovered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_index.get(ts, -1)
                    if idx != -1 and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1:
                break
            block.append(best_x)

        block.sort()
        # Update counts and uncovered
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            count[idx] += 1
            if count[idx] == 1:
                uncovered.discard(idx)
        blocks.append(block)

    # ---- Phase 2: Simulated Annealing ----
    # Precompute for each block the list of t-subset indices it covers
    block_tsub_lists = [block_tsubs(b) for b in blocks]

    def fitness():
        """Calculate fitness: number of uncovered t-subsets + number of blocks."""
        return uncovered_count + len(blocks)

    best_blocks = blocks[:]
    best_fitness = fitness()
    current_temperature = 1000.0
    cooling_rate = 0.9999

    while time.time() < deadline - 10:
        # Choose a move
        move_type = rng.choice(['add', 'remove', 'swap', 'mutate'])
        new_blocks = blocks[:]
        new_block_tsub_lists = block_tsub_lists[:]
        new_count = count[:]
        new_uncovered_count = uncovered_count
        improved = False

        if move_type == 'add':
            # Add a random block
            new_block = sorted(rng.sample(range(v), k))
            new_block_ts = block_tsubs(new_block)
            for idx in new_block_ts:
                new_count[idx] += 1
                if new_count[idx] == 1:
                    new_uncovered_count -= 1
            new_blocks.append(new_block)
            new_block_tsub_lists.append(new_block_ts)

        elif move_type == 'remove':
            # Remove a random block if it doesn't uncover any t-subsets
            if not blocks:
                continue
            bi = rng.randint(0, len(blocks) - 1)
            if all(new_count[idx] > 1 for idx in block_tsub_lists[bi]):
                for idx in block_tsub_lists[bi]:
                    new_count[idx] -= 1
                    if new_count[idx] == 1:
                        new_uncovered_count += 1
                new_blocks.pop(bi)
                new_block_tsub_lists.pop(bi)

        elif move_type == 'swap':
            # Replace a random block with a new random block
            if not blocks:
                continue
            bi = rng.randint(0, len(blocks) - 1)
            old_block_ts = block_tsub_lists[bi]
            for idx in old_block_ts:
                new_count[idx] -= 1
                if new_count[idx] == 1:
                    new_uncovered_count += 1
            new_block = sorted(rng.sample(range(v), k))
            new_block_ts = block_tsubs(new_block)
            for idx in new_block_ts:
                new_count[idx] += 1
                if new_count[idx] == 1:
                    new_uncovered_count -= 1
            new_blocks[bi] = new_block
            new_block_tsub_lists[bi] = new_block_ts

        elif move_type == 'mutate':
            # Randomly change one element in a random block
            if not blocks:
                continue
            bi = rng.randint(0, len(blocks) - 1)
            old_block = blocks[bi][:]
            old_block_ts = block_tsub_lists[bi]
            for idx in old_block_ts:
                new_count[idx] -= 1
                if new_count[idx] == 1:
                    new_uncovered_count += 1
            pos = rng.randint(0, k - 1)
            new_elem = rng.choice([x for x in range(v) if x not in old_block])
            old_block[pos] = new_elem
            old_block.sort()
            new_block_ts = block_tsubs(old_block)
            for idx in new_block_ts:
                new_count[idx] += 1
                if new_count[idx] == 1:
                    new_uncovered_count -= 1
            new_blocks[bi] = old_block
            new_block_tsub_lists[bi] = new_block_ts

        new_fitness = new_uncovered_count + len(new_blocks)
        if new_fitness < best_fitness:
            best_blocks = new_blocks[:]
            best_fitness = new_fitness

        # Accept the move based on simulated annealing
        if new_fitness <= fitness() or rng.random() < pow(2.718, -(new_fitness - fitness()) / current_temperature):
            blocks = new_blocks
            block_tsub_lists = new_block_tsub_lists
            count = new_count
            uncovered_count = new_uncovered_count

        current_temperature *= cooling_rate

    # Final pass: remove any remaining redundant blocks
    for bi in range(len(blocks) - 1, -1, -1):
        if time.time() > deadline - 5:
            break
        if all(count[idx] > 1 for idx in block_tsub_lists[bi]):
            for idx in block_tsub_lists[bi]:
                count[idx] -= 1
            blocks.pop(bi)
            block_tsub_lists.pop(bi)

    return blocks
