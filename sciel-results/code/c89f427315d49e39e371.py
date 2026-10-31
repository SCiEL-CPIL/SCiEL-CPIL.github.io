"""Covering design C(19,7,4) solver: greedy init + simulated annealing local search."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import math

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_remaining():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    random.seed(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute bitmask for each possible k-subset (block)
    block_bitmasks = {}
    for blk in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(blk, t):
            mask |= (1 << tsub_index[ts])
        block_bitmasks[blk] = mask

    full_mask = (1 << n_tsubs) - 1

    # --- Phase 1: Greedy construction ---
    def _greedy_construct():
        uncovered = full_mask
        blocks = []
        while uncovered and _time_remaining() > 10.0:
            # Find first uncovered t-subset
            idx = (uncovered & -uncovered).bit_length() - 1
            anchor = all_tsubs[idx]
            block = list(anchor)
            block_set = set(block)
            while len(block) < k:
                if _time_remaining() < 10.0:
                    break
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        bit = 1 << tsub_index[ts]
                        if uncovered & bit:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
            block.sort()
            blocks.append(block)
            uncovered &= ~block_bitmasks[tuple(block)]
        return blocks

    blocks = _greedy_construct()

    # --- Helper: compute coverage mask for a list of blocks ---
    def _coverage_mask(blks):
        mask = 0
        for b in blks:
            mask |= block_bitmasks[tuple(b)]
        return mask

    # --- Helper: check if solution is valid ---
    def _is_valid(blks):
        return _coverage_mask(blks) == full_mask

    # --- Phase 2: Simulated Annealing ---
    def _simulated_annealing(blocks):
        """SA with moves: add block, remove block, swap point in block."""
        if not _is_valid(blocks):
            # Fix validity first with greedy
            blocks = _greedy_construct()
        
        best_blocks = list(blocks)
        best_cost = len(blocks)
        current_cost = best_cost
        current_blocks = list(blocks)

        T = 5.0  # initial temperature
        T_min = 0.01
        cooling = 0.995
        max_steps = 50000

        step = 0
        while _time_remaining() > 5.0 and step < max_steps:
            step += 1
            move_type = random.choice(['add', 'remove', 'swap'])

            if move_type == 'add':
                # Add a random block
                if len(current_blocks) > 100:  # cap
                    continue
                new_block = list(random.sample(range(v), k))
                new_block.sort()
                candidate = current_blocks + [new_block]
                new_cost = len(candidate)
                # Check if valid
                if _is_valid(candidate):
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_blocks = candidate
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                # Also try adding a block that covers an uncovered t-subset
                cov = _coverage_mask(current_blocks)
                if cov != full_mask:
                    uncov = full_mask & ~cov
                    idx = (uncov & -uncov).bit_length() - 1
                    anchor = all_tsubs[idx]
                    # Build a block around anchor
                    block = list(anchor)
                    block_set = set(block)
                    while len(block) < k:
                        candidates = [x for x in range(v) if x not in block_set]
                        if not candidates:
                            break
                        # Pick point that maximizes coverage of uncovered
                        best_x = None
                        best_gain = -1
                        for x in candidates:
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                ts = tuple(sorted(rest + (x,)))
                                bit = 1 << tsub_index[ts]
                                if uncov & bit:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        block.append(best_x)
                        block_set.add(best_x)
                    if len(block) < k:
                        for x in range(v):
                            if len(block) >= k:
                                break
                            if x not in block_set:
                                block.append(x)
                                block_set.add(x)
                    block.sort()
                    candidate = current_blocks + [block]
                    new_cost = len(candidate)
                    if _is_valid(candidate):
                        delta = new_cost - current_cost
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            current_blocks = candidate
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_blocks = list(current_blocks)

            elif move_type == 'remove':
                # Try removing a random block
                if len(current_blocks) <= 1:
                    continue
                i = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:i] + current_blocks[i+1:]
                new_cost = len(candidate)
                if _is_valid(candidate):
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_blocks = candidate
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                else:
                    # Reject with probability based on how many t-subsets lost
                    cov_new = _coverage_mask(candidate)
                    cov_old = _coverage_mask(current_blocks)
                    lost = cov_old & ~cov_new
                    n_lost = bin(lost).count('1')
                    penalty = n_lost * 0.1
                    delta = new_cost - current_cost + penalty
                    if random.random() < math.exp(-delta / T):
                        current_blocks = candidate
                        current_cost = new_cost

            else:  # swap
                # Swap a point in a random block with a random other point
                if len(current_blocks) == 0:
                    continue
                i = random.randint(0, len(current_blocks) - 1)
                bi = current_blocks[i]
                pos = random.randint(0, k - 1)
                orig = bi[pos]
                # Pick a new point
                candidates = [x for x in range(v) if x != orig]
                if not candidates:
                    continue
                new_pt = random.choice(candidates)
                # Check if new_pt is already in the block
                if new_pt in bi:
                    continue
                new_bi = bi[:pos] + [new_pt] + bi[pos+1:]
                new_bi.sort()
                candidate = current_blocks[:i] + [new_bi] + current_blocks[i+1:]
                # Ensure no duplicate blocks
                seen = set()
                valid_candidate = True
                for b in candidate:
                    tb = tuple(b)
                    if tb in seen:
                        valid_candidate = False
                        break
                    seen.add(tb)
                if not valid_candidate:
                    continue
                new_cost = len(candidate)
                if _is_valid(candidate):
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_blocks = candidate
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                else:
                    cov_new = _coverage_mask(candidate)
                    cov_old = _coverage_mask(current_blocks)
                    lost = cov_old & ~cov_new
                    n_lost = bin(lost).count('1')
                    penalty = n_lost * 0.05
                    delta = new_cost - current_cost + penalty
                    if random.random() < math.exp(-delta / T):
                        current_blocks = candidate
                        current_cost = new_cost

            # Cool down
            T = max(T_min, T * cooling)

        return best_blocks

    blocks = _simulated_annealing(blocks)

    # --- Phase 3: Final cleanup - remove redundant blocks ---
    def _remove_redundant(blocks):
        changed = True
        while changed and _time_remaining() > 2.0:
            changed = False
            n = len(blocks)
            for i in range(n):
                if _time_remaining() < 2.0:
                    break
                bi = blocks[i]
                redundant = True
                for ts in itertools.combinations(bi, t):
                    found = False
                    for j in range(n):
                        if j == i:
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    blocks = _remove_redundant(blocks)

    # Ensure validity
    if not _is_valid(blocks):
        # Fallback to greedy
        blocks = _greedy_construct()
        blocks = _remove_redundant(blocks)

    return blocks
# EVOLVE-BLOCK-END
