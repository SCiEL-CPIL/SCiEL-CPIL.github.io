"""Improved covering design C(19,8,5) via randomized greedy + aggressive local search."""
import itertools
import random
import time
import numpy as np


def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings."""
    tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}

    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)

    # For each block, store the indices of t-subsets it covers
    block_tsubs = {}
    for b in blocks:
        idxs = np.array([tsub_idx[ts] for ts in itertools.combinations(b, t)], dtype=np.int32)
        block_tsubs[b] = idxs

    # For each t-subset, store the list of blocks containing it
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tsub_blocks[tsub_idx[ts]].append(b)

    return tsubs, tsub_idx, blocks, block_tsubs, tsub_blocks, n_tsubs, n_blocks


def _greedy_cover(v, k, t, rng, precomp):
    """Randomized greedy set cover using precomputed data."""
    tsubs, tsub_idx, blocks, block_tsubs, tsub_blocks, n_tsubs, n_blocks = precomp

    cov = np.zeros(n_tsubs, dtype=np.int32)
    uncovered_count = n_tsubs
    selected = []
    uncovered_set = set(range(n_tsubs))

    while uncovered_count > 0:
        # Pick a random uncovered t-subset
        target = rng.choice(list(uncovered_set))
        candidates = tsub_blocks[target]

        best_block = None
        best_gain = -1

        for b in candidates:
            idxs = block_tsubs[b]
            gain = 0
            for idx in idxs:
                if idx in uncovered_set:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b

        if best_block is None:
            break

        selected.append(best_block)
        idxs = block_tsubs[best_block]
        for idx in idxs:
            if idx in uncovered_set:
                uncovered_set.discard(idx)
                uncovered_count -= 1

    return selected


def _local_search(v, k, t, blocks, rng, precomp, time_limit=100.0):
    """Aggressive local search: delete redundant, swap to reduce count."""
    start_time = time.time()
    tsubs, tsub_idx, all_blocks, block_tsubs, tsub_blocks, n_tsubs, n_blocks = precomp

    blocks = [tuple(b) for b in blocks]

    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for b in blocks:
        idxs = block_tsubs[b]
        cov[idxs] += 1

    best_blocks = list(blocks)
    best_len = len(blocks)

    while time.time() - start_time < time_limit:
        improved = False

        # Phase 1: Delete redundant blocks (pass multiple times)
        for _pass in range(5):
            if time.time() - start_time > time_limit:
                break
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() - start_time > time_limit:
                    break
                b = blocks[i]
                idxs = block_tsubs[b]
                if np.all(cov[idxs] > 1):
                    cov[idxs] -= 1
                    blocks.pop(i)
                    improved = True
                    break
            if not improved and len(blocks) == len(best_blocks):
                break
            if improved:
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                # After deletion, loop again to find more

        # Phase 2: Swap blocks
        if len(blocks) == 0:
            break

        found_swap = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break

            i = rng.randrange(len(blocks))
            b = blocks[i]
            idxs_b = block_tsubs[b]

            critical_mask = cov[idxs_b] == 1
            critical_idxs = idxs_b[critical_mask]

            if len(critical_idxs) == 0:
                # This block is redundant now
                cov[idxs_b] -= 1
                blocks.pop(i)
                improved = True
                found_swap = True
                break

            # Find replacement covering all critical t-subsets
            first_crit = critical_idxs[0]
            candidates = tsub_blocks[first_crit]

            # Shuffle candidates for randomness
            cand_list = list(candidates)
            rng.shuffle(cand_list)

            replacement = None
            for b_prime in cand_list:
                if b_prime == b:
                    continue
                idxs_bp = block_tsubs[b_prime]
                # Check if b' covers all critical t-subsets
                ok = True
                for ci in critical_idxs:
                    if ci not in idxs_bp:
                        ok = False
                        break
                if ok:
                    replacement = b_prime
                    break

            if replacement is not None:
                idxs_bp = block_tsubs[replacement]
                cov[idxs_b] -= 1
                cov[idxs_bp] += 1
                blocks[i] = replacement
                improved = True
                found_swap = True
                break

        if not found_swap and not improved:
            # Try a "kick": remove a random block, then greedily re-add
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                b = blocks[i]
                idxs_b = block_tsubs[b]
                cov[idxs_b] -= 1
                blocks.pop(i)

                # Re-add blocks to cover newly uncovered t-subsets
                newly_uncovered = np.where(cov == 0)[0]
                for nu in newly_uncovered:
                    cands = tsub_blocks[nu]
                    if not cands:
                        break
                    best_b = None
                    best_gain = -1
                    for cb in cands:
                        if cb in blocks:
                            continue
                        idxs_cb = block_tsubs[cb]
                        gain = 0
                        for idx in idxs_cb:
                            if cov[idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_b = cb
                    if best_b is not None:
                        blocks.append(best_b)
                        idxs_cb = block_tsubs[best_b]
                        cov[idxs_cb] += 1
                improved = True

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)

        if not improved:
            break

    return best_blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 145.0

    precomp = _precompute(v, k, t)

    best_blocks = None
    best_count = float('inf')

    # Multiple randomized greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        rng = random.Random(r * 104729 + 7)
        blocks = _greedy_cover(v, k, t, rng, precomp)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

    # Local search on the best solution
    if best_blocks is not None:
        if time.time() < deadline - 5:
            rng = random.Random(12345)
            remaining = min(110.0, deadline - time.time() - 3.0)
            best_blocks = _local_search(v, k, t, best_blocks, rng, precomp, time_limit=remaining)

    # Final validation and cleanup
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, precomp)

    seen = set()
    final_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))

    return final_blocks
