"""Covering design C(24,7,3): minimize number of 7-subsets of 24 points
so that every 3-subset is contained in some block.
Entry point: construct(v, k, t) -> list of blocks (sorted k-subsets)."""

import itertools
import time
import numpy as np
from math import comb


def construct(v, k, t):
    start = time.time()
    deadline = start + 115.0  # safety margin before 150s kill

    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)

    # block_triple_indices[i, j] = index of j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    # triple_to_blocks: for each triple, which blocks contain it
    triple_to_blocks = [[] for _ in range(n_triples)]
    for i in range(n_blocks):
        for j in range(k_t):
            t_idx = block_triple_indices[i, j]
            triple_to_blocks[t_idx].append(i)

    # triple_freq: how many blocks contain each triple
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    for i in range(n_triples):
        triple_freq[i] = len(triple_to_blocks[i])

    def _greedy_cover(random_seed, weighted=False):
        """Vectorized greedy: repeatedly pick the block covering the most
        still-uncovered triples, with random tie-breaking.
        If weighted=True, use 1/freq weighting to prioritize rare triples."""
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        selected_mask = np.zeros(n_blocks, dtype=bool)

        # Precompute weights
        if weighted:
            weights = 1.0 / (triple_freq.astype(np.float64) + 1e-9)
        else:
            weights = None

        while remaining > 0:
            if time.time() > deadline - 2.0:
                break

            # Compute gains for all unselected blocks
            # uncovered_mask[block_triple_indices] -> (n_blocks, k_t) bool
            if weights is None:
                gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            else:
                # Weighted gain: sum of weights for uncovered triples in block
                um = uncovered_mask[block_triple_indices]  # (n_blocks, k_t) bool
                gains = np.sum(um * weights[block_triple_indices], axis=1)

            gains[selected_mask] = -1

            max_gain = np.max(gains)
            if max_gain <= 0:
                break

            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]

            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))

            # Update: mark newly covered triples
            b_triples = block_triple_indices[best_idx]
            newly = b_triples[uncovered_mask[b_triples]]
            if len(newly) > 0:
                uncovered_mask[newly] = False
                remaining -= len(newly)

        return selected_blocks

    def _local_improve(blocks, random_seed=0, max_rounds=30):
        """Remove redundant blocks and attempt targeted rebuilds to reduce count."""
        if not blocks:
            return blocks

        # Build coverage counts
        all_triples_set = list(itertools.combinations(range(v), t))
        triple_to_idx_local = {trip: i for i, trip in enumerate(all_triples_set)}
        n_t = len(all_triples_set)

        cov = np.zeros(n_t, dtype=np.int32)
        block_triple_sets = []
        for b in blocks:
            trs = list(itertools.combinations(b, t))
            t_indices = np.array([triple_to_idx_local[tr] for tr in trs], dtype=np.int32)
            block_triple_sets.append(t_indices)
            cov[t_indices] += 1

        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            if time.time() > deadline:
                break

            # 1) Drop redundant block
            for i in range(len(block_triple_sets)):
                t_indices = block_triple_sets[i]
                if np.all(cov[t_indices] > 1):
                    cov[t_indices] -= 1
                    block_triple_sets.pop(i)
                    blocks.pop(i)
                    improved = True
                    break

            # 2) Try replacing the "most critical" block
            if not improved and len(blocks) > 1:
                # Find block whose removal would leave the most uncovered triples
                best_i = -1
                best_uncovered = 0
                for i in range(len(block_triple_sets)):
                    t_indices = block_triple_sets[i]
                    unc = np.sum(cov[t_indices] == 1)
                    if unc > best_uncovered:
                        best_uncovered = unc
                        best_i = i

                if best_i >= 0 and best_uncovered > 0:
                    old_block = blocks[best_i]
                    old_triples = block_triple_sets[best_i]
                    cov[old_triples] -= 1

                    # Find uncovered triples
                    uncovered = np.where(cov == 0)[0]
                    if len(uncovered) > 0:
                        # Greedily build a replacement block targeting uncovered triples
                        seed_triple = all_triples_set[uncovered[0]]
                        new_block = list(seed_triple)
                        in_b = set(new_block)

                        # Build remaining k-3 points greedily
                        while len(new_block) < k:
                            best_x = None
                            best_cnt = 0
                            for x in range(v):
                                if x in in_b:
                                    continue
                                # Count how many uncovered triples this x would complete
                                cnt = 0
                                for rest in itertools.combinations(new_block, t - 1):
                                    tr = tuple(sorted(rest + (x,)))
                                    if tr in triple_to_idx_local:
                                        idx = triple_to_idx_local[tr]
                                        if cov[idx] == 0:
                                            cnt += 1
                                if cnt > best_cnt:
                                    best_cnt = cnt
                                    best_x = x
                            if best_x is None:
                                best_x = next(p for p in range(v) if p not in in_b)
                            new_block.append(best_x)
                            in_b.add(best_x)

                        new_block.sort()
                        new_triples = np.array(
                            [triple_to_idx_local[tr] for tr in itertools.combinations(new_block, t)],
                            dtype=np.int32
                        )
                        # Count how many previously-uncovered triples this covers
                        covered_new = np.sum((cov[new_triples] == 0))
                        if covered_new >= best_uncovered:
                            blocks[best_i] = new_block
                            block_triple_sets[best_i] = new_triples
                            cov[new_triples] += 1
                            improved = True
                        else:
                            # Revert
                            blocks[best_i] = old_block
                            block_triple_sets[best_i] = old_triples
                            cov[old_triples] += 1

        return blocks

    def _is_valid(blocks):
        """Check every t-subset is covered."""
        covered = set()
        for b in blocks:
            for tr in itertools.combinations(b, t):
                covered.add(tr)
        return len(covered) == n_triples

    # 2. Run multiple greedy restarts (both unweighted and weighted)
    best_blocks = None
    best_len = float('inf')
    n_restarts = 25

    for i in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        # Alternate between weighted and unweighted
        weighted = (i % 2 == 1)
        blocks = _greedy_cover(i, weighted=weighted)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    # 3. Local improvement on the best
    if best_blocks is not None:
        improved = _local_improve(best_blocks, random_seed=42, max_rounds=40)
        if len(improved) < best_len:
            best_blocks = improved
            best_len = best_len  # already updated

    # 4. Additional restarts + local improvement if time remains
    attempt = n_restarts
    while time.time() < deadline - 5.0:
        weighted = (attempt % 2 == 1)
        blocks = _greedy_cover(attempt, weighted=weighted)
        if len(blocks) < best_len:
            blocks = _local_improve(blocks, random_seed=attempt, max_rounds=20)
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
        attempt += 1
        if attempt > 70:
            break

    # 5. Safety: validate and deduplicate
    if best_blocks is None or not _is_valid(best_blocks):
        # Fallback: single deterministic greedy
        best_blocks = _greedy_cover(999)

    seen = set()
    out = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
