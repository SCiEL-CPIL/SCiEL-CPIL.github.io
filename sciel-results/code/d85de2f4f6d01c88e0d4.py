import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 118.0
    rng = random.Random(123456789)
    np_rng = np.random.default_rng(123456789)

    # --- enumerate all t-subsets (columns) ---
    all_subs = list(itertools.combinations(range(v), t))
    n_sub = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    # --- enumerate all k-blocks (rows) ---
    all_blocks = list(itertools.combinations(range(v), k))
    n_blk = len(all_blocks)

    # --- incidence: block -> list of t-subset indices it covers ---
    block_subs = [[] for _ in range(n_blk)]
    for bi, b in enumerate(all_blocks):
        for s in itertools.combinations(b, t):
            block_subs[bi].append(sub_to_idx[s])

    # --- incidence: t-subset -> list of blocks covering it ---
    sub_blocks = [[] for _ in range(n_sub)]
    for bi, lst in enumerate(block_subs):
        for si in lst:
            sub_blocks[si].append(bi)

    # --- Phase 1: vectorized greedy set cover ---
    covered = np.zeros(n_sub, dtype=bool)
    gains = np.array([len(lst) for lst in block_subs], dtype=np.int64)
    chosen = []
    chosen_set = set()

    while time.time() < deadline:
        if covered.all():
            break
        g = gains.copy()
        g[covered] = 0
        # light random tie-break for diversity
        g = g + np_rng.random(n_blk) * 0.001
        bi = int(np.argmax(g))
        if g[bi] <= 0:
            break
        chosen.append(bi)
        chosen_set.add(bi)
        for si in block_subs[bi]:
            if not covered[si]:
                covered[si] = True
                for bj in sub_blocks[si]:
                    gains[bj] -= 1

    # --- coverage counts for the chosen set ---
    cov = np.zeros(n_sub, dtype=np.int64)
    for bi in chosen:
        for si in block_subs[bi]:
            cov[si] += 1

    best = list(chosen)
    best_len = len(best)

    # --- Phase 2: redundant-block deletion ---
    def delete_pass(blocks, cov, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(blocks):
                    continue
                bi = blocks[pos]
                redundant = True
                for si in block_subs[bi]:
                    if cov[si] == 1:
                        redundant = False
                        break
                if redundant:
                    for si in block_subs[bi]:
                        cov[si] -= 1
                    blocks.pop(pos)
                    changed = True
                    break  # restart with fresh indices
        return blocks

    delete_pass(chosen, cov, deadline - 2.0)
    if len(chosen) < best_len:
        best_len = len(chosen)
        best = list(chosen)

    # --- Phase 3: swap local search ---
    # For a block, the subsets it uniquely covers must all be covered by the new block.
    # Candidate new blocks = intersection of sub_blocks[si] over those unique subsets.
    def swap_pass(blocks, cov, chosen_set, deadline, max_swaps):
        swaps = 0
        while swaps < max_swaps and time.time() < deadline:
            order = list(range(len(blocks)))
            rng.shuffle(order)
            progress = False
            for pos in order:
                if pos >= len(blocks):
                    continue
                old = blocks[pos]
                # unique subsets of old
                unique = [si for si in block_subs[old] if cov[si] == 1]
                if not unique:
                    # not critical; swap with a random distinct block for diversity
                    cand = rng.randrange(n_blk)
                    if cand == old or cand in chosen_set:
                        continue
                    # remove old, add cand
                    for si in block_subs[old]:
                        cov[si] -= 1
                    for si in block_subs[cand]:
                        cov[si] += 1
                    chosen_set.discard(old)
                    chosen_set.add(cand)
                    blocks[pos] = cand
                    swaps += 1
                    progress = True
                else:
                    # intersection of candidate blocks
                    cands = None
                    for si in unique:
                        if cands is None:
                            cands = set(sub_blocks[si])
                        else:
                            cands.intersection_update(sub_blocks[si])
                            if not cands:
                                break
                    if not cands:
                        continue
                    cands.discard(old)
                    # also drop blocks already present (to keep distinctness)
                    cands -= chosen_set
                    if not cands:
                        continue
                    new = rng.choice(sorted(cands))
                    for si in block_subs[old]:
                        cov[si] -= 1
                    for si in block_subs[new]:
                        cov[si] += 1
                    chosen_set.discard(old)
                    chosen_set.add(new)
                    blocks[pos] = new
                    swaps += 1
                    progress = True
            if not progress:
                break
        return blocks

    swap_pass(chosen, cov, chosen_set, deadline - 1.5, 400)
    delete_pass(chosen, cov, deadline - 0.5)
    if len(chosen) < best_len:
        best_len = len(chosen)
        best = list(chosen)

    # return best set found as list of sorted point lists
    return [sorted(all_blocks[i]) for i in best]
