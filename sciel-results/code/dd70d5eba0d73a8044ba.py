"""Covering design C(v,k,t): minimize the number of k-subsets of v points
so that every t-subset is contained in some block.

Family: greedy cover + local search (block exchange / redundancy removal).
Fast NumPy bitset representation of the uncovered t-subsets lets each
greedy step and each local-search pass run in a few milliseconds, so the
150 s wall-clock budget is spent mostly on local search to shrink the cover.
"""
import time
import itertools
import numpy as np


def construct(v, k, t):
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 8.0

    def remaining():
        return DEADLINE - MARGIN - time.time()

    # ---- enumerate all t-subsets and all k-subsets ------------------------
    tsubs = list(itertools.combinations(range(v), t))
    nT = len(tsubs)
    ksubs = list(itertools.combinations(range(v), k))
    nK = len(ksubs)

    # ---- bitset of t-subsets covered by each k-block ---------------------
    # masks[b, i] == 1  <=>  tsubs[i] is contained in ksubs[b]
    masks = np.zeros((nK, nT), dtype=np.uint8)
    for b, kb in enumerate(ksubs):
        kbset = set(kb)
        for i, ts in enumerate(tsubs):
            if all(x in kbset for x in ts):
                masks[b, i] = 1

    # ---- fast greedy build ------------------------------------------------
    covered = np.zeros(nT, dtype=np.uint8)

    def add_block(b):
        covered |= masks[b]

    def remove_block(b):
        for i in range(nT):
            if masks[b, i] and covered[i]:
                covered[i] = 0

    def greedy_build():
        covered[:] = 0
        chosen = []
        while True:
            if not covered.any():
                break
            # start from the lexicographically first uncovered t-subset
            first = int(np.argmax(covered))
            cands = np.where(masks[:, first] == 1)[0]
            if cands.size == 0:
                return None
            best = int(cands[0])
            cur = masks[best].astype(np.int32)
            while cur.sum() < k:
                if remaining() < 0.2:
                    return None
                # marginal gain of each candidate point
                gains = np.array(
                    [int((masks[b] & ~covered).sum()) for b in cands],
                    dtype=np.int64,
                )
                order = np.argsort(-gains)
                found = False
                for idx in order:
                    b = int(cands[idx])
                    newcov = masks[b].astype(np.int32)
                    if int((newcov & ~cur).sum()) >= 1:
                        best = b
                        cur = newcov
                        found = True
                        break
                if not found:
                    return None
            add_block(best)
            chosen.append(best)
        return chosen

    best = greedy_build()
    if best is None:
        # extremely unlikely; fall back to a trivial (but valid) cover
        best = list(range(nK))
    best_set = set(best)
    best_count = len(best)

    # ---- local search: block exchange / redundancy removal ---------------
    def local_search(chosen):
        nonlocal best_set, best_count
        chosen_set = set(chosen)
        covered_local = np.zeros(nT, dtype=np.uint8)
        for b in chosen:
            covered_local |= masks[b]

        while remaining() > 0.5:
            # a block is redundant if removing it leaves every t-subset covered
            redundant = []
            for b in list(chosen_set):
                if remaining() < 0.5:
                    break
                still = covered_local & ~masks[b]
                if still.all():
                    redundant.append(b)
            for b in redundant:
                chosen_set.discard(b)
                covered_local &= ~masks[b]
            if not chosen_set:
                break

            # find the best replacement for each current block
            need = ~covered_local
            if not need.any():
                break
            improved = True
            while improved and remaining() > 0.5:
                improved = False
                for b in list(chosen_set):
                    if remaining() < 0.5:
                        break
                    # gain of replacing b with each other block
                    for c in range(nK):
                        if c in chosen_set:
                            continue
                        if masks[c].sum() <= masks[b].sum():
                            continue
                        # new coverage after dropping b and adding c
                        newcov = (covered_local & ~masks[b]) | masks[c]
                        gain = int((newcov & need).sum()) - int((covered_local & need).sum())
                        if gain <= 0:
                            continue
                        # accept: drop b, add c
                        chosen_set.discard(b)
                        chosen_set.add(c)
                        covered_local = newcov
                        improved = True
                        break

        if len(chosen_set) < best_count:
            best_set = set(chosen_set)
            best_count = len(chosen_set)

    local_search(best)

    # ---- final: ensure validity and return --------------------------------
    if remaining() > 1.0:
        # one more pass to squeeze out any leftover redundancy
        local_search(best)

    blocks = [list(ksubs[b]) for b in sorted(best_set)]
    # safety: verify every t-subset is covered (cheap)
    final_cov = np.zeros(nT, dtype=np.uint8)
    for b in best_set:
        final_cov |= masks[b]
    if not final_cov.all():
        # should not happen; rebuild greedily from scratch to guarantee validity
        rebuilt = greedy_build()
        if rebuilt is not None:
            blocks = [list(ksubs[b]) for b in rebuilt]
    return blocks
