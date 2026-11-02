import itertools
import random
import time
import math
import numpy as np

_DEADLINE = 145.0
_start_time = None


def _elapsed():
    return time.time() - _start_time if _start_time is not None else 0.0


def construct(v, k, t):
    global _start_time
    _start_time = time.time()

    # Precompute all t-subsets (targets)
    all_ts = list(itertools.combinations(range(v), t))
    n_ts = len(all_ts)
    ts_index = {ts: i for i, ts in enumerate(all_ts)}

    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # For each block, the list of t-subset indices it covers
    block_ts = []
    for blk in all_blocks:
        block_ts.append([ts_index[ts] for ts in itertools.combinations(blk, t)])

    # For each t-subset, the list of blocks containing it
    ts_to_blocks = [[] for _ in range(n_ts)]
    for b_idx, ts_list in enumerate(block_ts):
        for s_idx in ts_list:
            ts_to_blocks[s_idx].append(b_idx)

    # ---- Smooth surrogate objective (soft-min of coverage) ----
    # For a selected set S, coverage[s] = number of blocks in S covering s.
    # We want min_s coverage[s] >= 1.  Use a smooth penalty:
    #   L = sum_s softmin(coverage[s])  with softmin via log-sum-exp of -coverage.
    # This is differentiable in the "which blocks" sense via per-block marginal
    # contributions, and drives the local search toward the hard-to-cover subsets.
    def softmin_pen(coverage, beta=8.0):
        # coverage: int array, >=0.  softmin over values: -logsumexp(-beta*c)/beta
        # For c >= 1, softmin ~ 1 (good).  For c = 0, softmin ~ 0 (bad, penalty).
        # We want to *maximize* coverage at the minimum, so minimize a penalty that
        # is large when any c < 1.  Use: penalty = sum_s max(0, 1 - softmin(c_s))
        # but to keep it smooth, use a logistic: pen_s = 1/(1+exp(beta*(c_s-1)))
        # c_s=1 -> 0.5, c_s=0 -> ~1, c_s=2 -> ~0.  Sum this.
        return np.sum(1.0 / (1.0 + np.exp(beta * (coverage - 1.0))))

    # ---- Greedy initialization ----
    def greedy_init(seed):
        local_rng = random.Random(seed)
        uncovered = np.ones(n_ts, dtype=bool)
        uncovered_count = n_ts
        gains = np.array([len(ts) for ts in block_ts], dtype=np.int32)
        selected = []
        while uncovered_count > 0:
            if time.time() - _start_time > _DEADLINE - 8.0:
                break
            best_b = int(np.argmax(gains))
            best_gain = gains[best_b]
            if best_gain <= 0:
                break
            selected.append(best_b)
            for s_idx in block_ts[best_b]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    uncovered_count -= 1
                    for other_b in ts_to_blocks[s_idx]:
                        if gains[other_b] > 0:
                            gains[other_b] -= 1
        return selected

    # ---- Polish: remove redundant blocks ----
    def polish(blocks):
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            for s in block_ts[b]:
                coverage[s] += 1
        changed = True
        while changed and time.time() - _start_time < _DEADLINE - 8.0:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                redundant = True
                for s in block_ts[b]:
                    if coverage[s] == 1:
                        redundant = False
                        break
                if redundant:
                    for s in block_ts[b]:
                        coverage[s] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # ---- Local search with pair-replacement, swaps, and smooth surrogate ----
    def local_search(blocks, seed):
        local_rng = random.Random(seed)
        blocks = list(blocks)

        def coverage_of(blk_set):
            cov = np.zeros(n_ts, dtype=np.int32)
            for b in blk_set:
                for s in block_ts[b]:
                    cov[s] += 1
            return cov

        cov = coverage_of(blocks)
        # Objective: minimize (len(blocks) + lambda * softmin_pen(cov))
        LAMBDA = 5.0

        def objective(blk_list):
            return len(blk_list) + LAMBDA * softmin_pen(cov)

        best_len = len(blocks)
        no_improve = 0
        max_no_improve = 200

        while time.time() - _start_time < _DEADLINE - 10.0 and no_improve < max_no_improve:
            improved = False

            # (a) Deletion: try removing each block
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                # check if any subset covered only by b
                only = False
                for s in block_ts[b]:
                    if cov[s] == 1:
                        only = True
                        break
                if not only:
                    for s in block_ts[b]:
                        cov[s] -= 1
                    blocks.pop(i)
                    improved = True
                    no_improve = 0
                    break
            if improved:
                continue

            # (b) Pair replacement: remove two blocks, add one that covers their union's
            #     uncovered-by-them subsets.  Net -1 block.
            found = False
            n = len(blocks)
            # Sample a limited number of pairs to keep runtime bounded
            n_pairs_to_try = min(n * (n - 1) // 2, 4000)
            pairs = []
            if n_pairs_to_try < n * (n - 1) // 2:
                # random sample
                for _ in range(n_pairs_to_try):
                    i = local_rng.randrange(n)
                    j = local_rng.randrange(n)
                    if i != j:
                        pairs.append((i, j))
            else:
                for i in range(n):
                    for j in range(i + 1, n):
                        pairs.append((i, j))

            for (i, j) in pairs:
                if i == j:
                    continue
                if i >= len(blocks) or j >= len(blocks):
                    continue
                if i > j:
                    i, j = j, i
                bi, bj = blocks[i], blocks[j]
                # Subsets covered by exactly one of bi, bj (i.e. would become uncovered
                # if both removed) OR by neither.  Actually we need subsets that are
                # covered ONLY by bi or bj (cov==1 and the covering block is bi or bj).
                # Simpler: compute which subsets lose coverage.
                # A subset s loses coverage if cov[s] - (1 if s in bi else 0) -
                # (1 if s in bj else 0) < 1, i.e. becomes 0.
                # We need a new block c that covers ALL such subsets.
                # Build the set of "critical" subsets.
                critical = []
                for s in block_ts[bi]:
                    if cov[s] == 1:
                        critical.append(s)
                for s in block_ts[bj]:
                    if cov[s] == 1 and s not in block_ts[bi]:
                        critical.append(s)
                if not critical:
                    # Removing both is safe, but we want net -1, so add nothing -> that's
                    # a deletion of 2, net -2.  Just do it.
                    for s in block_ts[bi]:
                        cov[s] -= 1
                    for s in block_ts[bj]:
                        cov[s] -= 1
                    blocks.pop(j)
                    blocks.pop(i)
                    improved = True
                    no_improve = 0
                    found = True
                    break
                # Need a block covering all critical subsets
                # Find candidate blocks: intersect ts_to_blocks of each critical subset
                # Use the smallest critical list as seed
                crit_set = set(critical)
                # Candidate blocks must cover every s in crit_set
                # Start with blocks covering the first critical subset
                first = critical[0]
                candidates = ts_to_blocks[first]
                # Filter: must cover all critical
                # To keep it fast, only consider candidates not already in blocks
                best_c = -1
                best_extra = -1  # extra coverage beyond critical (good for robustness)
                for c in candidates:
                    if c == bi or c == bj:
                        continue
                    # check covers all critical
                    ok = True
                    cts = set(block_ts[c])
                    for s in critical:
                        if s not in cts:
                            ok = False
                            break
                    if not ok:
                        continue
                    # count how many currently-uncovered-by-removal it covers (all critical)
                    # plus extra coverage of subsets currently at cov==1 (robustness)
                    extra = 0
                    for s in block_ts[c]:
                        if s not in crit_set and cov[s] == 1:
                            extra += 1
                    if extra > best_extra:
                        best_extra = extra
                        best_c = c
                if best_c >= 0:
                    # Apply: remove bi, bj; add best_c
                    for s in block_ts[bi]:
                        cov[s] -= 1
                    for s in block_ts[bj]:
                        cov[s] -= 1
                    for s in block_ts[best_c]:
                        cov[s] += 1
                    blocks.pop(j)
                    blocks.pop(i)
                    blocks.append(best_c)
                    improved = True
                    no_improve = 0
                    found = True
                    break
            if found:
                continue

            # (c) Swap: replace one block with another (net 0) to improve robustness
            #     (helps future pair-replacements).  Use smooth surrogate to guide.
            swap_done = False
            n = len(blocks)
            for _ in range(min(200, n * 10)):
                i = local_rng.randrange(n)
                bi = blocks[i]
                # candidate new block: covers some subset at cov==1 (would drop to 0)
                # pick a random subset at cov==1 covered by bi
                cov1_in_bi = [s for s in block_ts[bi] if cov[s] == 1]
                if not cov1_in_bi:
                    continue
                s = local_rng.choice(cov1_in_bi)
                # candidate blocks covering s (other than bi)
                cands = [c for c in ts_to_blocks[s] if c != bi]
                if not cands:
                    continue
                # choose the candidate that maximizes coverage of other cov==1 subsets
                best_c = -1
                best_score = -1
                for c in cands[:50]:
                    score = 0
                    for ss in block_ts[c]:
                        if cov[ss] == 1:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_c = c
                if best_c < 0:
                    continue
                # Apply swap
                for ss in block_ts[bi]:
                    cov[ss] -= 1
                for ss in block_ts[best_c]:
                    cov[ss] += 1
                blocks[i] = best_c
                swap_done = True
                break
            if swap_done:
                no_improve = 0
                continue

            no_improve += 1

        return blocks

    # ---- Main ----
    # 1. Greedy init + polish
    init = greedy_init(42)
    init = polish(init)
    best = list(init)
    best_len = len(best)

    if time.time() - _start_time > _DEADLINE - 15.0:
        return [list(all_blocks[b]) for b in best]

    # 2. Local search from the greedy solution (multiple seeds / passes)
    #    Run local search; it is in-place improving.
    best = local_search(best, 1)
    best = polish(best)
    best_len = len(best)

    # 3. Try a few alternative greedy seeds, keep the best
    alt_seeds = [7, 123, 2024, 999]
    for seed in alt_seeds:
        if time.time() - _start_time > _DEADLINE - 12.0:
            break
        cand = greedy_init(seed)
        cand = polish(cand)
        cand = local_search(cand, seed)
        cand = polish(cand)
        if len(cand) < best_len:
            best = cand
            best_len = len(cand)

    # 4. Final polish
    best = polish(best)

    result = [list(all_blocks[b]) for b in best]
    return result
