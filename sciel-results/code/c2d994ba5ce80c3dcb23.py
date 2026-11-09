import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    rng = _rng(_SEED)

    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_all = len(all_tsubs)
    tsub_to_idx = {tr: i for i, tr in enumerate(all_tsubs)}

    # Precompute for each (t-1)-subset of points, the list of (tsub_idx, missing_point)
    # so that when we extend a (t-1)-subset by a point c, we can find the t-subset in O(1).
    from itertools import combinations as _comb
    t1_to_tsubs = defaultdict(list)
    for tr_idx, tr in enumerate(all_tsubs):
        for omit in range(t):
            sub = tuple(x for i, x in enumerate(tr) if i != omit)
            missing = tr[omit]
            t1_to_tsubs[sub].append((tr_idx, missing))

    # Count of how many times each t-subset is covered
    count = np.zeros(n_all, dtype=np.int32)
    blocks = []
    block_set = set()

    def add_block(block):
        if block in block_set:
            return False
        block_set.add(block)
        blocks.append(block)
        for tr in itertools.combinations(block, t):
            count[tsub_to_idx[tr]] += 1
        return True

    def remove_block(block):
        for tr in itertools.combinations(block, t):
            count[tsub_to_idx[tr]] -= 1
        block_set.discard(block)

    def block_triple_indices(block):
        return [tsub_to_idx[tr] for tr in itertools.combinations(block, t)]

    def compute_gain(block, count_arr):
        return int(np.sum(count_arr[block_triple_indices(block)] == 0))

    def compute_score(block, count_arr):
        """Gain minus a small redundancy penalty on already-covered t-subsets."""
        idxs = block_triple_indices(block)
        c = count_arr[idxs]
        gain = int(np.sum(c == 0))
        redundancy = float(np.sum(c[c > 0]))
        return gain - 0.05 * redundancy

    def build_block_greedy(start_tsub, count_arr):
        """Pure greedy: pick the candidate that maximizes new t-subset coverage."""
        block = list(start_tsub)
        remaining = set(range(v))
        for p in start_tsub:
            remaining.discard(p)

        for _ in range(k - t):
            best_cand = None
            best_s = -1
            # (t-1)-subsets of the current block
            t1_subs = list(itertools.combinations(block, t - 1))
            for cand in remaining:
                s = 0
                for sub in t1_subs:
                    for tr_idx, missing in t1_to_tsubs[sub]:
                        if missing == cand:
                            if count_arr[tr_idx] == 0:
                                s += 1
                            break
                if s > best_s:
                    best_s = s
                    best_cand = cand
            if best_cand is None:
                best_cand = min(remaining)
            block.append(best_cand)
            remaining.discard(best_cand)
        return tuple(sorted(block))

    def build_block_probabilistic(start_tsub, count_arr, alpha):
        """Softmax-temperature block builder. alpha=0 -> greedy."""
        block = list(start_tsub)
        remaining = set(range(v))
        for p in start_tsub:
            remaining.discard(p)

        for _ in range(k - t):
            t1_subs = list(itertools.combinations(block, t - 1))
            cand_scores = {}
            for cand in remaining:
                s = 0
                for sub in t1_subs:
                    for tr_idx, missing in t1_to_tsubs[sub]:
                        if missing == cand:
                            if count_arr[tr_idx] == 0:
                                s += 1
                            break
                cand_scores[cand] = s
            if not cand_scores:
                break
            cands = list(cand_scores.keys())
            scores = np.array([cand_scores[c] for c in cands], dtype=np.float64)
            if alpha > 1e-9 and len(scores) > 1:
                max_s = scores.max()
                exps = np.exp((scores - max_s) / alpha)
                tot = exps.sum()
                if tot > 0:
                    probs = exps / tot
                    best_cand = cands[int(rng.choice(len(cands), p=probs))]
                else:
                    best_cand = cands[int(np.argmax(scores))]
            else:
                best_cand = cands[int(np.argmax(scores))]
            block.append(best_cand)
            remaining.discard(best_cand)
        return tuple(sorted(block))

    def build_block(start_tsub, count_arr, alpha):
        if alpha <= 1e-9:
            return build_block_greedy(start_tsub, count_arr)
        return build_block_probabilistic(start_tsub, count_arr, alpha)

    def pick_best_start(uncovered, count_arr, alphas, sample_size):
        """Sample uncovered t-subsets, build candidate blocks, return best by score."""
        if not uncovered:
            return None, -1e9
        n = len(uncovered)
        if n <= sample_size:
            sample_idx = list(range(n))
        else:
            sample_idx = list(rng.choice(n, size=sample_size, replace=False))
        best_block = None
        best_score = -1e9
        for idx in sample_idx:
            tr = uncovered[idx]
            for alpha in alphas:
                b = build_block(tr, count_arr, alpha)
                sc = compute_score(b, count_arr)
                if sc > best_score:
                    best_score = sc
                    best_block = b
        return best_block, best_score

    # ---- Phase 1: Greedy / probabilistic construction ----
    alphas_build = [0.0, 0.15, 0.3, 0.5]
    while time.time() < deadline - 30:
        unc_idx = np.where(count == 0)[0]
        if len(unc_idx) == 0:
            break
        uncovered = [all_tsubs[i] for i in unc_idx]
        b, sc = pick_best_start(uncovered, count, alphas_build, sample_size=20)
        if b is None:
            break
        if not add_block(b):
            # duplicate; pick a different one
            continue

    # ---- Phase 2: Local replacement / deletion ----
    alphas_repl = [0.0, 0.1, 0.2, 0.3, 0.5]
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for bi in range(len(blocks)):
            if time.time() > deadline - 15:
                break
            removed = blocks[bi]
            remove_block(removed)
            unc_idx = np.where(count == 0)[0]
            if len(unc_idx) == 0:
                # block was redundant: drop it
                blocks.pop(bi)
                improved = True
                break
            uncovered = [all_tsubs[i] for i in unc_idx]
            # We need a replacement that covers all currently-uncovered t-subsets.
            # Build candidates and keep the one that covers the most uncovered,
            # breaking ties by score.
            n_unc = len(uncovered)
            sample_size = min(15, n_unc)
            if n_unc <= sample_size:
                sample_idx = list(range(n_unc))
            else:
                sample_idx = list(rng.choice(n_unc, size=sample_size, replace=False))
            best_repl = None
            best_cover = -1
            best_sc = -1e9
            for idx in sample_idx:
                tr = uncovered[idx]
                for alpha in alphas_repl:
                    b = build_block(tr, count, alpha)
                    # how many of the uncovered t-subsets does b cover?
                    b_tris = set(itertools.combinations(b, t))
                    cover = 0
                    for i in unc_idx:
                        if all_tsubs[i] in b_tris:
                            cover += 1
                    if cover > best_cover or (cover == best_cover and compute_score(b, count) > best_sc):
                        best_cover = cover
                        best_sc = compute_score(b, count)
                        best_repl = b
            if best_repl is not None and best_cover == n_unc:
                blocks[bi] = best_repl
                add_block(best_repl)
                improved = True
                break
            else:
                # restore original
                add_block(removed)

    # ---- Phase 3: Completion (ensure full coverage) ----
    while time.time() < deadline - 2:
        unc_idx = np.where(count == 0)[0]
        if len(unc_idx) == 0:
            break
        uncovered = [all_tsubs[i] for i in unc_idx]
        b, _ = pick_best_start(uncovered, count, [0.0, 0.3], sample_size=10)
        if b is None:
            break
        add_block(b)

    # ---- Phase 4: Final polish - try to drop any now-redundant blocks ----
    if time.time() < deadline - 1:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for bi in range(len(blocks)):
                if time.time() > deadline - 1:
                    break
                removed = blocks[bi]
                remove_block(removed)
                if np.all(count >= 1):
                    blocks.pop(bi)
                    changed = True
                    break
                else:
                    add_block(removed)

    # Final deduplication (defensive)
    seen = set()
    unique_blocks = []
    for b in blocks:
        if b not in seen:
            seen.add(b)
            unique_blocks.append(b)

    return [list(b) for b in unique_blocks]
