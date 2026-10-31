"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Branch-and-Bound with tight lower bound + greedy/local-search upper bound.
- Precomputes coverage masks for all k-subsets.
- Uses greedy + local search (redundancy removal + 1-for-1 swaps) for initial UB.
- B&B uses a combined lower bound (fractional + greedy disjoint packing) and
  dynamic variable ordering (min remaining coverers among uncovered t-subsets).
"""
import time
import itertools
import math

_START = None
_DEADLINE = 115.0  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _START

def construct(v, k, t):
    global _START
    _START = time.time()

    # 1. Generate all blocks (k-subsets) and all t-subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_t_subs = list(itertools.combinations(points, t))

    n_blocks = len(all_blocks)
    n_t = len(all_t_subs)

    # Map t-subsets to indices
    t_index = {ts: i for i, ts in enumerate(all_t_subs)}
    full_mask = (1 << n_t) - 1

    # 2. Precompute coverage masks for each block
    masks = [0] * n_blocks
    coverers = [[] for _ in range(n_t)]

    for i, blk in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(blk, t):
            j = t_index[ts]
            mask |= (1 << j)
            coverers[j].append(i)
        masks[i] = mask

    freq = [len(c) for c in coverers]

    # 3. Greedy construction with deficiency-ratio heuristic
    #    Score each block by (new coverage) / (1 + point-saturation penalty)
    #    to balance spread vs. concentration.
    def greedy_solve(use_deficiency=False):
        uncovered = full_mask
        selected = []
        # Track how many times each point is used (for deficiency penalty)
        point_count = [0] * v
        while uncovered:
            if _elapsed() > _DEADLINE - 3:
                return None
            best_block = -1
            best_score = -1.0
            for i in range(n_blocks):
                m = masks[i]
                inter = m & uncovered
                if inter:
                    gain = inter.bit_count()
                    if use_deficiency:
                        # Penalty: sum of 1/(1+count[p]) for points in block
                        # Lower penalty = less saturated points preferred
                        pen = 0.0
                        for p in all_blocks[i]:
                            pen += 1.0 / (1.0 + point_count[p])
                        score = gain / (1.0 + pen * 0.05)
                    else:
                        score = gain
                    if score > best_score:
                        best_score = score
                        best_block = i
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered &= ~masks[best_block]
            for p in all_blocks[best_block]:
                point_count[p] += 1
        return selected

    # 4. Local search: redundancy removal + 1-for-1 swaps
    def local_search(sel):
        if sel is None:
            return None
        sel = list(sel)
        improved = True
        while improved:
            if _elapsed() > _DEADLINE - 3:
                return sel
            improved = False
            # Redundancy removal: try removing each block
            for idx in range(len(sel)):
                if _elapsed() > _DEADLINE - 3:
                    return sel
                # Compute coverage without block idx
                cov = 0
                for j in range(len(sel)):
                    if j != idx:
                        cov |= masks[sel[j]]
                if cov == full_mask:
                    sel.pop(idx)
                    improved = True
                    break
            # 1-for-1 swap: replace block at idx with a better one
            if not improved:
                for idx in range(len(sel)):
                    if _elapsed() > _DEADLINE - 3:
                        return sel
                    # Coverage of all other blocks
                    other_cov = 0
                    for j in range(len(sel)):
                        if j != idx:
                            other_cov |= masks[sel[j]]
                    # Uncovered by others = what block idx must cover
                    need = full_mask & ~other_cov
                    if need == 0:
                        sel.pop(idx)
                        improved = True
                        break
                    # Find a block that covers 'need' and adds more than sel[idx] does
                    cur_extra = (masks[sel[idx]] & ~other_cov).bit_count()
                    best_swap = -1
                    best_gain = cur_extra
                    for i in range(n_blocks):
                        if i == sel[idx]:
                            continue
                        if masks[i] & need:
                            gain = (masks[i] & ~other_cov).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_swap = i
                    if best_swap != -1:
                        sel[idx] = best_swap
                        improved = True
                        break
        return sel

    # Get initial upper bound
    incumbent = greedy_solve(use_deficiency=False)
    if incumbent is None:
        return []
    incumbent = local_search(incumbent)
    if incumbent is None:
        return []

    # Try deficiency-ratio greedy as alternative
    alt = greedy_solve(use_deficiency=True)
    if alt is not None:
        alt = local_search(alt)
        if alt is not None and len(alt) < len(incumbent):
            incumbent = alt

    best_len = len(incumbent)
    best_solution = incumbent[:]

    # 5. Lower bound: max of fractional and greedy disjoint packing
    def calc_lb(uncovered):
        if uncovered == 0:
            return 0
        # Fractional LB
        s = 0.0
        u = uncovered
        while u:
            lsb = u & -u
            j = lsb.bit_length() - 1
            s += 1.0 / freq[j]
            u ^= lsb
        frac_lb = math.ceil(s)

        # Greedy disjoint packing LB: repeatedly pick an uncovered t-subset
        # with minimum freq, count it, remove all t-subsets that share a point
        # (since one block can cover at most one of them... wait, that's not right.
        #  Actually: a single block of size k can cover multiple t-subsets.
        #  Better packing: pick t-subsets that are pairwise disjoint in their
        #  point sets? No - a block of size 7 can cover 4-subsets that share points.
        #  Correct packing: find a set of t-subsets such that no single k-block
        #  can cover more than one. This means the t-subsets must be "incompatible".
        #  Two 4-subsets are incompatible if their union has size > 7 (i.e., they
        #  share fewer than 4+4-7=1 points... wait: |A∪B| > 7 means |A∩B| < 1,
        #  i.e., they are disjoint). So disjoint 4-subsets form a packing.
        #  Actually |A∪B| = |A|+|B|-|A∩B| = 8-|A∩B|. For |A∪B|>7: |A∩B|<1.
        #  So two 4-subsets with empty intersection cannot be in the same 7-block.
        #  Greedy: pick uncovered 4-subsets that are pairwise disjoint.
        pack_lb = 0
        remaining = uncovered
        while remaining:
            if _elapsed() > _DEADLINE:
                return frac_lb
            # Pick the uncovered t-subset with minimum freq among remaining
            best_j = -1
            best_f = 10**9
            u = remaining
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                if freq[j] < best_f:
                    best_f = freq[j]
                    best_j = j
                u ^= lsb
            if best_j == -1:
                break
            pack_lb += 1
            # Remove all t-subsets in remaining that share any point with best_j
            # (since they could potentially be in the same block as best_j)
            # Actually for a valid packing we need DISJOINT t-subsets.
            # A block of size 7 can contain two 4-subsets iff |A∩B| >= 1 (since |A∪B|<=7).
            # So for packing, we need |A∩B| = 0.
            # Remove all remaining t-subsets that share at least one point with best_j.
            best_ts = all_t_subs[best_j]
            best_points = set(best_ts)
            new_remaining = 0
            u = remaining
            while u:
                lsb = u & -u
                j = lsb.bit_length() - 1
                u ^= lsb
                ts = all_t_subs[j]
                # Check if disjoint from best_ts
                disjoint = True
                for p in ts:
                    if p in best_points:
                        disjoint = False
                        break
                if disjoint:
                    new_remaining |= (1 << j)
            remaining = new_remaining

        return max(frac_lb, pack_lb)

    # 6. Branch and Bound
    stack = [(full_mask, 0, [])]

    while stack:
        if _elapsed() > _DEADLINE:
            break

        uncovered, depth, sol = stack.pop()

        lb = calc_lb(uncovered)
        if depth + lb >= best_len:
            continue

        if uncovered == 0:
            if depth < best_len:
                best_len = depth
                best_solution = sol[:]
            continue

        # Select variable: uncovered t-subset with minimum freq
        u = uncovered
        min_freq = 10**9
        target_j = -1
        while u:
            lsb = u & -u
            j = lsb.bit_length() - 1
            f = freq[j]
            if f < min_freq:
                min_freq = f
                target_j = j
            u ^= lsb

        if target_j == -1:
            continue

        # Branch on blocks covering target_j
        candidates = coverers[target_j]

        scored_candidates = []
        for i in candidates:
            m = masks[i]
            gain = (m & uncovered).bit_count()
            scored_candidates.append((gain, i))

        scored_candidates.sort(reverse=True)

        for gain, i in scored_candidates:
            new_uncovered = uncovered & ~masks[i]
            new_depth = depth + 1

            if new_depth >= best_len:
                continue

            if new_uncovered == 0:
                if new_depth < best_len:
                    best_len = new_depth
                    best_solution = sol + [i]
                continue

            nlb = calc_lb(new_uncovered)
            if new_depth + nlb >= best_len:
                continue

            stack.append((new_uncovered, new_depth, sol + [i]))

    # 7. Convert result
    if best_solution:
        result = [list(all_blocks[i]) for i in best_solution]
        return result
    else:
        return [list(all_blocks[i]) for i in incumbent]
