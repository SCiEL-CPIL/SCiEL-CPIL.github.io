import itertools
import time
import random
import numpy as np
import heapq
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115.0  # Safety margin before 120s limit

    # 1. Precomputation
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # block_tsubs[i] = list of t-subset indices covered by block i
    block_tsubs = [None] * n_blocks
    # tsub_blocks[j] = list of block indices covering t-subset j
    tsub_blocks = [[] for _ in range(n_tsubsets)]

    for bi, block in enumerate(all_blocks):
        tsubs = [tsub_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs[bi] = tsubs
        for ts_idx in tsubs:
            tsub_blocks[ts_idx].append(bi)

    # Precompute set versions for fast intersection
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    tsub_block_sets = [set(lst) for lst in tsub_blocks]

    # 2. LP relaxation for dual weights (bottleneck indicators)
    dual_weights = np.ones(n_tsubsets, dtype=np.float64)
    try:
        # Build sparse incidence: M[ts_idx, bi] = 1
        rows = []
        cols = []
        data = []
        for bi in range(n_blocks):
            for ts_idx in block_tsubs[bi]:
                rows.append(ts_idx)
                cols.append(bi)
                data.append(1.0)
        from scipy.sparse import csr_matrix
        from scipy.optimize import linprog
        M = csr_matrix((data, (rows, cols)), shape=(n_tsubsets, n_blocks))
        c = np.ones(n_blocks)
        A_ub = -M.T.tocsr()
        b_ub = -np.ones(n_tsubsets)
        bounds = [(0, 1)] * n_blocks
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            dw = np.abs(res.ineqlin.marginals)
            if np.sum(dw) > 0:
                dual_weights = dw
    except Exception:
        pass

    def greedy_cover(seed):
        """Heap-based greedy using LP dual weights as initial block scores."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsubsets
        # score[bi] = sum of dual_weights of uncovered t-subsets in block bi
        score = np.zeros(n_blocks, dtype=np.float64)
        for bi in range(n_blocks):
            score[bi] = np.sum(dual_weights[block_tsubs[bi]])

        # Max-heap: (-score, bi). We'll lazily update by pushing new entries.
        heap = [(-score[bi], bi) for bi in range(n_blocks)]
        heapq.heapify(heap)
        # current_score tracks the authoritative score
        cur_score = score.copy()

        selected = []
        while True:
            if time.time() > deadline - 5:
                break

            # Pop until valid
            best_bi = -1
            best_s = 0.0
            while heap:
                neg_s, bi = heapq.heappop(heap)
                s = -neg_s
                if abs(s - cur_score[bi]) < 1e-12 and s > 1e-9:
                    best_bi = bi
                    best_s = s
                    break
            if best_bi < 0 or best_s <= 1e-9:
                break

            selected.append(best_bi)

            # Update scores for newly covered t-subsets
            for ts_idx in block_tsubs[best_bi]:
                if not uncovered[ts_idx]:
                    continue
                uncovered[ts_idx] = False
                w_sub = dual_weights[ts_idx]
                if w_sub == 0:
                    continue
                for bj in tsub_blocks[ts_idx]:
                    if bj == best_bi:
                        continue
                    cur_score[bj] -= w_sub
                    if cur_score[bj] < 0:
                        cur_score[bj] = 0.0
                    heapq.heappush(heap, (-cur_score[bj], bj))
            cur_score[best_bi] = -1.0

        return selected

    def refine(block_indices, seed):
        """Local search: redundancy removal, single-swap, and 2-for-1 pair replacement."""
        rng = random.Random(seed)
        block_set = set(block_indices)
        cover_count = np.zeros(n_tsubsets, dtype=np.int32)
        for bi in block_set:
            cover_count[block_tsubs[bi]] += 1

        def find_replacement(unique_tsubs, exclude):
            """Find a block (not in exclude) covering all unique_tsubs via intersection."""
            if not unique_tsubs:
                return None
            # Start from the rarest t-subset
            order = sorted(unique_tsubs, key=lambda x: len(tsub_blocks[x]))
            candidates = set(tsub_block_sets[order[0]])
            for ts_idx in order[1:]:
                if not candidates:
                    break
                candidates &= tsub_block_sets[ts_idx]
                if not candidates:
                    break
            candidates -= exclude
            if not candidates:
                return None
            cand_list = list(candidates)
            rng.shuffle(cand_list)
            return cand_list

        def try_remove(bi):
            """Remove block bi if it has no uniquely-covered t-subset."""
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    return False
            block_set.discard(bi)
            cover_count[block_tsubs[bi]] -= 1
            return True

        def do_remove(bi):
            block_set.discard(bi)
            cover_count[block_tsubs[bi]] -= 1

        def do_add(bi):
            block_set.add(bi)
            cover_count[block_tsubs[bi]] += 1

        best_set = set(block_set)
        best_count = len(block_set)

        max_rounds = 60
        for rnd in range(max_rounds):
            if time.time() > deadline - 2:
                break

            improved = False

            # Phase 1: systematic redundancy removal
            to_remove = []
            for bi in list(block_set):
                redundant = True
                for ts_idx in block_tsubs[bi]:
                    if cover_count[ts_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(bi)
            if to_remove:
                for bi in to_remove:
                    do_remove(bi)
                improved = True
                if len(block_set) < best_count:
                    best_count = len(block_set)
                    best_set = set(block_set)
                continue

            # Phase 2: single-swap (replace one block with another, net 0 or -1)
            cur_list = list(block_set)
            rng.shuffle(cur_list)
            swapped = False
            for bi in cur_list:
                if time.time() > deadline - 2:
                    break
                if bi not in block_set:
                    continue
                unique_tsubs = [ts_idx for ts_idx in block_tsubs[bi]
                                if cover_count[ts_idx] == 1]
                if not unique_tsubs:
                    do_remove(bi)
                    improved = True
                    swapped = True
                    break
                cands = find_replacement(unique_tsubs, exclude={bi})
                if not cands:
                    continue
                # Try candidates; a swap is always non-worsening (net 0)
                # but we prefer ones that free up other blocks. Just take first.
                bj = cands[0]
                do_remove(bi)
                do_add(bj)
                improved = True
                swapped = True
                break
            if swapped:
                if len(block_set) < best_count:
                    best_count = len(block_set)
                    best_set = set(block_set)
                continue

            # Phase 3: 2-for-1 pair replacement (net -1)
            if len(block_set) >= 2:
                cur_list = list(block_set)
                rng.shuffle(cur_list)
                pair_done = False
                for i, b1 in enumerate(cur_list):
                    if time.time() > deadline - 2:
                        break
                    if b1 not in block_set:
                        continue
                    # pick a partner
                    others = [b for b in cur_list[i+1:] if b in block_set and b != b1]
                    if not others:
                        continue
                    b2 = others[rng.randint(0, len(others)-1)]
                    if b2 not in block_set:
                        continue

                    # Unique coverage of b1 and b2 (count==1 and in that block)
                    set1 = block_tsub_sets[b1]
                    set2 = block_tsub_sets[b2]
                    unique = set()
                    for ts_idx in set1:
                        if cover_count[ts_idx] == 1:
                            unique.add(ts_idx)
                    for ts_idx in set2:
                        if cover_count[ts_idx] == 1:
                            unique.add(ts_idx)
                    unique_list = list(unique)

                    if not unique_list:
                        # Both redundant (should have been caught in phase 1)
                        do_remove(b1)
                        do_remove(b2)
                        improved = True
                        pair_done = True
                        break

                    cands = find_replacement(unique_list, exclude={b1, b2})
                    if not cands:
                        continue
                    bk = cands[0]
                    do_remove(b1)
                    do_remove(b2)
                    do_add(bk)
                    improved = True
                    pair_done = True
                    break

                if pair_done:
                    if len(block_set) < best_count:
                        best_count = len(block_set)
                        best_set = set(block_set)
                    continue

            if not improved:
                # No improvement this round; try a random kick to escape local opt
                if block_set and rng.random() < 0.3:
                    # Add a random block (temporary worsening) then continue
                    ci = rng.randint(0, n_blocks - 1)
                    if ci not in block_set:
                        do_add(ci)
                else:
                    break

        # Final redundancy cleanup
        to_remove = []
        for bi in list(block_set):
            redundant = True
            for ts_idx in block_tsubs[bi]:
                if cover_count[ts_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(bi)
        for bi in to_remove:
            do_remove(bi)
        if len(block_set) < best_count:
            best_count = len(block_set)
            best_set = set(block_set)

        return list(best_set)

    best_blocks = None
    best_len = float('inf')

    # Multiple restarts
    num_restarts = 12
    for seed in range(num_restarts):
        if time.time() > deadline - 12:
            break
        init_blocks = greedy_cover(seed)
        if not init_blocks:
            continue
        refined = refine(init_blocks, seed)
        if len(refined) < best_len:
            best_len = len(refined)
            best_blocks = [all_blocks[bi] for bi in refined]

    if best_blocks is None:
        # Fallback: deterministic greedy
        uncovered = set(range(n_tsubsets))
        blocks_used = []
        gain = [len(block_tsubs[i]) for i in range(n_blocks)]
        while uncovered:
            max_gain = max(gain)
            if max_gain == 0:
                break
            chosen = gain.index(max_gain)
            blocks_used.append(chosen)
            for ts_idx in block_tsubs[chosen]:
                if ts_idx in uncovered:
                    uncovered.discard(ts_idx)
                    for bi in tsub_blocks[ts_idx]:
                        if gain[bi] > 0:
                            gain[bi] -= 1
            gain[chosen] = 0
        best_blocks = [all_blocks[bi] for bi in blocks_used]

    return best_blocks
