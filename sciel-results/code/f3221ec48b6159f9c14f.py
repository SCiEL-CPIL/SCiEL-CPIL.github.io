import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 115.0  # Safety margin for 150s limit

    points = list(range(v))

    # Precomputation
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)

    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Build block -> t-subset index lists
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # For each t-subset, list of blocks containing it
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # Precompute bitmasks for each block's t-subsets
    # N_t can be up to C(19,5) = 11628, so we need multiple 64-bit words
    n_words = (N_t + 63) // 64
    block_mask = []
    for subs in block_tsubs:
        mask = [0] * n_words
        for si in subs:
            word_idx = si // 64
            bit = si % 64
            mask[word_idx] |= (1 << bit)
        block_mask.append(mask)

    # Precompute reverse: for each t-subset, which blocks cover it (as bitmask of block indices)
    # N_blocks can be up to C(19,8) = 75582, so multiple words
    n_block_words = (N_blocks + 63) // 64
    tsub_block_mask = []
    for si in range(N_t):
        mask = [0] * n_block_words
        for g in tsub_blocks[si]:
            word_idx = g // 64
            bit = g % 64
            mask[word_idx] |= (1 << bit)
        tsub_block_mask.append(mask)

    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        # Coverage as bitset: list of 64-bit integers
        covered_mask = [0] * n_words
        uncovered_count = N_t
        selected_indices = []

        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = int(np.max(block_gain))
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = int(random.choice(candidates.tolist()))

            selected_indices.append(best_block_idx)

            subs = block_tsubs[best_block_idx]
            bm = block_mask[best_block_idx]
            for si in subs:
                word_idx = si // 64
                bit = si % 64
                if not (covered_mask[word_idx] & (1 << bit)):
                    covered_mask[word_idx] |= (1 << bit)
                    uncovered_count -= 1
                    # Update gains for all blocks covering this t-subset
                    tbm = tsub_block_mask[si]
                    for w in range(n_block_words):
                        bits = tbm[w]
                        while bits:
                            lb = bits & (-bits)
                            b_idx = (w * 64) + int(np.log2(lb))
                            block_gain[b_idx] -= 1
                            bits ^= lb

        return selected_indices

    def reduce_solution_fast(selected_indices):
        if not selected_indices:
            return []

        n_sel = len(selected_indices)
        local_of_global = np.full(N_blocks, -1, dtype=np.int32)
        for i, g in enumerate(selected_indices):
            local_of_global[g] = i

        total_count = np.zeros(N_t, dtype=np.int32)
        crit_count = np.zeros(n_sel, dtype=np.int32)

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                total_count[si] += 1

        for i in range(n_sel):
            g = selected_indices[i]
            for si in block_tsubs[g]:
                if total_count[si] == 1:
                    crit_count[i] += 1

        queue = [i for i in range(n_sel) if crit_count[i] == 0]
        removed = np.zeros(n_sel, dtype=np.bool_)

        while queue:
            i = queue.pop()
            if removed[i] or crit_count[i] > 0:
                continue

            removed[i] = True
            g = selected_indices[i]

            for si in block_tsubs[g]:
                old_c = total_count[si]
                total_count[si] -= 1
                new_c = total_count[si]

                if old_c == 2 and new_c == 1:
                    for g_other in tsub_blocks[si]:
                        if g_other == g:
                            continue
                        i_other = local_of_global[g_other]
                        if i_other >= 0 and not removed[i_other]:
                            crit_count[i_other] += 1

        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search(selected_indices, max_iters=50):
        """1-for-1 swap local search: try replacing a critical block with a better one."""
        if not selected_indices:
            return selected_indices

        n_sel = len(selected_indices)
        in_sol = np.zeros(N_blocks, dtype=np.bool_)
        for g in selected_indices:
            in_sol[g] = True
        total_count = np.zeros(N_t, dtype=np.int32)

        for g in selected_indices:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Precompute for each block, set of t-subsets it covers (as frozenset for fast lookup)
        # Actually, use the existing block_tsubs lists

        improved = True
        while improved and time.time() < DEADLINE - 2.0:
            improved = False
            n_sel = len(selected_indices)
            for i in range(n_sel):
                if time.time() > DEADLINE - 2.0:
                    return selected_indices
                g = selected_indices[i]
                # Find t-subsets uniquely covered by this block
                unique_si = []
                for si in block_tsubs[g]:
                    if total_count[si] == 1:
                        unique_si.append(si)
                if not unique_si:
                    continue

                # Tentatively remove g
                for si in block_tsubs[g]:
                    total_count[si] -= 1

                # Find blocks that cover all unique_si
                best_gain = 0
                best_g = -1
                for g_new in range(N_blocks):
                    if in_sol[g_new]:
                        continue
                    # Check if g_new covers all unique_si using bitmask
                    bm = block_mask[g_new]
                    ok = True
                    for si in unique_si:
                        word_idx = si // 64
                        bit = si % 64
                        if not (bm[word_idx] & (1 << bit)):
                            ok = False
                            break
                    if not ok:
                        continue
                    # Count how many previously-uncovered t-subsets it now covers
                    gain = 0
                    for si in block_tsubs[g_new]:
                        if total_count[si] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_g = g_new

                if best_g >= 0:
                    # Apply swap: remove g, add best_g
                    in_sol[g] = False
                    in_sol[best_g] = True
                    for si in block_tsubs[best_g]:
                        total_count[si] += 1
                    selected_indices[i] = best_g
                    improved = True
                else:
                    # Revert
                    for si in block_tsubs[g]:
                        total_count[si] += 1

        return selected_indices

    # First run: deterministic greedy
    sol_idx = run_greedy(0)
    sol_idx = reduce_solution_fast(sol_idx)
    best_sol_idx = list(sol_idx)
    best_len = len(best_sol_idx)

    # Randomized restarts
    while time.time() < DEADLINE - 12.0:
        seed = random.randint(0, 2**31)
        cand_idx = run_greedy(seed)
        cand_idx = reduce_solution_fast(cand_idx)

        if len(cand_idx) < best_len:
            best_len = len(cand_idx)
            best_sol_idx = list(cand_idx)
            # Try local search on the new best
            if time.time() < DEADLINE - 15.0:
                improved = local_search(list(cand_idx))
                improved = reduce_solution_fast(improved)
                if len(improved) < best_len:
                    best_len = len(improved)
                    best_sol_idx = list(improved)

    # Final local search pass on the best solution
    if time.time() < DEADLINE - 5.0:
        improved = local_search(list(best_sol_idx))
        improved = reduce_solution_fast(improved)
        if len(improved) < best_len:
            best_len = len(improved)
            best_sol_idx = list(improved)

    return [all_blocks[i] for i in best_sol_idx]
