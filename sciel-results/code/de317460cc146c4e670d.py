import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    DEADLINE = start_time + 118.0

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    all_t = list(itertools.combinations(points, t))
    N_t = len(all_t)
    t_idx = {s: i for i, s in enumerate(all_t)}

    # Precompute for each block the list of t-subset indices it covers
    block_tsubs = []
    for b in all_blocks:
        subs = [t_idx[s] for s in itertools.combinations(b, t)]
        block_tsubs.append(subs)

    # Map t-subset -> list of block indices
    tsub_blocks = [[] for _ in range(N_t)]
    for i, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_blocks[si].append(i)

    # For fast membership testing during local search
    block_tsub_sets = [set(subs) for subs in block_tsubs]

    def run_greedy(seed):
        random.seed(seed)
        block_gain = np.array([len(subs) for subs in block_tsubs], dtype=np.int32)
        is_covered = np.zeros(N_t, dtype=np.int8)
        uncovered_count = N_t
        selected_indices = []

        while uncovered_count > 0 and time.time() < DEADLINE:
            max_gain = np.max(block_gain)
            if max_gain <= 0:
                break
            candidates = np.where(block_gain == max_gain)[0]
            best_block_idx = random.choice(candidates.tolist())
            selected_indices.append(best_block_idx)

            subs = block_tsubs[best_block_idx]
            for si in subs:
                if is_covered[si] == 0:
                    is_covered[si] = 1
                    uncovered_count -= 1
                    block_gain[tsub_blocks[si]] -= 1

        return selected_indices

    def reduce_solution(selected_indices):
        """Remove redundant blocks via iterative peeling."""
        if not selected_indices:
            return []
        n_sel = len(selected_indices)
        g2l = {g: i for i, g in enumerate(selected_indices)}
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
        removed = [False] * n_sel

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
                        if g_other in g2l:
                            i_other = g2l[g_other]
                            if not removed[i_other]:
                                crit_count[i_other] += 1
                                if crit_count[i_other] == 1:
                                    pass  # just became critical, no queue action needed
        return [selected_indices[i] for i in range(n_sel) if not removed[i]]

    def local_search_polish(selected_indices, max_iters=200):
        """
        Aggressive local search:
        1) Peeling reduction
        2) Iterative single-block replacement: for each block, find a replacement
           that covers all its critical t-subsets and yields net gain (covers more
           unique t-subsets). Accept if valid.
        3) Repeat until no improvement or time budget exhausted.
        """
        if not selected_indices:
            return []
        current = list(selected_indices)
        current_set = set(current)

        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Phase 1: peel
        current = reduce_solution(current)
        current_set = set(current)
        total_count = np.zeros(N_t, dtype=np.int32)
        for g in current:
            for si in block_tsubs[g]:
                total_count[si] += 1

        # Phase 2: iterative replacement
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 8.0:
            improved = False
            iters += 1
            random.shuffle(current)
            for i in range(len(current)):
                g_old = current[i]
                old_subs = block_tsubs[g_old]
                # Critical t-subsets: those covered only by g_old
                critical = [si for si in old_subs if total_count[si] == 1]

                if not critical:
                    # Can simply remove
                    current.pop(i)
                    current_set.discard(g_old)
                    for si in old_subs:
                        total_count[si] -= 1
                    improved = True
                    break

                # Find candidate replacements covering all critical t-subsets
                # Intersect tsub_blocks for each critical
                cand_set = None
                for si in critical:
                    if cand_set is None:
                        cand_set = set(tsub_blocks[si])
                    else:
                        cand_set &= set(tsub_blocks[si])
                cand_set.discard(g_old)
                cand_set -= current_set

                if not cand_set:
                    continue

                candidates = list(cand_set)
                random.shuffle(candidates)

                best_new = None
                best_delta = 0
                for g_new in candidates[:50]:
                    new_subs = block_tsubs[g_new]
                    # Net gain = # new t-subsets covered by g_new not covered by g_old
                    # that are currently uncovered, minus any that become uncovered
                    # Simpler: check validity after swap
                    # Count how many t-subsets g_new covers that g_old doesn't
                    # and how many g_old covers that g_new doesn't
                    # For validity: every t-subset must remain covered >= 1
                    # t-subsets in old_subs that total_count==1 must be in new_subs
                    valid = all(si in block_tsub_sets[g_new] for si in critical)
                    if not valid:
                        continue
                    # Check: for si in old_subs, if total_count[si]==1, must be in new_subs
                    # Already checked via critical. For si in old_subs with total_count>1, fine.
                    # For si in new_subs not in old_subs: total_count increases, fine.
                    # The only risk is si in old_subs with total_count[si]==1 not in new_subs -> already excluded.
                    # But also need: no si becomes uncovered. Since we only remove g_old's coverage,
                    # the only risk is critical ones. Already handled.
                    # Additional benefit: g_new might cover t-subsets not in old_subs.
                    # Compute "unique gain": # t-subsets in new_subs not in old_subs with total_count==0
                    # (i.e., currently uncovered by anything else)
                    # Actually total_count includes g_old, so for si in new_subs not in old_subs,
                    # total_count[si] is the count without g_old (since g_old doesn't cover it).
                    # If total_count[si] == 0 before adding g_new, it's uncovered -> but we need cover!
                    # Wait: total_count currently counts all blocks in current including g_old.
                    # For si not in old_subs: total_count[si] doesn't include g_old.
                    # After swap, total_count[si] increases by 1.
                    # For si in old_subs: total_count[si] decreases by 1 (removing g_old) then increases by 1 if in new_subs.
                    # Validity: min total_count after swap must be >= 1.
                    # Risk: si in old_subs \ new_subs with total_count[si] == 1 -> becomes 0. Excluded by critical check.
                    # So valid = all critical in new_subs.
                    # Score: prefer blocks that cover more "scarce" t-subsets.
                    gain = 0
                    for si in new_subs:
                        if si not in block_tsub_sets[g_old] and total_count[si] <= 1:
                            gain += 1
                    if gain > best_delta:
                        best_delta = gain
                        best_new = g_new
                    if gain >= 20:
                        break

                if best_new is not None:
                    current[i] = best_new
                    current_set.discard(g_old)
                    current_set.add(best_new)
                    for si in old_subs:
                        total_count[si] -= 1
                    for si in block_tsubs[best_new]:
                        total_count[si] += 1
                    improved = True
                    break
        return current

    def is_valid(selected_indices):
        covered = np.zeros(N_t, dtype=np.int8)
        for g in selected_indices:
            for si in block_tsubs[g]:
                covered[si] = 1
        return np.all(covered == 1)

    # --- Main search loop ---
    best_sol_idx = run_greedy(0)
    best_sol_idx = reduce_solution(best_sol_idx)
    best_len = len(best_sol_idx)
    best_sol = [all_blocks[i] for i in best_sol_idx]

    # Multiple restarts with local search polish
    while time.time() < DEADLINE - 20.0:
        seed = random.randint(0, 2**31)
        cand = run_greedy(seed)
        cand = reduce_solution(cand)
        cand = local_search_polish(cand, max_iters=100)

        if len(cand) < best_len:
            best_len = len(cand)
            best_sol = [all_blocks[i] for i in cand]

    # Final polish on best
    if best_sol and time.time() < DEADLINE - 5.0:
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        best_idx = [block_to_idx[b] for b in best_sol]
        polished = local_search_polish(best_idx, max_iters=300)
        if len(polished) < best_len:
            best_len = len(polished)
            best_sol = [all_blocks[i] for i in polished]

    return best_sol
