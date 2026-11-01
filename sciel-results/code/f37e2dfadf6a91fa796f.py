import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

    # ---------- 1. Map t-subsets to IDs ----------
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_map = {sub: i for i, sub in enumerate(t_subs)}
    num_t = len(t_subs)

    # ---------- 2. Pre-compute block -> t-subset IDs (vectorized) ----------
    k_subs = list(itertools.combinations(range(v), k))
    num_k = len(k_subs)

    t_in_k = list(itertools.combinations(range(k), t))
    num_t_in_k = len(t_in_k)

    # Build block_t_ids via a single argsort (avoids per-element dict lookups)
    # t_subs is already in lexicographic order, so np.argsort gives the ID.
    all_t_subs = np.array(t_subs, dtype=np.int32)  # (num_t, t)
    all_t_flat = all_t_subs.reshape(-1)
    sort_order = np.argsort(all_t_flat, kind='stable')
    inverse = np.empty(all_t_flat.shape, dtype=np.int32)
    inverse[sort_order] = np.arange(all_t_flat.shape, dtype=np.int32)
    pos_in_flat = inverse.reshape(all_t_subs.shape)  # (num_t, t)

    # For each block, its t-subsets in the same generation order as t_in_k
    k_arr = np.array(k_subs, dtype=np.int32)  # (num_k, k)
    tidx = np.array(t_in_k, dtype=np.int32)   # (num_t_in_k, t)
    # gather: block_t_raw[i, j] = k_subs[i][t_in_k[j]]  (unsorted)
    block_t_raw = k_arr[:, tidx[0], :]  # (num_k, num_t_in_k, t)
    # sort each row of t elements to match lexicographic t-subset order
    block_t_sorted = np.sort(block_t_raw, axis=2)  # (num_k, num_t_in_k, t)
    # look up IDs
    block_t_ids = pos_in_flat[block_t_sorted].reshape(num_k, num_t_in_k).astype(np.int32)

    # ---------- 3. Greedy Cover (vectorized gain via bincount) ----------
    def greedy_cover(deadline_g):
        covered = np.zeros(num_t, dtype=bool)
        used_blocks = np.zeros(num_k, dtype=bool)
        solution = []
        # current uncovered count per block = number of t-subsets in block not yet covered
        # We recompute gain each iteration using bincount over uncovered t-subsets.
        while not covered.all():
            if time.time() > deadline_g:
                break
            # gain[i] = number of t-subsets in block i that are still uncovered
            # Efficient: for each uncovered t-subset, add 1 to all blocks containing it.
            # We precompute the reverse: which blocks contain each t-subset? Too big.
            # Instead use: gain = sum over j of (block_t_ids[:,j] in uncovered)
            # Vectorized: for each of num_t_in_k columns, count where block_t_ids[:,j] is uncovered.
            gain = np.zeros(num_k, dtype=np.int32)
            # This is num_t_in_k boolean reductions over num_k — acceptable.
            for j in range(num_t_in_k):
                gain += (~covered[block_t_ids[:, j]]).astype(np.int32)
            gain[used_blocks] = -1
            best_idx = int(np.argmax(gain))
            if gain[best_idx] <= 0:
                break
            used_blocks[best_idx] = True
            solution.append(list(k_subs[best_idx]))
            covered[block_t_ids[best_idx]] = True
        return solution

    # ---------- 4. Randomized greedy restarts ----------
    def greedy_random(deadline_g):
        covered = np.zeros(num_t, dtype=bool)
        used_blocks = np.zeros(num_k, dtype=bool)
        solution = []
        while not covered.all():
            if time.time() > deadline_g:
                break
            gain = np.zeros(num_k, dtype=np.int32)
            for j in range(num_t_in_k):
                gain += (~covered[block_t_ids[:, j]]).astype(np.int32)
            gain[used_blocks] = -1
            if gain.max() <= 0:
                break
            # pick among top candidates with some randomness
            top = gain >= gain.max()
            candidates = np.where(top)[0]
            best_idx = int(candidates[random.randrange(len(candidates))])
            used_blocks[best_idx] = True
            solution.append(list(k_subs[best_idx]))
            covered[block_t_ids[best_idx]] = True
        return solution

    # Build index mapping
    block_to_idx = {block: i for i, block in enumerate(k_subs)}

    # ---------- 5. Redundancy removal (incremental-friendly) ----------
    def remove_redundant(sol):
        if not sol:
            return sol, False
        cover_count = np.zeros(num_t, dtype=np.int32)
        sol_indices = [block_to_idx[tuple(b)] for b in sol]
        for idx in sol_indices:
            cover_count[block_t_ids[idx]] += 1

        new_sol = []
        changed = False
        for idx in sol_indices:
            tids = block_t_ids[idx]
            if np.min(cover_count[tids]) > 1:
                changed = True
                cover_count[tids] -= 1
            else:
                new_sol.append(list(k_subs[idx]))
        return new_sol, changed

    # ---------- 6. Pairwise merge ----------
    def pairwise_merge(sol, deadline_m):
        n = len(sol)
        if n < 2 or time.time() > deadline_m:
            return sol, False

        sol_indices = [block_to_idx[tuple(b)] for b in sol]
        cover_count = np.zeros(num_t, dtype=np.int32)
        for idx in sol_indices:
            cover_count[block_t_ids[idx]] += 1

        # unique t-subsets per block
        unique_covered = []
        for idx in sol_indices:
            tids = block_t_ids[idx]
            unique = tids[cover_count[tids] == 1]
            unique_covered.append(unique)

        changed = False
        new_sol = list(sol)

        for i in range(n):
            if time.time() > deadline_m:
                break
            for j in range(i + 1, n):
                if new_sol[i] is None or new_sol[j] is None:
                    continue
                union_unique = np.concatenate([unique_covered[i], unique_covered[j]])
                if len(union_unique) == 0:
                    continue
                # elements that must be in the new block
                all_elements = set()
                for tid in union_unique:
                    all_elements.update(t_subs[tid])
                if len(all_elements) > k:
                    continue
                need = k - len(all_elements)
                if need < 0:
                    continue
                remaining = [x for x in range(v) if x not in all_elements]
                # try a few random fillers
                found = False
                for _ in range(3):
                    if need == 0:
                        filler = []
                    else:
                        filler = random.sample(remaining, need)
                    new_block = tuple(sorted(all_elements | set(filler)))
                    if new_block in block_to_idx:
                        new_idx = block_to_idx[new_block]
                        new_tid_set = set(block_t_ids[new_idx].tolist())
                        if set(union_unique.tolist()).issubset(new_tid_set):
                            new_sol[i] = None
                            new_sol[j] = None
                            new_sol.append(list(new_block))
                            changed = True
                            found = True
                            break
                if found:
                    break
            if changed:
                break

        if changed:
            new_sol = [b for b in new_sol if b is not None]
        return new_sol, changed

    # ---------- 7. WalkSAT-style local search (sampled candidates) ----------
    def walksat(sol, deadline_w):
        included = np.zeros(num_k, dtype=bool)
        for b in sol:
            included[block_to_idx[tuple(b)]] = True

        cover_count = np.zeros(num_t, dtype=np.int32)
        for i in range(num_k):
            if included[i]:
                cover_count[block_t_ids[i]] += 1

        num_uncovered = int(np.sum(cover_count == 0))
        best_num_uncovered = num_uncovered
        best_included = included.copy()

        max_steps = 4000
        noise = 0.25
        sample_size = 2000
        step = 0
        while step < max_steps and time.time() < deadline_w:
            step += 1

            if random.random() < noise:
                i = random.randint(0, num_k - 1)
            else:
                # sample candidate blocks
                if sample_size < num_k:
                    cand = np.random.choice(num_k, size=sample_size, replace=False)
                else:
                    cand = np.arange(num_k)

                delta = np.zeros(len(cand), dtype=np.int32)
                for c in cand:
                    tids = block_t_ids[c]
                    if included[c]:
                        delta[cand == c] = np.sum(cover_count[tids] == 1)
                    else:
                        delta[cand == c] = -np.sum(cover_count[tids] == 0)
                best_c = cand[int(np.argmin(delta))]
                i = int(best_c)

            if included[i]:
                cover_count[block_t_ids[i]] -= 1
                included[i] = False
            else:
                cover_count[block_t_ids[i]] += 1
                included[i] = True

            num_uncovered = int(np.sum(cover_count == 0))
            if num_uncovered < best_num_uncovered:
                best_num_uncovered = num_uncovered
                best_included = included.copy()
                if best_num_uncovered == 0:
                    break
        best_solution = [list(k_subs[i]) for i in range(num_k) if best_included[i]]
        return best_solution

    # ---------- Main pipeline ----------
    # Initial greedy
    solution = greedy_cover(deadline - 120.0)

    # Randomized greedy restarts (if time)
    if time.time() < deadline - 110.0:
        for _ in range(5):
            if time.time() > deadline - 110.0:
                break
            cand = greedy_random(deadline - 110.0)
            if len(cand) < len(solution):
                solution = cand

    # Redundancy removal
    for _ in range(10):
        if time.time() > deadline - 100.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break

    # Pairwise merge
    for _ in range(30):
        if time.time() > deadline - 90.0:
            break
        solution, changed = pairwise_merge(solution, deadline - 90.0)
        if not changed:
            break

    # Redundancy again
    for _ in range(10):
        if time.time() > deadline - 85.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break

    # WalkSAT
    if time.time() < deadline - 15.0:
        ws_solution = walksat(solution, deadline - 15.0)
        if len(ws_solution) < len(solution):
            solution = ws_solution

    # Final polish: redundancy + merge
    for _ in range(20):
        if time.time() > deadline - 8.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break
    for _ in range(20):
        if time.time() > deadline - 4.0:
            break
        solution, changed = pairwise_merge(solution, deadline - 4.0)
        if not changed:
            break
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        solution, changed = remove_redundant(solution)
        if not changed:
            break

    return solution
