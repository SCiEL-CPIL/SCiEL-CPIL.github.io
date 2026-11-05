import time
import random
import numpy as np
import sys


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s wall-clock

    # ---- Generic fallback for non-target parameters ----
    if not (q == 3 and n == 7 and radius == 1):
        return _generic(q, n, radius)

    N = 3 ** n                       # 2187
    ball_size = 1 + n * (q - 1)      # 15

    # ---------- base-3 <-> int helpers ----------
    def int_to_base3(num, length):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % 3
            num //= 3
        return res

    def base3_to_int(digits):
        res = 0
        for d in digits:
            res = res * 3 + d
        return res

    # ---------- precompute balls (radius-1 neighborhoods) ----------
    balls = []
    for i in range(N):
        digits = int_to_base3(i, n)
        neighbors = [i]
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    neighbors.append(base3_to_int(new_digits))
        balls.append(neighbors)

    # cover_map[w] = list of codewords c whose ball contains w
    cover_map = [[] for _ in range(N)]
    for c in range(N):
        for w in balls[c]:
            cover_map[w].append(c)

    # numpy matrix B[c][w] = 1 if w in ball(c)  -> vectorized gain sums
    B = np.zeros((N, N), dtype=np.int32)
    for c in range(N):
        B[c, balls[c]] = 1

    # ---------- greedy set-cover with randomized tie-break (vectorized) ----------
    def greedy_cover(seed):
        rng = random.Random(seed)
        covered = np.zeros(N, dtype=bool)
        cnt = np.array([ball_size] * N, dtype=np.int32)
        code = []
        uncovered_count = N
        # Precompute: for each codeword c, the list of words in its ball as np array
        ball_arrays = [np.array(b, dtype=np.int64) for b in balls]
        while uncovered_count > 0:
            max_val = int(cnt.max())
            cands = np.flatnonzero(cnt == max_val)
            c = int(rng.choice(cands))
            code.append(c)
            ball_c = ball_arrays[c]
            # Vectorized: find newly covered words
            newly_mask = ~covered[ball_c]
            n_newly = int(newly_mask.sum())
            if n_newly > 0:
                newly = ball_c[newly_mask]
                covered[newly] = True
                uncovered_count -= n_newly
                # Update cnt: for each newly covered word w, decrement cnt for all codewords covering w
                # Vectorized approach: use B matrix
                # cnt[c'] -= number of newly covered words in ball(c')
                # = cnt -= B[newly, :].sum(axis=0) ... but B is N x N, B[row, col] = 1 if col in ball(row)
                # Actually B[c][w] = 1 if w in ball(c). So B[newly, :] gives for each newly word, which codewords cover it.
                # We want for each codeword c', how many newly words are in ball(c')
                # That's B.T[newly, :].sum(axis=0) i.e. sum over newly of B[c', w]
                # B.T is N x N where B.T[w, c'] = B[c'][w] = 1 if w in ball(c')
                # So B.T[newly, :].sum(axis=0) gives exactly what we need
                cnt -= B.T[newly, :].sum(axis=0, dtype=np.int32)
        return code

    # ---------- pure redundancy removal ----------
    def refine(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if not (cover_count[balls[c]] > 1).all():
                    continue
                cover_count[balls[c]] -= 1
                code_list.pop(i)
                changed = True
                break
        return code_list

    # ---------- local search: remove-one + relocate/swap + 2-swap + kick ----------
    def local_search(code, seed, time_limit):
        rng = random.Random(seed)
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        code_set = set(code_list)
        # Precompute ball arrays for faster indexing
        ball_arrays = [np.array(b, dtype=np.int64) for b in balls]

        kick_size = 5
        no_improve_count = 0
        max_no_improve = 6

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            indices = list(range(len(code_list)))
            rng.shuffle(indices)
            for i in indices:
                if time.time() > time_limit:
                    break
                c_remove = code_list[i]
                ball_r = ball_arrays[c_remove]
                cover_count[ball_r] -= 1
                uncovered = ball_r[cover_count[ball_r] == 0]
                if uncovered.size == 0:
                    # pure removal
                    code_list.pop(i)
                    code_set.discard(c_remove)
                    improved = True
                    no_improve_count = 0
                    break

                # relocation: vectorized scan for a codeword covering all uncovered
                covers_all = np.all(B[:, uncovered] == 1, axis=1)
                # mask out already-present codewords
                present_mask = np.zeros(N, dtype=bool)
                for cc in code_set:
                    present_mask[cc] = True
                present_mask[c_remove] = False
                cand = covers_all & (~present_mask)
                cand[c_remove] = False
                cand_idx = np.flatnonzero(cand)
                if cand_idx.size > 0:
                    found = int(cand_idx[0])
                    cover_count[ball_arrays[found]] += 1
                    code_list[i] = found
                    code_set.discard(c_remove)
                    code_set.add(found)
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    cover_count[ball_r] += 1

            # Kick: if stuck, try 2-swap, then 3-swap, then random kick
            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve and len(code_list) > 1:
                    no_improve_count = 0
                    # 2-swap
                    pair = rng.sample(range(len(code_list)), 2)
                    a, b = pair
                    ca, cb = code_list[a], code_list[b]
                    ba, bb = ball_arrays[ca], ball_arrays[cb]
                    cover_count[ba] -= 1
                    cover_count[bb] -= 1
                    unc_mask = (cover_count[ba] == 0) | (cover_count[bb] == 0)
                    unc_all = np.unique(np.concatenate([ba[cover_count[ba] == 0], bb[cover_count[bb] == 0]]))
                    success = False
                    if unc_all.size > 0:
                        covers_all = np.all(B[:, unc_all] == 1, axis=1)
                        present_mask = np.zeros(N, dtype=bool)
                        for cc in code_set:
                            present_mask[cc] = True
                        present_mask[ca] = False
                        present_mask[cb] = False
                        cand = covers_all & (~present_mask)
                        cand[ca] = False
                        cand[cb] = False
                        cand_idx = np.flatnonzero(cand)
                        if cand_idx.size >= 1:
                            picks = []
                            if cand_idx.size >= 2:
                                sel = rng.choice(cand_idx, size=2, replace=False)
                                picks = [int(sel[0]), int(sel[1])]
                            else:
                                picks = [int(cand_idx[0])]
                            for p in picks:
                                cover_count[ball_arrays[p]] += 1
                                code_set.add(p)
                            code_list[a] = picks[0]
                            if len(picks) > 1:
                                code_list[b] = picks[1]
                            else:
                                code_list.pop(b)
                            code_set.discard(ca)
                            code_set.discard(cb)
                            improved = True
                            success = True
                    if not success:
                        cover_count[ba] += 1
                        cover_count[bb] += 1
                        # Try 3-swap
                        if len(code_list) > 2:
                            triple = rng.sample(range(len(code_list)), 3)
                            c3 = [code_list[t] for t in triple]
                            b3 = [ball_arrays[c] for c in c3]
                            for bi in b3:
                                cover_count[bi] -= 1
                            unc3 = set()
                            for bi in b3:
                                unc3.update(bi[cover_count[bi] == 0].tolist())
                            unc3 = np.array(sorted(unc3), dtype=np.int64) if unc3 else np.array([], dtype=np.int64)
                            success3 = False
                            if unc3.size > 0:
                                covers_all3 = np.all(B[:, unc3] == 1, axis=1)
                                present_mask3 = np.zeros(N, dtype=bool)
                                for cc in code_set:
                                    present_mask3[cc] = True
                                for cc in c3:
                                    present_mask3[cc] = False
                                    covers_all3[cc] = False
                                cand3 = covers_all3 & (~present_mask3)
                                cand3_idx = np.flatnonzero(cand3)
                                if cand3_idx.size >= 1:
                                    n_pick = min(3, cand3_idx.size)
                                    sel3 = rng.choice(cand3_idx, size=n_pick, replace=False)
                                    picks3 = [int(x) for x in sel3]
                                    for p in picks3:
                                        cover_count[ball_arrays[p]] += 1
                                        code_set.add(p)
                                    for j, t in enumerate(triple):
                                        if j < len(picks3):
                                            code_list[t] = picks3[j]
                                        else:
                                            # remove this one
                                            code_list.pop(t)
                                    for cc in c3:
                                        code_set.discard(cc)
                                    improved = True
                                    success3 = True
                            if not success3:
                                for bi in b3:
                                    cover_count[bi] += 1
                        # Random kick
                        if not improved:
                            kick_indices = rng.sample(range(len(code_list)), min(kick_size, len(code_list)))
                            for ki in kick_indices:
                                old_c = code_list[ki]
                                cover_count[ball_arrays[old_c]] -= 1
                                code_set.discard(old_c)
                                new_c = rng.randint(0, N - 1)
                                tries = 0
                                while new_c in code_set and tries < 50:
                                    new_c = rng.randint(0, N - 1)
                                    tries += 1
                                code_list[ki] = new_c
                                cover_count[ball_arrays[new_c]] += 1
                                code_set.add(new_c)
                            improved = True

        return code_list

    # ---------- final exhaustive single-removal polish ----------
    def final_polish(code):
        code_list = list(code)
        if not code_list:
            return code_list
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            cover_count[balls[c]] += 1
        # Repeatedly try to remove any codeword whose ball is fully covered by others
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                if not (cover_count[balls[c]] > 1).all():
                    continue
                cover_count[balls[c]] -= 1
                code_list.pop(i)
                changed = True
                break
        return code_list

    # ---------- driver: restarts -> best -> local search ----------
    best_code = None
    best_len = float('inf')

    # greedy restarts (vectorized)
    n_restarts = 0
    while time.time() < deadline - 35.0:
        seed = int((time.time() - start_time) * 100000) % (10 ** 9)
        code = greedy_cover(seed)
        refined = refine(code)
        if len(refined) < best_len:
            best_len = len(refined)
            best_code = refined
        n_restarts += 1

    if best_code is not None and time.time() < deadline - 2.0:
        # Multiple local search passes with different seeds
        ls_seeds = [12345, 99999, 77777, 31415, 27182]
        remaining_time = deadline - 2.0
        n_ls = len(ls_seeds)
        per_ls = max(1.0, (remaining_time - 1.0) / n_ls)
        for si, ls_seed in enumerate(ls_seeds):
            ls_deadline = min(remaining_time - (n_ls - 1 - si) * 0.5, time.time() + per_ls)
            if time.time() >= ls_deadline - 0.5:
                break
            ls_code = local_search(best_code, ls_seed, ls_deadline)
            if len(ls_code) < best_len:
                best_len = len(ls_code)
                best_code = ls_code
        # Final polish
        if time.time() < deadline - 0.5:
            polished = final_polish(best_code)
            if len(polished) < best_len:
                best_len = len(polished)
                best_code = polished

    if best_code is not None:
        final_code = final_polish(best_code)
    else:
        final_code = []

    return [int_to_base3(c, n) for c in final_code]


def _generic(q, n, radius):
    N = q ** n

    def int_to_base(num, length, base):
        res = [0] * length
        for i in range(length - 1, -1, -1):
            res[i] = num % base
            num //= base
        return res

    balls = []
    for i in range(N):
        digits = int_to_base(i, n, q)
        neighbors = set([i])
        for pos in range(n):
            original = digits[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digits = digits[:]
                    new_digits[pos] = new_val
                    val = 0
                    for d in new_digits:
                        val = val * q + d
                    neighbors.add(val)
        balls.append(list(neighbors))

    covered = set()
    code = []
    remaining = set(range(N))
    while remaining:
        best_c = None
        best_count = -1
        for c in remaining:
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        code.append(best_c)
        covered.update(balls[best_c])
        remaining -= set(balls[best_c])

    # light refinement
    cover_count = [0] * N
    for c in code:
        for w in balls[c]:
            cover_count[w] += 1
    changed = True
    while changed:
        changed = False
        for i in range(len(code)):
            c = code[i]
            if all(cover_count[w] > 1 for w in balls[c]):
                for w in balls[c]:
                    cover_count[w] -= 1
                code.pop(i)
                changed = True
                break

    return [int_to_base(c, n, q) for c in code]
