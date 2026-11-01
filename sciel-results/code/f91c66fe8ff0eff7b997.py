import time
import random
import numpy as np
from scipy.optimize import linprog

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 145.0

    total = q ** n
    ball_size = 1 + n * (q - 1)

    # Precompute base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Compact ball representation: each codeword -> 2187-bit int bitmask
    ball_mask = [0] * total
    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        m = 1 << c
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                m |= 1 << w
        ball_mask[c] = m

    # Coverage matrix (float, small 2187x2187) for fast vectorized pricing
    cov = np.zeros((total, total), dtype=np.float64)
    for c in range(total):
        temp = c
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        cov[c, c] = 1.0
        for coord in range(n):
            orig = digits[coord]
            for v in range(q):
                if v == orig:
                    continue
                w = c + (v - orig) * base[coord]
                cov[c, w] = 1.0

    # ---------- integer cover repair (bitmask based) ----------
    def integer_cover(initial):
        code = list(dict.fromkeys(initial))
        covered = 0
        for c in code:
            covered |= ball_mask[c]
        for _ in range(10):
            changed = False
            for c in list(code):
                if (covered & ~ball_mask[c]) == (covered ^ ball_mask[c]) or \
                   (covered & ball_mask[c]) == ball_mask[c] and (covered & ~ball_mask[c]) != 0:
                    pass
                # remove c if no word is uniquely covered by it
                if (covered & ~ball_mask[c]) != 0:
                    # check if every word in ball_mask[c] is covered by another
                    if (covered & ball_mask[c]) == ball_mask[c] and \
                       (covered ^ ball_mask[c]) != 0:
                        # recompute: unique = ball_mask[c] & ~(covered ^ ball_mask[c])
                        unique = ball_mask[c] & ~(covered ^ ball_mask[c])
                        if unique == 0:
                            covered ^= ball_mask[c]
                            code.remove(c)
                            changed = True
            if not changed:
                break
        return code

    def try_delete(code):
        covered = 0
        for c in code:
            covered |= ball_mask[c]
        changed = True
        while changed:
            changed = False
            for c in list(code):
                unique = ball_mask[c] & ~(covered ^ ball_mask[c])
                if unique == 0:
                    covered ^= ball_mask[c]
                    code.remove(c)
                    changed = True
        return code

    def local_search(code, time_budget=5.0):
        code = list(dict.fromkeys(code))
        t_start = time.time()
        best_size = len(code)
        best_code = list(code)

        def recompute():
            nonlocal covered
            covered = 0
            for c in code:
                covered |= ball_mask[c]
        covered = 0
        for c in code:
            covered |= ball_mask[c]

        try_delete(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        no_success = 0
        while time.time() - t_start < time_budget:
            if len(code) == 0:
                break
            deleted = False
            for c in list(code):
                unique = ball_mask[c] & ~(covered ^ ball_mask[c])
                if unique == 0:
                    covered ^= ball_mask[c]
                    code.remove(c)
                    deleted = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                    break
            if deleted:
                continue

            # 1-for-1 swap
            c_remove = random.choice(code)
            critical = ball_mask[c_remove] & ~(covered ^ ball_mask[c_remove])
            if critical == 0:
                covered ^= ball_mask[c_remove]
                code.remove(c_remove)
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
                continue

            critical_words = [w for w in range(total) if (critical >> w) & 1]
            # find candidates covering all critical words (vectorized)
            need = np.zeros(total, dtype=bool)
            for w in critical_words:
                need |= (cov[:, w] == 1)
            in_set = np.zeros(total, dtype=bool)
            for c in code:
                in_set[c] = True
            cand = np.where(need & ~in_set)[0]
            if len(cand) > 0:
                c_add = int(random.choice(cand))
                covered = (covered & ~ball_mask[c_remove]) | ball_mask[c_add]
                code.remove(c_remove)
                code.append(c_add)
                no_success = 0
                try_delete(code)
                recompute()
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = list(code)
            else:
                no_success += 1

            # 2-for-1
            if no_success > 400 and time.time() - t_start < time_budget - 0.3:
                c_remove = random.choice(code)
                critical = ball_mask[c_remove] & ~(covered ^ ball_mask[c_remove])
                critical_words = [w for w in range(total) if (critical >> w) & 1]
                if critical_words:
                    for _ in range(40):
                        c1 = random.randrange(total)
                        c2 = random.randrange(total)
                        if c1 == c2 or c1 in code or c2 in code or c1 == c_remove or c2 == c_remove:
                            continue
                        ok = True
                        for w in critical_words:
                            if not (((ball_mask[c1] >> w) & 1) or ((ball_mask[c2] >> w) & 1)):
                                ok = False
                                break
                        if ok:
                            covered = (covered & ~ball_mask[c_remove]) | ball_mask[c1] | ball_mask[c2]
                            code.remove(c_remove)
                            code.append(c1)
                            code.append(c2)
                            no_success = 0
                            try_delete(code)
                            recompute()
                            if len(code) < best_size:
                                best_size = len(code)
                                best_code = list(code)
                            break
                    else:
                        no_success += 1
                else:
                    no_success += 1

            if no_success > 1200:
                rand_c = random.randrange(total)
                if rand_c not in code:
                    code.append(rand_c)
                    recompute()
                    try_delete(code)
                    recompute()
                no_success = 0

        return best_code, best_size

    # ---------- greedy ----------
    def greedy_build(seed=None, price=None):
        if seed is not None:
            random.seed(seed)
        covered = 0
        code = []
        uncovered = total
        remaining = list(range(total))
        while uncovered > 0:
            # gain = number of currently-uncovered words in ball
            gains = np.zeros(total, dtype=np.float64)
            # uncovered indicator
            unc = np.zeros(total, dtype=np.float64)
            # build uncovered vector from bitmask
            m = (~covered) & ((1 << total) - 1)
            # vectorize: use cov @ (1 - covered_vec)
            covvec = np.zeros(total, dtype=np.float64)
            # set covvec[w]=1 if uncovered
            # iterate bits is slow; instead compute via matrix on covered indicator
            # covered indicator
            covind = np.zeros(total, dtype=np.float64)
            for c in code:
                covind[cov[c] == 1] = 1.0
            gains = cov @ (1.0 - covind)
            maxg = np.max(gains)
            if maxg == 0:
                break
            cands = np.where(gains == maxg)[0]
            best_c = int(random.choice(cands))
            code.append(best_c)
            covered |= ball_mask[best_c]
            uncovered = total - int((covered).bit_count())
        return code

    # ---------- LP column generation ----------
    def lp_column_generation(seed=None, time_budget=20.0):
        if seed is not None:
            random.seed(seed)
        # start with a greedy solution as initial columns
        init = greedy_build(seed=seed)
        cols = set(init)
        p = np.ones(total, dtype=np.float64) / ball_size  # initial dual prices
        t_start = time.time()
        best = None
        best_size = len(init)
        best_code = list(init)

        while time.time() - t_start < time_budget:
            # pricing: find codeword maximizing sum of p over its ball
            gain = cov @ p
            # exclude already-in-columns from being the only new one? allow all
            maxg = np.max(gain)
            new_c = int(np.argmax(gain))
            # add a few top candidates
            top = np.argsort(-gain)[:5]
            added = False
            for tc in top:
                tc = int(tc)
                if tc not in cols:
                    cols.add(tc)
                    added = True
            if not added:
                break

            # solve LP relaxation on current columns
            col_list = list(cols)
            A = cov[:, col_list]  # total x len(cols)
            cvec = np.ones(len(col_list))
            res = linprog(cvec, A_ub=-A, b_ub=-np.ones(total),
                          bounds=[(0, 1)] * len(col_list),
                          method='highs', options={'time_limit': 2.0, 'presolve': True})
            if res.status not in (0, 1):
                # update p from infeasibility? just use gains
                p = gain / (np.sum(gain) + 1e-9)
                continue
            x = res.x
            # update dual prices: p ~ gain normalized (use current gain as price signal)
            p = gain / (np.sum(gain) + 1e-9)
            # round: take columns with x > 0.5, then fill gaps greedily
            selected = [col_list[i] for i in range(len(col_list)) if x[i] > 0.4]
            covered = 0
            for c in selected:
                covered |= ball_mask[c]
            # fill remaining uncovered with greedy
            while True:
                unc = total - int(covered.bit_count())
                if unc == 0:
                    break
                covind = np.zeros(total, dtype=np.float64)
                m = covered
                # build covered indicator
                for c in selected:
                    covind[cov[c] == 1] = 1.0
                gains = cov @ (1.0 - covind)
                mg = np.max(gains)
                if mg == 0:
                    break
                nc = int(np.argmax(gains))
                selected.append(nc)
                covered |= ball_mask[nc]
            selected = list(dict.fromkeys(selected))
            selected = try_delete(selected)
            if len(selected) < best_size:
                best_size = len(selected)
                best_code = list(selected)
                best = selected
            if time.time() - t_start > time_budget:
                break
        return best_code, best_size

    # ---------- main ----------
    best_code_overall = None
    best_size_overall = total

    # initial LP-guided
    code, sz = lp_column_generation(seed=42, time_budget=18.0)
    ls_code, ls_sz = local_search(code, time_budget=12.0)
    best_code_overall = ls_code
    best_size_overall = ls_sz

    seed = 0
    while time.time() < deadline - 12.0 and best_size_overall > 170:
        seed += 1
        if time.time() > deadline - 12.0:
            break
        time_left = deadline - 12.0 - time.time()
        # alternate LP-guided and greedy
        if seed % 2 == 1:
            code, sz = lp_column_generation(seed=seed, time_budget=min(12.0, max(3.0, time_left / 6.0)))
        else:
            code = greedy_build(seed=seed)
        tb = min(10.0, max(2.0, (deadline - 12.0 - time.time()) / 6.0))
        ls_code, ls_sz = local_search(code, time_budget=tb)
        if ls_sz < best_size_overall:
            best_size_overall = ls_sz
            best_code_overall = ls_code

    # final polish
    if time.time() < deadline - 3.0:
        tb = min(8.0, deadline - 3.0 - time.time())
        ls_code, ls_sz = local_search(best_code_overall, time_budget=tb)
        if ls_sz < best_size_overall:
            best_size_overall = ls_sz
            best_code_overall = ls_code

    result = []
    for idx in best_code_overall:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
    return result
