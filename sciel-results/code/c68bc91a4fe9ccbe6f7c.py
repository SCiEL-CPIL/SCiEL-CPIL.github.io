import time
import numpy as np
import random


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin under 150s wall-clock limit

    total = q ** n
    ball_size = 1 + n * (q - 1)  # 15 for q=3, n=7

    # index <-> word base
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q

    # Precompute balls as numpy index arrays
    ball_indices = [None] * total
    for i in range(total):
        temp = i
        digits = [0] * n
        for j in range(n - 1, -1, -1):
            digits[j] = temp % q
            temp //= q
        ball = [i]
        for coord in range(n):
            orig = digits[coord]
            b = base[coord]
            for v in range(q):
                if v == orig:
                    continue
                ball.append(i + (v - orig) * b)
        ball_indices[i] = np.array(ball, dtype=np.int64)

    def uncovered_count(cnt):
        return int(np.count_nonzero(cnt == 0))

    # ---- Size-limited greedy: stop at target, then repair uncovered points ----
    def greedy_build(seed, target):
        random.seed(seed)
        cnt = np.zeros(total, dtype=np.int32)
        code = []
        covered_count = 0
        remaining = target
        while remaining > 0 and covered_count < total:
            # candidate must cover >=1 uncovered point
            mask = np.any(cnt[ball_indices] == 0, axis=1)
            cand = np.flatnonzero(mask)
            if cand.size == 0:
                break
            # weight by new coverage
            newcov = np.count_nonzero(cnt[ball_indices[cand]] == 0, axis=1)
            # prefer high new coverage; tie-break randomly
            maxnc = newcov.max()
            top = cand[newcov >= max(nc * 0.5 for nc in [maxnc])]
            # pick from the top group with a bias toward the very best
            pool = top if top.size <= 40 else top
            best_idx = pool[random.randrange(pool.size)]
            b = ball_indices[best_idx]
            nc = int((cnt[b] == 0).sum())
            cnt[b] += 1
            covered_count += nc
            code.append(int(best_idx))
            remaining -= 1
        # repair: cover any still-uncovered points
        while True:
            uc = np.flatnonzero(cnt == 0)
            if uc.size == 0:
                break
            # pick a codeword covering the most uncovered
            best = -1
            bestnc = -1
            # sample a subset of candidates for speed
            step = max(1, total // 4000)
            for c in range(0, total, step):
                nc = int((cnt[ball_indices[c]] == 0).sum())
                if nc > bestnc:
                    bestnc = nc
                    best = c
            if best < 0 or bestnc <= 0:
                # fallback: brute force over uncovered points' neighborhoods
                for x in uc[:200]:
                    for c in ball_indices[x]:
                        nc = int((cnt[ball_indices[c]] == 0).sum())
                        if nc > bestnc:
                            bestnc = nc
                            best = int(c)
            if best < 0:
                break
            b = ball_indices[best]
            cnt[b] += 1
            code.append(int(best))
        return code, cnt

    # ---- Local search: deletion + simulated annealing on size-neutral moves ----
    def local_search(code_list, time_budget):
        t_start = time.time()
        code_set = set(code_list)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1

        best_size = len(code_set)
        best_code = list(code_set)

        def try_delete():
            nonlocal best_size, best_code
            changed = True
            while changed:
                changed = False
                for c in sorted(code_set):
                    if np.min(cnt[ball_indices[c]]) >= 2:
                        code_set.remove(c)
                        cnt[ball_indices[c]] -= 1
                        changed = True
                        if len(code_set) < best_size:
                            best_size = len(code_set)
                            best_code = list(code_set)

        try_delete()

        T = 1.0
        T_min = 0.02
        cool = 0.999

        def do_delete():
            nonlocal best_size, best_code
            for c in list(code_set):
                if np.min(cnt[ball_indices[c]]) >= 2:
                    code_set.remove(c)
                    cnt[ball_indices[c]] -= 1
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)

        def do_11():
            # size-neutral swap; accept by coverage-slack surrogate
            c_remove = random.choice(list(code_set))
            c_add = random.randint(0, total - 1)
            if c_add in code_set or c_add == c_remove:
                return
            ball_rem = ball_indices[c_remove]
            ball_add = ball_indices[c_add]
            add_set = set(ball_add.tolist())
            # points lost: in ball_rem not ball_add with cnt==1
            lost = 0
            for x in ball_rem:
                if x not in add_set and cnt[x] == 1:
                    lost += 1
                    if lost > 2:
                        return
            # gained slack: in ball_add not ball_rem with cnt==1
            gained = 0
            for x in ball_add:
                if x not in set(ball_rem.tolist()) and cnt[x] == 1:
                    gained += 1
            delta = gained - lost  # surrogate: more slack = better
            if delta >= 0 or random.random() < np.exp(delta / T):
                code_set.remove(c_remove)
                code_set.add(c_add)
                cnt[ball_rem] -= 1
                cnt[ball_add] += 1
                do_delete()

        def do_21():
            # reduce size by 1
            if len(code_set) < 2:
                return
            c_rem1, c_rem2 = random.sample(list(code_set), 2)
            c_add1 = random.randint(0, total - 1)
            if c_add1 in code_set:
                return
            br1 = ball_indices[c_rem1]
            br2 = ball_indices[c_rem2]
            ba = ball_indices[c_add1]
            add_set = set(ba.tolist())
            ok = True
            for x in br1:
                rem = (1 if x in set(br2.tolist()) else 0)
                add = 1 if x in add_set else 0
                if cnt[x] - (1 if True else 0) + add < 1 and not (x in set(br2.tolist())):
                    # x only covered by c_rem1
                    if cnt[x] == 1 and x not in add_set and x not in set(br2.tolist()):
                        ok = False
                        break
            # full exact check
            for x in br1:
                r = 1
                if x in set(br2.tolist()):
                    r = 0
                a = 1 if x in add_set else 0
                if cnt[x] - r + a < 1:
                    ok = False
                    break
            if ok:
                for x in br2:
                    r = 1
                    if x in set(br1.tolist()):
                        r = 0
                    a = 1 if x in add_set else 0
                    if cnt[x] - r + a < 1:
                        ok = False
                        break
            if ok:
                code_set.remove(c_rem1)
                code_set.remove(c_rem2)
                code_set.add(c_add1)
                cnt[br1] -= 1
                cnt[br2] -= 1
                cnt[ba] += 1
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                do_delete()

        while time.time() - t_start < time_budget:
            r = random.random()
            if r < 0.45:
                do_11()
            elif r < 0.75:
                do_21()
            else:
                do_delete()
            T = max(T_min, T * cool)

        return best_code, best_size

    best_code_overall = []
    best_size_overall = total

    # First run: greedy to a reasonable target, then long LS
    gcode, gcnt = greedy_build(42, 210)
    best_code, best_size = local_search(gcode, time_budget=15.0)
    best_code_overall = best_code
    best_size_overall = best_size

    # Multi-restart with adaptive targets
    seed = 0
    while time.time() < deadline - 6.0 and best_size_overall > 160:
        seed += 1
        target = max(160, best_size_overall - random.randint(0, 15))
        gcode, gcnt = greedy_build(seed, target)
        time_left = deadline - 6.0 - time.time()
        tb = min(8.0, max(0.8, time_left / 10.0))
        best_code, best_size = local_search(gcode, time_budget=tb)
        if best_size < best_size_overall:
            best_size_overall = best_size
            best_code_overall = best_code

    # Final polish: exact deletion sweep
    if time.time() < deadline - 1.0:
        code_set = set(best_code_overall)
        cnt = np.zeros(total, dtype=np.int32)
        for c in code_set:
            cnt[ball_indices[c]] += 1
        changed = True
        while changed:
            changed = False
            for c in sorted(code_set):
                if np.min(cnt[ball_indices[c]]) >= 2:
                    code_set.remove(c)
                    cnt[ball_indices[c]] -= 1
                    changed = True
        best_code_overall = list(code_set)
        best_size_overall = len(code_set)

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
