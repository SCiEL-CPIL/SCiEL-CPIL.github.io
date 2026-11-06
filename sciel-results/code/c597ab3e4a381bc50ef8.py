import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 85.0

    # ---------- Precomputation ----------
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append(idx)
        ap_index_by_elem[b].append(idx)
        ap_index_by_elem[c].append(idx)

    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    mask4 = [[] for _ in range(n + 1)]
    for ap in four_aps:
        for i in range(4):
            x = ap[i]
            others = 0
            for j in range(4):
                if j != i:
                    others |= (1 << (ap[j] - 1))
            mask4[x].append(others)

    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for idx in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            others = [y for y in (a, b, c) if y != x]
            gain_pairs[x].append((1 << (others[0] - 1), 1 << (others[1] - 1)))

    # ---------- Core operations ----------
    def can_add(x, S):
        for m in mask4[x]:
            if (S & m) == m:
                return False
        return True

    def gain_add(x, S):
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S & bit1) and (S & bit2):
                g += 1
        return g

    def loss_remove(x, S):
        S_wo = S & ~(1 << (x - 1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo & bit1) and (S_wo & bit2):
                g += 1
        return g

    def score_full(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S >> (a - 1)) & 1 and (S >> (b - 1)) & 1 and (S >> (c - 1)) & 1:
                cnt += 1
        return cnt

    def is_valid(S):
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                return False
        return True

    # ---------- Local search ----------
    def local_search(S, sc, max_rounds=50, time_budget=None):
        S = int(S)
        for rnd in range(max_rounds):
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False

            # Greedy add pass
            for x in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget:
                    break
                if (S >> (x - 1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x - 1))
                    sc += g
                    improved = True

            if not improved:
                break

            # Try 1-swap moves
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget:
                    break
                if not ((S >> (y - 1)) & 1):
                    continue
                S_wo = S & ~(1 << (y - 1))
                sc_wo = sc - loss_remove(y, S)
                best_x = -1
                best_g = 0
                for x in range(1, n + 1):
                    if (S_wo >> (x - 1)) & 1:
                        continue
                    if not can_add(x, S_wo):
                        continue
                    g = gain_add(x, S_wo)
                    if g > best_g:
                        best_g = g
                        best_x = x
                if best_x > 0 and sc_wo + best_g > sc:
                    S = S_wo | (1 << (best_x - 1))
                    sc = sc_wo + best_g
                    improved = True

        return S, sc

    # ---------- SA ----------
    def sa_search(S, sc, max_iters=300, time_budget=None):
        S = int(S)
        T = 4.0
        T_min = 0.1
        cool = 0.98
        for _ in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break
            r = random.random()
            if r < 0.45:
                cands = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1) and can_add(x, S)]
                if not cands:
                    continue
                x = random.choice(cands)
                S_new = S | (1 << (x - 1))
                delta = gain_add(x, S)
            elif r < 0.75:
                in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
                if not in_s:
                    continue
                y = random.choice(in_s)
                S_new = S & ~(1 << (y - 1))
                delta = -loss_remove(y, S)
            else:
                in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
                not_s = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1)]
                if not in_s or not not_s:
                    continue
                y = random.choice(in_s)
                S_wo = S & ~(1 << (y - 1))
                valid = [x for x in not_s if can_add(x, S_wo)]
                if not valid:
                    continue
                x = random.choice(valid)
                S_new = S_wo | (1 << (x - 1))
                delta = gain_add(x, S_wo) - loss_remove(y, S)

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                sc += delta
            T = max(T_min, T * cool)

        for x in range(1, n + 1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x - 1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x - 1))
                sc += g
        return S, sc

    # ---------- Tabu Search ----------
    def tabu_search(S, sc, max_iters=500, tabu_len=30, time_budget=None):
        S = int(S)
        tabu = {}  # element -> iteration when it was last moved
        best_S = S
        best_sc = sc
        T = 2.0
        T_min = 0.1
        cool = 0.995

        for it in range(max_iters):
            if time_budget is not None and time.time() > time_budget:
                break

            in_s = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]
            not_s = [x for x in range(1, n + 1) if not ((S >> (x - 1)) & 1)]
            if not in_s or not not_s:
                break

            best_move = None
            best_delta = -float('inf')
            best_is_tabu = False

            # Try add moves
            for x in not_s:
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                delta = g
                is_tabu = (x in tabu and it - tabu[x] < tabu_len)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('add', x, None)
                    best_is_tabu = is_tabu

            # Try remove moves
            for y in in_s:
                l = loss_remove(y, S)
                delta = -l
                is_tabu = (y in tabu and it - tabu[y] < tabu_len)
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', y, None)
                    best_is_tabu = is_tabu

            # Try swap moves (sample for speed)
            n_swaps = min(20, len(in_s) * len(not_s))
            for _ in range(n_swaps):
                y = random.choice(in_s)
                x = random.choice(not_s)
                S_wo = S & ~(1 << (y - 1))
                if not can_add(x, S_wo):
                    continue
                delta = gain_add(x, S_wo) - loss_remove(y, S)
                is_tabu = ((y in tabu and it - tabu[y] < tabu_len) or
                          (x in tabu and it - tabu[x] < tabu_len))
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('swap', x, y)
                    best_is_tabu = is_tabu

            if best_move is None:
                break

            # Accept if not tabu, or if it improves best
            accept = (not best_is_tabu) or (sc + best_delta > best_sc)
            if accept:
                move_type = best_move[0]
                if move_type == 'add':
                    x = best_move[1]
                    S |= (1 << (x - 1))
                    tabu[x] = it
                elif move_type == 'remove':
                    y = best_move[1]
                    S &= ~(1 << (y - 1))
                    tabu[y] = it
                else:  # swap
                    x, y = best_move[1], best_move[2]
                    S &= ~(1 << (y - 1))
                    S |= (1 << (x - 1))
                    tabu[y] = it
                    tabu[x] = it

                sc += best_delta
                if sc > best_sc:
                    best_sc = sc
                    best_S = S

            T = max(T_min, T * cool)

        # Final greedy polish
        for x in range(1, n + 1):
            if time_budget is not None and time.time() > time_budget:
                break
            if (S >> (x - 1)) & 1:
                continue
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g > 0:
                S |= (1 << (x - 1))
                sc += g

        if sc > best_sc:
            best_sc = sc
            best_S = S
        return best_S, best_sc

    # ---------- Construction helpers ----------
    def greedy_build(order, min_g=0):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            g = gain_add(x, S)
            if g >= min_g:
                S |= (1 << (x - 1))
        return S

    def prune_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                best_loss = float('inf')
                best_elem = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_elem = e
                S &= ~(1 << (best_elem - 1))
        return S

    def random_density(p):
        S = 0
        for x in range(1, n + 1):
            if random.random() < p:
                S |= (1 << (x - 1))
        return S

    def repair(S):
        for ap in four_aps:
            if all((S >> (e - 1)) & 1 for e in ap):
                best_loss = float('inf')
                best_e = ap[0]
                for e in ap:
                    l = loss_remove(e, S)
                    if l < best_loss:
                        best_loss = l
                        best_e = e
                S &= ~(1 << (best_e - 1))
        return S

    # ---------- Main search ----------
    random.seed(12345)
    best_S = 0
    best_score = 0

    def update_best(S, sc):
        nonlocal best_S, best_score
        if sc > best_score:
            best_S = S
            best_score = sc
            return True
        return False

    # Strategy A: Prune from full + local search
    if time.time() < deadline - 35:
        S = prune_full()
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=40, time_budget=deadline - 30)
        update_best(S, sc)

    # Strategy B: Random restarts with greedy build + local search
    n_restarts = 60
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=20, time_budget=deadline - 20)
        update_best(S, sc)

    # Strategy C: Random density + repair + local search
    for p in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        if time.time() > deadline - 25:
            break
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=20, time_budget=deadline - 20)
        update_best(S, sc)

    # Strategy D: SA from best
    if time.time() < deadline - 20 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = sa_search(S, sc, max_iters=800, time_budget=deadline - 15)
        update_best(S, sc)

    # Strategy E: Tabu search from best
    if time.time() < deadline - 18 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = tabu_search(S, sc, max_iters=600, tabu_len=30, time_budget=deadline - 14)
        update_best(S, sc)

    # Strategy F: More random restarts with SA
    for r in range(30):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = sa_search(S, sc, max_iters=250, time_budget=deadline - 12)
        update_best(S, sc)

    # Strategy G: SA from random density
    for p in [0.35, 0.45, 0.55, 0.65]:
        if time.time() > deadline - 15:
            break
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = sa_search(S, sc, max_iters=300, time_budget=deadline - 12)
        update_best(S, sc)

    # Strategy H: Tabu from random starts
    for r in range(15):
        if time.time() > deadline - 12:
            break
        p = random.uniform(0.3, 0.6)
        S = random_density(p)
        S = repair(S)
        sc = score_full(S)
        S, sc = tabu_search(S, sc, max_iters=400, tabu_len=25, time_budget=deadline - 10)
        update_best(S, sc)

    # Strategy I: SA from tabu results
    if time.time() < deadline - 10 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = sa_search(S, sc, max_iters=500, time_budget=deadline - 7)
        update_best(S, sc)

    # Strategy J: More diverse restarts
    for r in range(30):
        if time.time() > deadline - 8:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = greedy_build(order, min_g=0)
        sc = score_full(S)
        S, sc = local_search(S, sc, max_rounds=25, time_budget=deadline - 6)
        update_best(S, sc)

    # Strategy K: Final tabu polish
    if time.time() < deadline - 5 and best_S:
        S = int(best_S)
        sc = best_score
        S, sc = tabu_search(S, sc, max_iters=400, tabu_len=20, time_budget=deadline - 3)
        update_best(S, sc)

    # Final polish
    if time.time() < deadline - 2 and best_S:
        best_S, best_score = local_search(best_S, best_score, max_rounds=50, time_budget=deadline - 0.5)

    # Verify and return
    result = [x for x in range(1, n + 1) if (best_S >> (x - 1)) & 1]
    if not result:
        result = [1]

    # Final validity check
    if not is_valid(best_S):
        S = 0
        for x in range(1, n + 1):
            if can_add(x, S):
                S |= (1 << (x - 1))
        result = [x for x in range(1, n + 1) if (S >> (x - 1)) & 1]

    return result
