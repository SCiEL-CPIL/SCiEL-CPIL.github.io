"""Combined program: fast bitmask precomputation + conflict-aware greedy +
simulated annealing with 2-swap moves + multi-restart, under a single deadline."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 88.0   # safety margin under 90 s wall-clock
    N = 256

    # ---- Precompute 3-AP and 4-AP participation lists (fast incremental ops) ----
    elem_to_3aps = [[] for _ in range(N + 1)]
    three_ap_bitmasks = []
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            idx = len(three_ap_bitmasks)
            three_ap_bitmasks.append(mask)
            elem_to_3aps[a].append(idx)
            elem_to_3aps[b].append(idx)
            elem_to_3aps[c].append(idx)

    elem_to_4aps = [[] for _ in range(N + 1)]
    four_ap_bitmasks = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            idx = len(four_ap_bitmasks)
            four_ap_bitmasks.append(mask)
            elem_to_4aps[a].append(idx)
            elem_to_4aps[b].append(idx)
            elem_to_4aps[c].append(idx)
            elem_to_4aps[e].append(idx)

    # ---- Fast incremental helpers ----
    def count_3aps(mask):
        c = 0
        for m in three_ap_bitmasks:
            if (mask & m) == m:
                c += 1
        return c

    def creates_4ap(x, mask):
        xb = 1 << (x - 1)
        nm = mask | xb
        for i in elem_to_4aps[x]:
            m = four_ap_bitmasks[i]
            if (nm & m) == m:
                return True
        return False

    def gain_3ap(x, mask):
        xb = 1 << (x - 1)
        g = 0
        for i in elem_to_3aps[x]:
            m = three_ap_bitmasks[i]
            if (mask & (m & ~xb)) == (m & ~xb):
                g += 1
        return g

    # ---- Conflict-aware greedy construction ----
    def greedy_conflict(w1=1.0, w2=0.06, seed=None, time_budget=None):
        if seed is not None:
            random.seed(seed)
        mask = 0
        remaining = set(range(1, N + 1))
        while remaining and (time_budget is None or time.time() < time_budget):
            best_x = None
            best_score = -1e18
            for x in remaining:
                if creates_4ap(x, mask):
                    continue
                g = gain_3ap(x, mask)
                # conflict proxy: how many remaining elements share a 4-AP structure with x
                conflict = 0
                for i in elem_to_4aps[x]:
                    m = four_ap_bitmasks[i]
                    conflict += bin(m & ~((1 << (x - 1))) & mask | m).count("0")  # placeholder, replaced below
                # cheaper proxy: count remaining elements that co-occur with x in a 4-AP
                conflict = 0
                for i in elem_to_4aps[x]:
                    m = four_ap_bitmasks[i]
                    # elements of this 4-AP other than x
                    others = m & ~(1 << (x - 1))
                    conflict += (others & mask).bit_count() if hasattr(int, 'bit_count') else bin(others & mask).count('1')
                score = g * w1 - conflict * w2
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x is None:
                break
            mask |= (1 << (best_x - 1))
            remaining.remove(best_x)
        return mask

    # ---- Simulated annealing with 1-move and periodic 2-swap ----
    def anneal(mask, T0, T_min, cooling, time_budget, do_2swap=True):
        cur_score = count_3aps(mask)
        best_mask = mask
        best_score = cur_score
        in_set = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
        T = T0
        it = 0
        while time.time() < time_budget and T > T_min:
            it += 1
            if do_2swap and (it % 60 == 0) and len(in_set) >= 2:
                # 2-swap: remove two lowest-loss, add two best-gain
                losses = {}
                for a in in_set:
                    ma = mask & ~(1 << (a - 1))
                    losses[a] = cur_score - count_3aps(ma)
                rem_sorted = sorted(in_set, key=lambda x: losses[x])
                a1, a2 = rem_sorted[0], rem_sorted[1]
                m2 = mask & ~(1 << (a1 - 1)) & ~(1 << (a2 - 1))
                cands = [x for x in range(1, N + 1) if not (m2 >> (x - 1)) & 1 and not creates_4ap(x, m2)]
                cands.sort(key=lambda x: -gain_3ap(x, m2))
                added = []
                mm = m2
                for x in cands[:2]:
                    if not creates_4ap(x, mm):
                        mm |= (1 << (x - 1))
                        added.append(x)
                if len(added) == 2:
                    ns = count_3aps(mm)
                    if ns >= cur_score or random.random() < math.exp((ns - cur_score) / max(T, 1e-9)):
                        mask = mm
                        in_set = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
                        cur_score = ns
                        if cur_score > best_score:
                            best_score = cur_score
                            best_mask = mask
            else:
                if random.random() < 0.4 and in_set:
                    x = random.choice(in_set)
                    nm = mask & ~(1 << (x - 1))
                    ns = count_3aps(nm)
                    if ns >= cur_score or random.random() < math.exp((ns - cur_score) / max(T, 1e-9)):
                        mask = nm
                        in_set.remove(x)
                        cur_score = ns
                        if cur_score > best_score:
                            best_score = cur_score
                            best_mask = mask
                else:
                    avail = [x for x in range(1, N + 1) if not (mask >> (x - 1)) & 1]
                    if avail:
                        x = random.choice(avail)
                        if not creates_4ap(x, mask):
                            nm = mask | (1 << (x - 1))
                            ns = count_3aps(nm)
                            if ns >= cur_score or random.random() < math.exp((ns - cur_score) / max(T, 1e-9)):
                                mask = nm
                                in_set.append(x)
                                cur_score = ns
                                if cur_score > best_score:
                                    best_score = cur_score
                                    best_mask = mask
            T *= cooling
        return best_mask, best_score

    # ---- Multi-restart driver ----
    global_best_mask = 0
    global_best_score = -1

    # Restart 1: conflict-aware greedy with tuned weights + long anneal
    m = greedy_conflict(w1=1.0, w2=0.06, seed=1, time_budget=deadline - 25.0)
    bm, bs = anneal(m, 80.0, 0.5, 0.9995, deadline - 15.0, do_2swap=True)
    if bs > global_best_score:
        global_best_score = bs
        global_best_mask = bm

    # Restart 2: lower conflict penalty (denser) + anneal
    if time.time() < deadline - 25.0:
        m = greedy_conflict(w1=1.0, w2=0.02, seed=2, time_budget=deadline - 20.0)
        bm, bs = anneal(m, 80.0, 0.5, 0.9995, deadline - 10.0, do_2swap=True)
        if bs > global_best_score:
            global_best_score = bs
            global_best_mask = bm

    # Final long anneal from the best found
    if time.time() < deadline - 3.0:
        bm, bs = anneal(global_best_mask, 60.0, 0.5, 0.9997, deadline - 2.0, do_2swap=True)
        if bs > global_best_score:
            global_best_score = bs
            global_best_mask = bm

    result = [x for x in range(1, N + 1) if (global_best_mask >> (x - 1)) & 1]
    if not result:
        result = [1]
    return sorted(result)
# EVOLVE-BLOCK-END
