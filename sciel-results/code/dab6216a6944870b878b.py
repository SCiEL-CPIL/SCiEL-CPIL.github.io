import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute all 3-APs and 4-APs
    three_aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element y, precompute:
    # - three_ap_pairs[y]: list of (x, z) where {x,y,z} is a 3-AP
    # - four_ap_others[y]: list of bitmasks of the other 3 elements in each 4-AP containing y

    three_ap_pairs = [[] for _ in range(n + 1)]
    for (a, b, c) in three_aps:
        three_ap_pairs[a].append((b, c))
        three_ap_pairs[b].append((a, c))
        three_ap_pairs[c].append((a, b))

    four_ap_others = [[] for _ in range(n + 1)]
    for (a, b, c, dd) in four_aps:
        for y in (a, b, c, dd):
            others = [z for z in (a, b, c, dd) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_others[y].append(m)

    def count_3aps(mask):
        count = 0
        for a, b, c in three_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1:
                count += 1
        return count

    def has_4ap(mask):
        for a, b, c, dd in four_aps:
            if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1 and (mask >> (c - 1)) & 1 and (mask >> (dd - 1)) & 1:
                return True
        return False

    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    def adds_4ap(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return False
        for om in four_ap_others[y]:
            if (mask & om) == om:
                return True
        return False

    def gain_add(y, mask):
        bit = 1 << (y - 1)
        if mask & bit:
            return 0
        cnt = 0
        for (x, z) in three_ap_pairs[y]:
            if (mask >> (x - 1)) & 1 and (mask >> (z - 1)) & 1:
                cnt += 1
        return cnt

    def loss_rem(y, mask):
        bit = 1 << (y - 1)
        if not (mask & bit):
            return 0
        cnt = 0
        for (x, z) in three_ap_pairs[y]:
            if (mask >> (x - 1)) & 1 and (mask >> (z - 1)) & 1:
                cnt += 1
        return cnt

    # Parallel tempering: run R replicas at different temperatures
    R = 8
    T_values = [20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2]

    def init_replica(rep_idx):
        """Initialize replica with a diverse starting set."""
        mask = 0
        if rep_idx == 0:
            # Densest greedy
            for x in range(1, n + 1):
                if not adds_4ap(x, mask):
                    g = gain_add(x, mask)
                    if g >= 0:
                        mask |= (1 << (x - 1))
        elif rep_idx == 1:
            # Reverse greedy
            for x in range(n, 0, -1):
                if not adds_4ap(x, mask):
                    mask |= (1 << (x - 1))
        elif rep_idx == 2:
            # Odds first
            for x in list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)):
                if not adds_4ap(x, mask):
                    mask |= (1 << (x - 1))
        elif rep_idx == 3:
            # Evens first
            for x in list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)):
                if not adds_4ap(x, mask):
                    mask |= (1 << (x - 1))
        else:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
            for x in order:
                if not adds_4ap(x, mask):
                    mask |= (1 << (x - 1))
        score = count_3aps(mask)
        return mask, score

    replicas = []
    for r in range(R):
        mask, score = init_replica(r)
        replicas.append({'mask': mask, 'score': score, 'T': T_values[r]})

    best_score = max(rep['score'] for rep in replicas)
    best_mask = max(replicas, key=lambda r: r['score'])['mask']

    def update_best(mask, score):
        nonlocal best_score, best_mask
        if score > best_score:
            best_score = score
            best_mask = mask

    # Parallel tempering loop
    steps_per_swap = 50
    total_steps = 0
    max_total_steps = 80000

    while total_steps < max_total_steps:
        if time.time() > deadline - 5:
            break

        for r in range(R):
            if time.time() > deadline - 5:
                break

            mask = replicas[r]['mask']
            score = replicas[r]['score']
            T = replicas[r]['T']

            for step in range(steps_per_swap):
                if total_steps >= max_total_steps or time.time() > deadline - 5:
                    break

                move_type = random.random()
                if move_type < 0.35:
                    # Add
                    cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                    if not cands:
                        continue
                    y = random.choice(cands)
                    if adds_4ap(y, mask):
                        continue
                    g = gain_add(y, mask)
                    new_mask = mask | (1 << (y - 1))
                    new_score = score + g
                    delta = new_score - score
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score = new_score
                elif move_type < 0.55:
                    # Remove
                    elems = mask_to_list(mask)
                    if not elems:
                        continue
                    x = random.choice(elems)
                    g = -loss_rem(x, mask)
                    new_mask = mask & ~(1 << (x - 1))
                    new_score = score + g
                    delta = new_score - score
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score = new_score
                else:
                    # Swap
                    elems = mask_to_list(mask)
                    cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
                    if not elems or not cands:
                        continue
                    x = random.choice(elems)
                    y = random.choice(cands)
                    tm = mask & ~(1 << (x - 1))
                    if adds_4ap(y, tm):
                        continue
                    lx = loss_rem(x, mask)
                    gy = gain_add(y, tm)
                    # Subtract shared 3-APs containing both x and y
                    shared = 0
                    for idx, (a, b, c) in enumerate(three_aps):
                        if x in (a, b, c) and y in (a, b, c):
                            oth = [z for z in (a, b, c) if z not in (x, y)]
                            if all((mask >> (z - 1)) & 1 for z in oth):
                                shared += 1
                    g = -lx + (gy - shared)
                    new_mask = tm | (1 << (y - 1))
                    new_score = score + g
                    delta = new_score - score
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score = new_score

                total_steps += 1

            replicas[r]['mask'] = mask
            replicas[r]['score'] = score
            update_best(mask, score)

        # Swap adjacent replicas with Metropolis acceptance
        for r in range(R - 1):
            if time.time() > deadline - 5:
                break
            i, j = r, r + 1
            T_i, T_j = replicas[i]['T'], replicas[j]['T']
            S_i, S_j = replicas[i]['score'], replicas[j]['score']
            # Delta energy for swap
            delta_E = (S_i - S_j) * (1 / max(T_j, 1e-10) - 1 / max(T_i, 1e-10))
            if delta_E <= 0 or random.random() < np.exp(-delta_E):
                # Swap masks and scores
                replicas[i]['mask'], replicas[j]['mask'] = replicas[j]['mask'], replicas[i]['mask']
                replicas[i]['score'], replicas[j]['score'] = replicas[j]['score'], replicas[i]['score']

    # Final verification
    if best_mask and has_4ap(best_mask):
        S = set()
        for x in range(1, n + 1):
            ok = True
            for d in range(1, (x - 1) // 3 + 1):
                if all((x - j * d) in S for j in (1, 2, 3)):
                    ok = False
                    break
            if ok:
                S.add(x)
        best_mask = 0
        for x in S:
            best_mask |= (1 << (x - 1))
        best_score = count_3aps(best_mask)

    return mask_to_list(best_mask)
