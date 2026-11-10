import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    # Precompute all 3-APs and 4-APs
    three_aps = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))

    four_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each element y, precompute the pairs (x, z) such that {x,y,z} is a 3-AP
    three_ap_pairs = [[] for _ in range(n + 1)]
    for (a, b, c) in three_aps:
        three_ap_pairs[a].append((b, c))
        three_ap_pairs[b].append((a, c))
        three_ap_pairs[c].append((a, b))

    # For 4-APs: for each y, store list of bitmasks of the other 3 elements
    four_ap_others = [[] for _ in range(n + 1)]
    for (a, b, c, dd) in four_aps:
        for y in (a, b, c, dd):
            others = [z for z in (a, b, c, dd) if z != y]
            m = 0
            for z in others:
                m |= (1 << (z - 1))
            four_ap_others[y].append(m)

    # Precompute bit for each element
    bits = [0] + [1 << (i - 1) for i in range(1, n + 1)]

    def count_3aps(mask):
        count = 0
        for a, b, c in three_aps:
            if (mask & bits[a]) and (mask & bits[b]) and (mask & bits[c]):
                count += 1
        return count

    def has_4ap(mask):
        for a, b, c, dd in four_aps:
            if (mask & bits[a]) and (mask & bits[b]) and (mask & bits[c]) and (mask & bits[dd]):
                return True
        return False

    def mask_to_list(mask):
        return [i + 1 for i in range(n) if (mask >> i) & 1]

    def adds_4ap(y, mask):
        bit = bits[y]
        if mask & bit:
            return False
        for om in four_ap_others[y]:
            if (mask & om) == om:
                return True
        return False

    def gain_add(y, mask):
        bit = bits[y]
        if mask & bit:
            return 0
        cnt = 0
        for (x, z) in three_ap_pairs[y]:
            if (mask & bits[x]) and (mask & bits[z]):
                cnt += 1
        return cnt

    def loss_rem(y, mask):
        bit = bits[y]
        if not (mask & bit):
            return 0
        cnt = 0
        for (x, z) in three_ap_pairs[y]:
            if (mask & bits[x]) and (mask & bits[z]):
                cnt += 1
        return cnt

    def swap_gain(x, y, mask):
        """Gain from removing x and adding y (x in mask, y not in mask)."""
        mask_wo_x = mask & ~bits[x]
        # Loss from removing x
        lx = 0
        for (a, b) in three_ap_pairs[x]:
            if (mask & bits[a]) and (mask & bits[b]):
                lx += 1
        # Gain from adding y to mask_wo_x
        gy = 0
        for (a, b) in three_ap_pairs[y]:
            if (mask_wo_x & bits[a]) and (mask_wo_x & bits[b]):
                gy += 1
        return gy - lx

    # Energy: E = -3AP_count + penalty * 4AP_count
    # We use a hard constraint: if adding y creates a 4-AP, reject immediately.
    # So effectively we only work in the 4-AP-free space.
    # The "energy" we minimize is just -3AP_count.

    # Parallel tempering: run R replicas at different temperatures
    R = 10  # number of replicas
    # Temperatures: from hot to cold
    T_values = [25.0, 15.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    
    # Greedy by gain: repeatedly add the element with highest gain that doesn't create 4-AP
    def greedy_by_gain(seed_mask=0, seed_score=0):
        mask = seed_mask
        score = seed_score
        while True:
            best_g = 0
            best_y = None
            for y in range(1, n + 1):
                if mask & bits[y]:
                    continue
                if adds_4ap(y, mask):
                    continue
                g = gain_add(y, mask)
                if g > best_g:
                    best_g = g
                    best_y = y
            if best_y is None:
                break
            mask |= bits[best_y]
            score += best_g
        return mask, score

    # Initialize replicas with diverse starting points
    def init_replica(rep_idx):
        """Initialize replica with a diverse starting set."""
        if rep_idx == 0:
            # Greedy by gain (best first)
            mask, score = greedy_by_gain()
        elif rep_idx == 1:
            # Reverse greedy
            mask = 0
            for x in range(n, 0, -1):
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            score = count_3aps(mask)
        elif rep_idx == 2:
            # Odds first, then greedy polish
            mask = 0
            for x in list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)):
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            mask, score = greedy_by_gain(mask, count_3aps(mask))
        elif rep_idx == 3:
            # Evens first, then greedy polish
            mask = 0
            for x in list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)):
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            mask, score = greedy_by_gain(mask, count_3aps(mask))
        elif rep_idx == 4:
            # Multiples of 3 first
            mask = 0
            for x in list(range(3, n + 1, 3)) + [x for x in range(1, n + 1) if x % 3 != 0]:
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            mask, score = greedy_by_gain(mask, count_3aps(mask))
        elif rep_idx == 5:
            # Powers of 2 neighborhood first
            mask = 0
            order = sorted(range(1, n + 1), key=lambda x: (abs(x - 128), x))
            for x in order:
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            mask, score = greedy_by_gain(mask, count_3aps(mask))
        else:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
            mask = 0
            for x in order:
                if not adds_4ap(x, mask):
                    mask |= bits[x]
            mask, score = greedy_by_gain(mask, count_3aps(mask))
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
    steps_per_swap = 30  # steps between replica swaps
    total_steps = 0
    max_total_steps = 120000  # total budget
    
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
                if move_type < 0.30:
                    # Add
                    cands = [y for y in range(1, n + 1) if not (mask & bits[y])]
                    if not cands:
                        continue
                    y = random.choice(cands)
                    if adds_4ap(y, mask):
                        continue
                    g = gain_add(y, mask)
                    new_mask = mask | bits[y]
                    delta = g
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score += delta
                elif move_type < 0.50:
                    # Remove
                    elems = mask_to_list(mask)
                    if not elems:
                        continue
                    x = random.choice(elems)
                    g = -loss_rem(x, mask)
                    new_mask = mask & ~bits[x]
                    delta = g
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score += delta
                elif move_type < 0.75:
                    # Swap
                    elems = mask_to_list(mask)
                    cands = [y for y in range(1, n + 1) if not (mask & bits[y])]
                    if not elems or not cands:
                        continue
                    x = random.choice(elems)
                    y = random.choice(cands)
                    # Check feasibility
                    mask_wo_x = mask & ~bits[x]
                    if adds_4ap(y, mask_wo_x):
                        continue
                    g = swap_gain(x, y, mask)
                    new_mask = mask_wo_x | bits[y]
                    delta = g
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score += delta
                else:
                    # 2-swap: remove two, add two (or remove one, add two)
                    elems = mask_to_list(mask)
                    cands = [y for y in range(1, n + 1) if not (mask & bits[y])]
                    if len(elems) >= 2 and len(cands) >= 2:
                        x1, x2 = random.sample(elems, 2)
                        y1, y2 = random.sample(cands, 2)
                        mask_wo = mask & ~bits[x1] & ~bits[x2]
                        if not adds_4ap(y1, mask_wo) and not adds_4ap(y2, mask_wo):
                            g1 = 0
                            for (a, b) in three_ap_pairs[x1]:
                                if (mask & bits[a]) and (mask & bits[b]):
                                    g1 += 1
                            g2 = 0
                            for (a, b) in three_ap_pairs[x2]:
                                if (mask & bits[a]) and (mask & bits[b]):
                                    g2 += 1
                            # Subtract overlap: pairs containing both x1 and x2
                            overlap = 0
                            for (a, b) in three_ap_pairs[x1]:
                                if a == x2 or b == x2:
                                    if (mask & bits[a]) and (mask & bits[b]):
                                        overlap += 1
                            gy1 = 0
                            for (a, b) in three_ap_pairs[y1]:
                                if (mask_wo & bits[a]) and (mask_wo & bits[b]):
                                    gy1 += 1
                            gy2 = 0
                            for (a, b) in three_ap_pairs[y2]:
                                if (mask_wo & bits[a]) and (mask_wo & bits[b]):
                                    gy2 += 1
                            # Subtract overlap for y1,y2
                            gy_overlap = 0
                            for (a, b) in three_ap_pairs[y1]:
                                if a == y2 or b == y2:
                                    if (mask_wo & bits[a]) and (mask_wo & bits[b]):
                                        gy_overlap += 1
                            delta = (gy1 + gy2 - gy_overlap) - (g1 + g2 - overlap)
                            if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                                mask = mask_wo | bits[y1] | bits[y2]
                                score += delta
                
                total_steps += 1
            
            replicas[r]['mask'] = mask
            replicas[r]['score'] = score
            update_best(mask, score)
        
        # Swap adjacent replicas (Metropolis acceptance)
        for r in range(R - 1):
            if time.time() > deadline - 5:
                break
            m1, s1, T1 = replicas[r]['mask'], replicas[r]['score'], replicas[r]['T']
            m2, s2, T2 = replicas[r + 1]['mask'], replicas[r + 1]['score'], replicas[r + 1]['T']
            if T1 == T2:
                accept = True
            else:
                delta_E = (s1 - s2) * (1.0 / T1 - 1.0 / T2)
                if delta_E <= 0 or random.random() < np.exp(-delta_E):
                    accept = True
                else:
                    accept = False
            if accept:
                replicas[r]['mask'], replicas[r + 1]['mask'] = m2, m1
                replicas[r]['score'], replicas[r + 1]['score'] = s2, s1
                update_best(replicas[r]['mask'], replicas[r]['score'])
                update_best(replicas[r + 1]['mask'], replicas[r + 1]['score'])
    
    # Final hill-climbing polish on the best replica
    mask = best_mask
    score = best_score
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        best_g = 0
        best_move = None
        # Try all adds
        for y in range(1, n + 1):
            if mask & bits[y]:
                continue
            if adds_4ap(y, mask):
                continue
            g = gain_add(y, mask)
            if g > best_g:
                best_g = g
                best_move = ('add', y, None)
        # Try all removes
        for x in mask_to_list(mask):
            g = -loss_rem(x, mask)
            if g > best_g:
                best_g = g
                best_move = ('remove', None, x)
        # Try all swaps
        elems = mask_to_list(mask)
        cands = [y for y in range(1, n + 1) if not (mask & bits[y])]
        if elems and cands:
            for x in elems:
                for y in cands:
                    mask_wo_x = mask & ~bits[x]
                    if adds_4ap(y, mask_wo_x):
                        continue
                    g = swap_gain(x, y, mask)
                    if g > best_g:
                        best_g = g
                        best_move = ('swap', x, y)
        if best_move and best_g > 0:
            mt = best_move[0]
            if mt == 'add':
                mask |= bits[best_move[1]]
            elif mt == 'remove':
                mask &= ~bits[best_move[2]]
            else:
                mask &= ~bits[best_move[1]]
                mask |= bits[best_move[2]]
            score += best_g
            improved = True
            update_best(mask, score)

    # Second polish pass: try 2-adds and 2-removes
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            best_g = 0
            best_move = None
            elems = mask_to_list(mask)
            cands = [y for y in range(1, n + 1) if not (mask & bits[y])]
            # Try 2-adds
            if len(cands) >= 2:
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        y1, y2 = cands[i], cands[j]
                        if adds_4ap(y1, mask) or adds_4ap(y2, mask):
                            continue
                        if adds_4ap(y2, mask | bits[y1]):
                            continue
                        g1 = gain_add(y1, mask)
                        g2 = gain_add(y2, mask | bits[y1])
                        g = g1 + g2
                        if g > best_g:
                            best_g = g
                            best_move = ('2add', y1, y2)
            # Try 2-removes
            if len(elems) >= 2:
                for i in range(len(elems)):
                    for j in range(i + 1, len(elems)):
                        x1, x2 = elems[i], elems[j]
                        mask_wo = mask & ~bits[x1] & ~bits[x2]
                        g1 = -loss_rem(x1, mask)
                        g2 = -loss_rem(x2, mask_wo)
                        g = g1 + g2
                        if g > best_g:
                            best_g = g
                            best_move = ('2rem', x1, x2)
            if best_move and best_g > 0:
                mt = best_move[0]
                if mt == '2add':
                    mask |= bits[best_move[1]] | bits[best_move[2]]
                else:
                    mask &= ~bits[best_move[1]] & ~bits[best_move[2]]
                score += best_g
                improved = True
                update_best(mask, score)

    # Final verification
    if best_mask and has_4ap(best_mask):
        mask = 0
        for x in range(1, n + 1):
            if not adds_4ap(x, mask):
                mask |= bits[x]
        best_mask = mask
        best_score = count_3aps(mask)

    # Ensure nonempty
    if best_mask == 0:
        best_mask = 1
        best_score = 0

    return mask_to_list(best_mask)
