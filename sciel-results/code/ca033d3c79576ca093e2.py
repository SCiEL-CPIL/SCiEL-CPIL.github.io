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
    # - three_ap_partners[y]: list of (x, role) where x and y are in a 3-AP
    #   role indicates which other element completes the AP
    # - four_ap_partners[y]: list of masks of the other 3 elements in each 4-AP containing y

    # For 3-APs: for element y, find all 3-APs containing y and the other two elements
    # We'll use a different approach: for each y, precompute the set of pairs {x,z} 
    # such that {x,y,z} forms a 3-AP. Then ΔE for adding y = -(# of such pairs both in mask)
    # For efficiency, for each y, store list of (x, z) pairs.
    
    three_ap_pairs = [[] for _ in range(n + 1)]  # three_ap_pairs[y] = list of (x, z) where {x,y,z} is a 3-AP
    for (a, b, c) in three_aps:
        # y=a: other two are b,c
        three_ap_pairs[a].append((b, c))
        # y=b: other two are a,c
        three_ap_pairs[b].append((a, c))
        # y=c: other two are a,b
        three_ap_pairs[c].append((a, b))

    # For 4-APs: for each y, store list of masks of the other 3 elements
    four_ap_others = [[] for _ in range(n + 1)]  # four_ap_others[y] = list of bitmasks
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

    # Energy: E = -3AP_count + penalty * 4AP_count
    # We use a hard constraint: if adding y creates a 4-AP, reject immediately.
    # So effectively we only work in the 4-AP-free space.
    # The "energy" we minimize is just -3AP_count.

    # Parallel tempering: run R replicas at different temperatures
    R = 8  # number of replicas
    # Temperatures: from hot to cold
    T_values = [20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2]
    
    # Initialize replicas with diverse starting points
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
    steps_per_swap = 50  # steps between replica swaps
    total_steps = 0
    max_total_steps = 80000  # total budget
    
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
                    # Check feasibility
                    mask_wo_x = mask & ~(1 << (x - 1))
                    if adds_4ap(y, mask_wo_x):
                        continue
                    # Compute gain
                    lx = 0
                    for (a, b) in three_ap_pairs[x]:
                        if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                            lx += 1
                    gy = 0
                    for (a, b) in three_ap_pairs[y]:
                        if (mask_wo_x >> (a - 1)) & 1 and (mask_wo_x >> (b - 1)) & 1:
                            gy += 1
                    g = gy - lx
                    new_mask = mask_wo_x | (1 << (y - 1))
                    new_score = score + g
                    delta = new_score - score
                    if delta >= 0 or random.random() < np.exp(delta / max(T, 1e-10)):
                        mask = new_mask
                        score = new_score
                
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
            # Acceptance probability for swapping
            # ΔE = (E1/T2 + E2/T1) - (E1/T1 + E2/T2) where E = -score
            # = -s1/T2 - s2/T1 + s1/T1 + s2/T2
            # = s1*(1/T1 - 1/T2) + s2*(1/T2 - 1/T1)
            # = (s1 - s2) * (1/T1 - 1/T2)
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
            if (mask >> (y - 1)) & 1:
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
        # Try swaps (sampled)
        elems = mask_to_list(mask)
        cands = [y for y in range(1, n + 1) if not (mask >> (y - 1)) & 1]
        if elems and cands:
            for _ in range(200):
                x = random.choice(elems)
                y = random.choice(cands)
                mask_wo_x = mask & ~(1 << (x - 1))
                if adds_4ap(y, mask_wo_x):
                    continue
                lx = 0
                for (a, b) in three_ap_pairs[x]:
                    if (mask >> (a - 1)) & 1 and (mask >> (b - 1)) & 1:
                        lx += 1
                gy = 0
                for (a, b) in three_ap_pairs[y]:
                    if (mask_wo_x >> (a - 1)) & 1 and (mask_wo_x >> (b - 1)) & 1:
                        gy += 1
                g = gy - lx
                if g > best_g:
                    best_g = g
                    best_move = ('swap', x, y)
        if best_move and best_g > 0:
            mt = best_move[0]
            if mt == 'add':
                mask |= (1 << (best_move[1] - 1))
            elif mt == 'remove':
                mask &= ~(1 << (best_move[2] - 1))
            else:
                mask &= ~(1 << (best_move[1] - 1))
                mask |= (1 << (best_move[2] - 1))
            score += best_g
            improved = True
            update_best(mask, score)

    # Final verification
    if best_mask and has_4ap(best_mask):
        mask = 0
        for x in range(1, n + 1):
            if not adds_4ap(x, mask):
                mask |= (1 << (x - 1))
        best_mask = mask
        best_score = count_3aps(mask)

    # Ensure nonempty
    if best_mask == 0:
        best_mask = 1
        best_score = 0

    return mask_to_list(best_mask)
