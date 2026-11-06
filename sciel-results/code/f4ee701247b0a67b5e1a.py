import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 82.0

    # ---------- Precomputation ----------
    # All 3-APs as (a,b,c)
    aps_3 = []
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            aps_3.append((a, a + d, a + 2 * d))
    m3 = len(aps_3)

    # For each element x, list of (ap_idx, pos) where x appears in ap
    ap_index_by_elem = [[] for _ in range(n + 1)]
    for idx, (a, b, c) in enumerate(aps_3):
        ap_index_by_elem[a].append((idx, 0))
        ap_index_by_elem[b].append((idx, 1))
        ap_index_by_elem[c].append((idx, 2))

    # All 4-APs
    four_aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for s in range(1, n - 3 * d + 1):
            four_aps.append((s, s + d, s + 2 * d, s + 3 * d))
    m4 = len(four_aps)

    # For each x, list of (ap_idx, pos) for 4-APs containing x
    four_aps_by_elem = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(four_aps):
        for pos, elem in enumerate(ap):
            four_aps_by_elem[elem].append((idx, pos))

    # Precompute mask4[x] = list of bitmasks of other 3 elements in each 4-AP containing x
    mask4 = [None] * (n + 1)
    for x in range(1, n + 1):
        lst = four_aps_by_elem[x]
        if lst:
            mask4[x] = []
            for ap_idx, pos in lst:
                ap = four_aps[ap_idx]
                m = 0
                for j in range(4):
                    if j != pos:
                        m |= (1 << (ap[j] - 1))
                mask4[x].append(m)
        else:
            mask4[x] = []

    # gain_pairs[x] = list of (bit1, bit2) where both must be in S for adding x to complete a 3-AP
    gain_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        pairs = []
        for idx, pos in ap_index_by_elem[x]:
            a, b, c = aps_3[idx]
            if pos == 0:
                pairs.append((1 << (b-1), 1 << (c-1)))
            elif pos == 1:
                pairs.append((1 << (a-1), 1 << (c-1)))
            else:
                pairs.append((1 << (a-1), 1 << (b-1)))
        gain_pairs[x] = pairs

    # ---------- Core operations on bitmask S (bit x-1 set if x in S) ----------
    def can_add(x, S):
        masks = mask4[x]
        for m in masks:
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
        S_wo_x = S & ~(1 << (x-1))
        g = 0
        for bit1, bit2 in gain_pairs[x]:
            if (S_wo_x & bit1) and (S_wo_x & bit2):
                g += 1
        return g

    def score(S):
        cnt = 0
        for a, b, c in aps_3:
            if (S & (1 << (a-1))) and (S & (1 << (b-1))) and (S & (1 << (c-1))):
                cnt += 1
        return cnt

    def is_valid(S):
        for ap in four_aps:
            if (S & (1 << (ap[0]-1))) and (S & (1 << (ap[1]-1))) and (S & (1 << (ap[2]-1))) and (S & (1 << (ap[3]-1))):
                return False
        return True

    # ---------- Local search with add, remove-then-add, swap, and 2-swap moves ----------
    def local_search(S, current_score, max_iters=50, time_budget=None):
        S = int(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            improved = False
            iters += 1

            # Phase 1: Greedily add elements with positive gain
            for x in range(1, n + 1):
                if (S >> (x-1)) & 1:
                    continue
                if not can_add(x, S):
                    continue
                g = gain_add(x, S)
                if g > 0:
                    S |= (1 << (x-1))
                    current_score += g
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 2: Remove-then-greedy-add (try each element in S)
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.5:
                    break
                if not ((S >> (y-1)) & 1):
                    continue
                S_trial = S & ~(1 << (y-1))
                score_trial = current_score - loss_remove(y, S)
                # Greedily add back
                for x in range(1, n + 1):
                    if (S_trial >> (x-1)) & 1:
                        continue
                    if not can_add(x, S_trial):
                        continue
                    g = gain_add(x, S_trial)
                    if g > 0:
                        S_trial |= (1 << (x-1))
                        score_trial += g
                if score_trial > current_score:
                    S = S_trial
                    current_score = score_trial
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 3: 1-swap moves
            for y in range(1, n + 1):
                if time_budget is not None and time.time() > time_budget - 0.5:
                    break
                if not ((S >> (y-1)) & 1):
                    continue
                S_wo_y = S & ~(1 << (y-1))
                score_wo_y = current_score - loss_remove(y, S)
                best_x = -1
                best_gain = 0
                for x in range(1, n + 1):
                    if (S_wo_y >> (x-1)) & 1:
                        continue
                    if not can_add(x, S_wo_y):
                        continue
                    g = gain_add(x, S_wo_y)
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                if best_x > 0 and score_wo_y + best_gain > current_score:
                    S = S_wo_y | (1 << (best_x - 1))
                    current_score = score_wo_y + best_gain
                    improved = True
                    if time_budget is not None and time.time() > time_budget:
                        break

            if time_budget is not None and time.time() > time_budget:
                break

            # Phase 4: 2-swap moves (remove y1, y2; add x1, x2)
            # Sample a few pairs to avoid O(N^4) complexity
            in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
            if len(in_set) >= 2:
                # Randomly sample 10 pairs
                for _ in range(10):
                    if time_budget is not None and time.time() > time_budget - 0.5:
                        break
                    y1, y2 = random.sample(in_set, 2)
                    S_wo_y1y2 = S & ~(1 << (y1-1)) & ~(1 << (y2-1))
                    score_wo_y1y2 = current_score - loss_remove(y1, S) - loss_remove(y2, S & ~(1 << (y1-1)))
                    
                    best_x1 = -1
                    best_x2 = -1
                    best_gain = 0
                    
                    # Find best x1
                    for x1 in range(1, n + 1):
                        if (S_wo_y1y2 >> (x1-1)) & 1:
                            continue
                        if not can_add(x1, S_wo_y1y2):
                            continue
                        g1 = gain_add(x1, S_wo_y1y2)
                        if g1 > best_gain:
                            best_gain = g1
                            best_x1 = x1
                            best_x2 = -1 # Reset x2 if x1 changes
                            
                            # Find best x2 for this x1
                            S_trial = S_wo_y1y2 | (1 << (x1-1))
                            max_g2 = 0
                            for x2 in range(1, n + 1):
                                if (S_trial >> (x2-1)) & 1:
                                    continue
                                if not can_add(x2, S_trial):
                                    continue
                                g2 = gain_add(x2, S_trial)
                                if g2 > max_g2:
                                    max_g2 = g2
                                    best_x2 = x2
                            best_gain = g1 + max_g2
                            
                    if best_x1 > 0 and best_x2 > 0 and score_wo_y1y2 + best_gain > current_score:
                        S = S_wo_y1y2 | (1 << (best_x1 - 1)) | (1 << (best_x2 - 1))
                        current_score = score_wo_y1y2 + best_gain
                        improved = True
                        in_set = [x for x in range(1, n + 1) if (S >> (x-1)) & 1]
                        if time_budget is not None and time.time() > time_budget:
                            break

        return S, current_score

    # ---------- Simulated Annealing local search ----------
    def sa_local_search(S, current_score, max_iters=200, time_budget=None):
        S = int(S)
        T = 5.0
        T_min = 0.1
        cooling = 0.97
        iters = 0
        best_S = S
        best_score = current_score
        while iters < max_iters:
            if time_budget is not None and time.time() > time_budget:
                break
            iters += 1
            if T < T_min:
                break

            # Pick a random move
            move_type = random.random()
            if move_type < 0.4:
                # Add
                candidates = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1) and can_add(x, S)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                S_new = S | (1 << (x-1))
                delta = gain_add(x, S)
            elif move_type < 0.7:
                # Remove
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                if not in_set:
                    continue
                y = random.choice(in_set)
                S_new = S & ~(1 << (y-1))
                delta = -loss_remove(y, S)
            else:
                # Swap
                in_set = [x for x in range(1, n+1) if (S >> (x-1)) & 1]
                not_in_set = [x for x in range(1, n+1) if not ((S >> (x-1)) & 1)]
                if not in_set or not not_in_set:
                    continue
                y = random.choice(in_set)
                S_wo_y = S & ~(1 << (y-1))
                valid_adds = [x for x in not_in_set if can_add(x, S_wo_y)]
                if not valid_adds:
                    continue
                x = random.choice(valid_adds)
                S_new = S_wo_y | (1 << (x-1))
                delta = -loss_remove(y, S) + gain_add(x, S_wo_y)

            # Accept/reject
            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = S
            T *= cooling

        return best_S, best_score

    # ---------- Construction strategies ----------
    def greedy_random(order, use_score=True):
        S = 0
        for x in order:
            if not can_add(x, S):
                continue
            if use_score:
                g = gain_add(x, S)
                if g > 0 or bin(S).count('1') < 20:
                    S |= (1 << (x-1))
            else:
                S |= (1 << (x-1))
        return S

    def prune_from_full():
        S = (1 << n) - 1
        for ap in four_aps:
            if all((S >> (elem-1)) & 1 for elem in ap):
                min_loss = float('inf')
                min_elem = ap[0]
                for elem in ap:
                    l = loss_remove(elem, S)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << (min_elem - 1))
        return S

    def density_heuristic():
        # Sort elements by number of 3-APs they participate in
        # Elements in more APs are "denser"
        ap_count = [0] * (n + 1)
        for a, b, c in aps_3:
            ap_count[a] += 1
            ap_count[b] += 1
            ap_count[c] += 1
        
        order = list(range(1, n + 1))
        order.sort(key=lambda x: ap_count[x], reverse=True)
        return greedy_random(order, use_score=True)

    # ---------- Main search ----------
    best_S = 0
    best_score = 0

    # Strategy 1: Natural order greedy
    if time.time() < deadline - 20:
        S1 = greedy_random(range(1, n + 1), use_score=False)
        sc1 = score(S1)
        S1, sc1 = local_search(S1, sc1, max_iters=20, time_budget=deadline - 10)
        if sc1 > best_score:
            best_score = sc1
            best_S = S1

    # Strategy 2: Density heuristic
    if time.time() < deadline - 20:
        S_d = density_heuristic()
        sc_d = score(S_d)
        S_d, sc_d = local_search(S_d, sc_d, max_iters=20, time_budget=deadline - 10)
        if sc_d > best_score:
            best_score = sc_d
            best_S = S_d

    # Strategy 3: Many random restarts with local search
    random.seed(12345)
    n_restarts = 40
    for r in range(n_restarts):
        if time.time() > deadline - 25:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True)
        sc_r = score(S_r)
        S_r, sc_r = local_search(S_r, sc_r, max_iters=15, time_budget=deadline - 15)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Strategy 4: Prune from full
    if time.time() < deadline - 20:
        S_p = prune_from_full()
        sc_p = score(S_p)
        S_p, sc_p = local_search(S_p, sc_p, max_iters=25, time_budget=deadline - 10)
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Strategy 5: More random restarts with SA
    for r in range(30):
        if time.time() > deadline - 15:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r = greedy_random(order, use_score=True)
        sc_r = score(S_r)
        S_r, sc_r = sa_local_search(S_r, sc_r, max_iters=150, time_budget=deadline - 8)
        if sc_r > best_score:
            best_score = sc_r
            best_S = S_r

    # Final local search on best
    if time.time() < deadline - 2 and best_S:
        best_S, best_score = local_search(best_S, best_score, max_iters=30, time_budget=deadline - 1)

    # Final SA polish
    if time.time() < deadline - 1 and best_S:
        best_S, best_score = sa_local_search(best_S, best_score, max_iters=100, time_budget=deadline - 0.5)

    # Convert bitmask to sorted list
    result = [x for x in range(1, n + 1) if (best_S >> (x-1)) & 1]
    if not result:
        result = [1]
    return result
