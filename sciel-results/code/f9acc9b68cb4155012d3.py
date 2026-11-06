"""QUBO-inspired parallel tempering with incremental scoring for 4-AP-free set maximizing 3-AP count."""
import time
import random
import math

N = 256
TIME_LIMIT = 84.0  # safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))
    
    # For each element, store the OTHER elements in each 4-AP it participates in
    elem_to_4aps = [[] for _ in range(N)]
    for ap in four_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(4) if j != i)
            elem_to_4aps[x].append(others)
    
    # For each element, store the OTHER elements in each 3-AP it participates in
    elem_to_3aps = [[] for _ in range(N)]
    for ap in three_aps:
        for i, x in enumerate(ap):
            others = tuple(ap[j] for j in range(3) if j != i)
            elem_to_3aps[x].append(others)
    
    # Precompute 3-AP weights (total number of 3-APs each element is in)
    elem_3ap_weight = [len(elem_to_3aps[x]) for x in range(N)]
    # Precompute 4-AP weights (total number of 4-APs each element is in)
    elem_4ap_weight = [len(elem_to_4aps[x]) for x in range(N)]
    
    def is_safe_add(S, x):
        """Check if adding x to S (bitmask) creates any 4-AP involving x."""
        for others in elem_to_4aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                return False
        return True
    
    def gain_3aps(S, x):
        """Number of new 3-APs formed by adding x to S."""
        gain = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                gain += 1
        return gain
    
    def loss_3aps(S, x):
        """Number of 3-APs lost by removing x from S (x must be in S)."""
        loss = 0
        for others in elem_to_3aps[x]:
            ok = True
            for y in others:
                if not ((S >> y) & 1):
                    ok = False
                    break
            if ok:
                loss += 1
        return loss
    
    def count_3aps_bitmask(S):
        """Count 3-APs in bitmask S (0-indexed)."""
        cnt = 0
        for a, b, c in three_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt
    
    def count_4aps_bitmask(S):
        """Count 4-APs fully contained in bitmask S."""
        cnt = 0
        for a, b, c, d in four_aps:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                cnt += 1
        return cnt
    
    def repair_4aps(S):
        """Remove elements to eliminate all 4-APs, prioritizing low 3-AP weight."""
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                    # Remove the element with lowest 3-AP weight (tie: highest 4-AP weight)
                    elems = (a, b, c, d)
                    worst_idx = min(range(4), key=lambda i: (elem_3ap_weight[elems[i]], -elem_4ap_weight[elems[i]]))
                    worst = elems[worst_idx]
                    S &= ~(1 << worst)
                    changed = True
                    break
        return S
    
    def greedy_construct(order):
        """Greedy: add elements in given order, skip if creates 4-AP."""
        S = 0
        for x in order:
            if is_safe_add(S, x):
                S |= (1 << x)
        return S
    
    def local_search_2opt(S, max_iters=300):
        """2-opt local search: try all (remove x, add y) pairs to improve 3-AP count.
        Maintains no 4-APs constraint. Uses incremental scoring."""
        current_S = S
        current_score = count_3aps_bitmask(current_S)
        
        for it in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            improved = False
            
            # Get members and non-members
            members = [x for x in range(N) if (current_S >> x) & 1]
            non_members = [x for x in range(N) if not ((current_S >> x) & 1)]
            
            # Try single adds
            for x in non_members:
                if not is_safe_add(current_S, x):
                    continue
                new_score = current_score + gain_3aps(current_S, x)
                if new_score > current_score:
                    current_S |= (1 << x)
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try single removes (only if it helps - rare, but check)
            for x in members:
                new_score = current_score - loss_3aps(current_S, x)
                if new_score > current_score:
                    current_S &= ~(1 << x)
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove x, add y
            best_swap = None
            best_gain = 0
            for x_remove in members:
                S_after_remove = current_S & ~(1 << x_remove)
                loss = loss_3aps(current_S, x_remove)
                for x_add in non_members:
                    if not is_safe_add(S_after_remove, x_add):
                        continue
                    gain = gain_3aps(S_after_remove, x_add)
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_remove, x_add)
            
            if best_swap is not None and best_gain > 0:
                x_remove, x_add = best_swap
                current_S = (current_S & ~(1 << x_remove)) | (1 << x_add)
                current_score += best_gain
                improved = True
                continue
            
            # No improvement possible
            break
        
        return current_S, current_score
    
    # Phase 1: Greedy construction with multiple orderings
    best_mask = 0
    best_score = -1
    
    # Ordering 1: by decreasing 3-AP weight
    order1 = sorted(range(N), key=lambda x: (-elem_3ap_weight[x], x))
    mask1 = greedy_construct(order1)
    score1 = count_3aps_bitmask(mask1)
    if score1 > best_score:
        best_score = score1
        best_mask = mask1
    
    # Ordering 2: by increasing 4-AP weight (fewer conflicts first)
    order2 = sorted(range(N), key=lambda x: (elem_4ap_weight[x], x))
    mask2 = greedy_construct(order2)
    score2 = count_3aps_bitmask(mask2)
    if score2 > best_score:
        best_score = score2
        best_mask = mask2
    
    # Ordering 3: random
    order3 = list(range(N))
    rng.shuffle(order3)
    mask3 = greedy_construct(order3)
    score3 = count_3aps_bitmask(mask3)
    if score3 > best_score:
        best_score = score3
        best_mask = mask3
    
    # Phase 2: Local search on best greedy solution
    best_mask, best_score = local_search_2opt(best_mask, max_iters=200)
    
    # Phase 3: Parallel Tempering (Replica Exchange) with quantum-inspired tunneling
    # Multiple replicas at different temperatures, periodically exchange states
    num_replicas = 5
    temps = [10.0, 5.0, 2.0, 1.0, 0.5]
    replicas = [best_mask] * num_replicas
    replica_scores = [best_score] * num_replicas
    
    # Track global best
    global_best_mask = best_mask
    global_best_score = best_score
    
    max_pt_iters = 3000
    pt_iter = 0
    
    while pt_iter < max_pt_iters:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        pt_iter += 1
        
        # For each replica, perform a move
        for r in range(num_replicas):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            S = replicas[r]
            T = temps[r]
            
            # Generate a move: add, remove, or swap
            move_type = rng.random()
            new_S = S
            
            if move_type < 0.4:
                # Add an element
                x = rng.randint(0, N - 1)
                if not ((S >> x) & 1):
                    new_S |= (1 << x)
            elif move_type < 0.7:
                # Remove an element
                if S:
                    sel_list = [x for x in range(N) if (S >> x) & 1]
                    if sel_list:
                        x = rng.choice(sel_list)
                        new_S &= ~(1 << x)
            else:
                # Swap: remove one, add another
                if S:
                    sel_list = [x for x in range(N) if (S >> x) & 1]
                    if sel_list:
                        x_remove = rng.choice(sel_list)
                        x_add = rng.randint(0, N - 1)
                        if not ((S >> x_add) & 1):
                            new_S = (S & ~(1 << x_remove)) | (1 << x_add)
            
            # Compute energy change (QUBO-style: maximize 3-APs, penalize 4-APs)
            # E = -3ap_count + penalty * 4ap_count
            # Delta E = -delta_3ap + penalty * delta_4ap
            old_3ap = count_3aps_bitmask(S)
            new_3ap = count_3aps_bitmask(new_S)
            old_4ap = count_4aps_bitmask(S)
            new_4ap = count_4aps_bitmask(new_S)
            
            delta_3ap = new_3ap - old_3ap
            delta_4ap = new_4ap - old_4ap
            
            # Energy: lower is better
            # E = -3ap + 100*4ap
            delta_E = -delta_3ap + 100 * delta_4ap
            
            # Accept if delta_E <= 0 or with probability exp(-delta_E / T)
            if delta_E <= 0 or rng.random() < math.exp(-delta_E / T):
                replicas[r] = new_S
                replica_scores[r] = new_3ap
            
            # Track global best feasible solution
            if count_4aps_bitmask(replicas[r]) == 0:
                if replica_scores[r] > global_best_score:
                    global_best_score = replica_scores[r]
                    global_best_mask = replicas[r]
        
        # Periodic replica exchange: swap states between adjacent replicas
        # based on their energies (accept with probability exp(-(E_i - E_j)/T_avg))
        if pt_iter % 20 == 0:
            for r in range(num_replicas - 1):
                T_avg = (temps[r] + temps[r+1]) / 2.0
                E_i = -replica_scores[r] + 100 * count_4aps_bitmask(replicas[r])
                E_j = -replica_scores[r+1] + 100 * count_4aps_bitmask(replicas[r+1])
                delta_E = E_i - E_j
                if delta_E <= 0 or rng.random() < math.exp(-delta_E / T_avg):
                    # Swap
                    replicas[r], replicas[r+1] = replicas[r+1], replicas[r]
                    replica_scores[r], replica_scores[r+1] = replica_scores[r+1], replica_scores[r]
        
        # Periodic repair: fix 4-AP violations in all replicas
        if pt_iter % 50 == 0:
            for r in range(num_replicas):
                replicas[r] = repair_4aps(replicas[r])
                replica_scores[r] = count_3aps_bitmask(replicas[r])
                if count_4aps_bitmask(replicas[r]) == 0 and replica_scores[r] > global_best_score:
                    global_best_score = replica_scores[r]
                    global_best_mask = replicas[r]
    
    # Phase 4: Final local search on global best
    if time.time() - start_time < TIME_LIMIT - 3:
        global_best_mask, global_best_score = local_search_2opt(global_best_mask, max_iters=200)
    
    # Verify no 4-APs
    assert count_4aps_bitmask(global_best_mask) == 0, "4-AP violation found!"
    
    # Convert mask to list of 1-based indices
    result = [x + 1 for x in range(N) if (global_best_mask >> x) & 1]
    return result
