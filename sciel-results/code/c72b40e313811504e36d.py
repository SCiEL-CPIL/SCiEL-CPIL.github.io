import math
import time

def construct():
    N = 256
    t0 = time.time()
    deadline = t0 + 70  # safety margin before 90s kill
    
    # Weights: vertex i (0-indexed) has weight i+1
    weights = [i + 1 for i in range(N)]
    
    # Build conflict graph: neighbors[i] is bitset of j where gcd(i+1, j+1) > 1
    neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if i != j and math.gcd(vi, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask
    
    # can_coexist[i] = all vertices j where j==i or gcd(i+1,j+1)==1
    can_coexist = [0] * N
    all_mask = (1 << N) - 1
    for i in range(N):
        can_coexist[i] = all_mask ^ neighbors[i]
    
    # Greedy initial solution: pick highest-weight vertices that don't conflict
    def greedy_solution():
        avail = all_mask
        total = 0
        mask = 0
        while avail:
            # Pick the vertex with highest weight in avail
            best_v = -1
            best_w = -1
            tmp = avail
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            if best_v == -1:
                break
            mask |= (1 << best_v)
            total += best_w
            avail = avail & can_coexist[best_v] & ~(1 << best_v)
        return total, mask
    
    # Try multiple greedy orders for a better initial bound
    best_init_sum = 0
    best_init_mask = 0
    
    # Standard greedy (highest weight first)
    s, m = greedy_solution()
    best_init_sum = s
    best_init_mask = m
    
    # Greedy: try including each vertex first, then fill greedily
    for start_v in range(N):
        avail = all_mask & can_coexist[start_v] & ~(1 << start_v)
        total = weights[start_v]
        mask = (1 << start_v)
        while avail:
            best_v = -1
            best_w = -1
            tmp = avail
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            if best_v == -1:
                break
            mask |= (1 << best_v)
            total += best_v + 1
            avail = avail & can_coexist[best_v] & ~(1 << best_v)
        if total > best_init_sum:
            best_init_sum = total
            best_init_mask = mask
        if time.time() > deadline - 50:
            break
    
    best_sum = [best_init_sum]
    best_mask = [best_init_mask]
    
    # Fast upper bound: sum of weights of all available vertices
    # (weak but fast)
    def fast_ub(avail):
        total = 0
        tmp = avail
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            total += weights[idx]
            tmp ^= lsb
        return total
    
    # Better upper bound using clique cover (expensive, use selectively)
    def clique_cover_bound(avail):
        if avail == 0:
            return 0
        
        remaining = avail
        bound = 0
        while remaining:
            # Pick vertex with highest weight to start a clique
            best_v = -1
            best_w = -1
            tmp = remaining
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                if weights[idx] > best_w:
                    best_w = weights[idx]
                    best_v = idx
                tmp ^= lsb
            
            if best_v == -1:
                break
            
            # Greedily build a clique
            new_clique = 1 << best_v
            new_clique_max = weights[best_v]
            candidates = neighbors[best_v] & remaining
            while candidates:
                # Find highest weight candidate adjacent to all in new_clique
                best_cand = -1
                best_cand_w = -1
                tmp = candidates
                while tmp:
                    lsb = tmp & (-tmp)
                    idx = lsb.bit_length() - 1
                    # Check if idx is adjacent to all in new_clique
                    ok = True
                    nq = new_clique
                    while nq:
                        lsb2 = nq & (-nq)
                        idx2 = lsb2.bit_length() - 1
                        if not (neighbors[idx] & (1 << idx2)):
                            ok = False
                            break
                        nq ^= lsb2
                    if ok and weights[idx] > best_cand_w:
                        best_cand_w = weights[idx]
                        best_cand = idx
                    tmp ^= lsb
                if best_cand == -1:
                    break
                new_clique |= (1 << best_cand)
                new_clique_max = max(new_clique_max, best_cand_w)
                # Update candidates: must be adjacent to best_cand too
                candidates &= neighbors[best_cand]
            
            bound += new_clique_max
            remaining &= ~new_clique
        
        return bound
    
    # Precompute weight sums for quick reference
    weight_sum_cache = {}
    
    def get_weight_sum(avail):
        if avail in weight_sum_cache:
            return weight_sum_cache[avail]
        total = 0
        tmp = avail
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            total += weights[idx]
            tmp ^= lsb
        if len(weight_sum_cache) < 100000:
            weight_sum_cache[avail] = total
        return total
    
    # Branch and bound with improved strategies
    def branch_and_bound(available, current_sum, current_mask, depth=0):
        if time.time() > deadline:
            return
        
        if available == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_mask[0] = current_mask
            return
        
        # Quick upper bound: sum of all available weights
        avail_sum = get_weight_sum(available)
        if current_sum + avail_sum <= best_sum[0]:
            return
        
        # Use clique cover bound for deeper nodes (more expensive but tighter)
        if depth > 5:
            ub = current_sum + clique_cover_bound(available)
            if ub <= best_sum[0]:
                return
        
        # Select vertex: use a heuristic to pick a good branching vertex
        # Pick vertex with highest weight / degree ratio to maximize pruning
        best_v = -1
        best_score = -1
        tmp = available
        while tmp:
            lsb = tmp & (-tmp)
            idx = lsb.bit_length() - 1
            deg = bin(neighbors[idx] & available).count('1')
            score = weights[idx] / (deg + 1)
            if score > best_score:
                best_score = score
                best_v = idx
            tmp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: include best_v
        new_available = available & can_coexist[best_v] & ~(1 << best_v)
        branch_and_bound(new_available, current_sum + weights[best_v], current_mask | (1 << best_v), depth + 1)
        
        # Branch 2: exclude best_v
        new_available2 = available & ~(1 << best_v)
        branch_and_bound(new_available2, current_sum, current_mask, depth + 1)
    
    branch_and_bound(all_mask, 0, 0)
    
    # Extract the set
    result_mask = best_mask[0]
    result = []
    for i in range(N):
        if result_mask & (1 << i):
            result.append(i + 1)
    
    # Ensure 1 is included if possible (it's coprime with everything)
    if 1 not in result:
        result.append(1)
    
    return [int(x) for x in result]
