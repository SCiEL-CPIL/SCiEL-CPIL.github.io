import time
import random
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors for the specific case q=3, n=7, radius=1
    # If parameters differ, fall back to generic method
    if q == 3 and n == 7 and radius == 1:
        return _solve_k3_7_1()
    else:
        return _generic_solve(q, n, radius)

def _generic_solve(q, n, radius):
    # Fallback to the original logic if not the specific case
    # (Simplified version of the original program)
    total_words = q ** n
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    def greedy_construct(random_seed):
        if random_seed is not None:
            random.seed(random_seed)
        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []
        while uncovered.any():
            gains = np.sum(uncovered[neighbors_arr], axis=1)
            valid = gains > 0
            if not valid.any():
                break
            max_gain = np.max(gains[valid])
            best_candidates = np.where((gains == max_gain) & valid)[0]
            chosen = random.choice(best_candidates.tolist())
            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False
        return code

    def reduce_code(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search(code, deadline):
        code = list(code)
        if not code:
            return code
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)
            for i in indices:
                if time.time() >= deadline:
                    break
                c_remove = code[i]
                neighbors_c = neighbors_arr[c_remove, :]
                counts[neighbors_c] -= 1
                if np.any(counts == 0):
                    holes = np.where(counts == 0)[0]
                    cover_count_per_candidate = np.zeros(total_words, dtype=np.int16)
                    for h in holes:
                        cover_count_per_candidate[neighbors_arr[h, :]] += 1
                    all_holes = len(holes)
                    candidates = np.where(cover_count_per_candidate == all_holes)[0]
                    candidates = candidates[candidates != c_remove]
                    if len(candidates) == 0:
                        counts[neighbors_c] += 1
                        continue
                    cand_list = candidates.tolist()
                    if len(cand_list) > 100:
                        cand_list = random.sample(cand_list, 100)
                    cand_indices = np.array(cand_list, dtype=np.int32)
                    cand_neighbors = neighbors_arr[cand_indices, :]
                    scores = np.sum(counts[cand_neighbors], axis=1)
                    min_score = np.min(scores)
                    best_cands = np.where(scores == min_score)[0]
                    if len(best_cands) > 1 and random.random() < 0.3:
                        top_cands = np.where(scores <= min_score + 2)[0]
                        chosen_idx = random.choice(top_cands.tolist())
                    else:
                        chosen_idx = random.choice(best_cands.tolist())
                    best_cand = cand_list[chosen_idx]
                    code[i] = best_cand
                    counts[neighbors_arr[best_cand, :]] += 1
                    improved = True
                    break
                else:
                    code.pop(i)
                    improved = True
                    break
        return code

    start_time = time.time()
    deadline = start_time + 115
    best_code = None
    best_size = float('inf')
    seed = 0
    while time.time() < deadline:
        code = greedy_construct(seed)
        code = reduce_code(code)
        if time.time() < deadline - 5:
            code = local_search(code, deadline - 2)
            code = reduce_code(code)
        if len(code) < best_size:
            best_size = len(code)
            best_code = code
        seed += 1
        
    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
    return result

def _solve_k3_7_1():
    q = 3
    n = 7
    total_words = q ** n # 2187
    
    # Precompute neighbors as bitsets
    # N[i] is a bitset of all j such that dist(i, j) <= 1
    N = [0] * total_words
    N2 = [0] * total_words # dist <= 2 for packing LB
    
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Build N and N2
    # For speed, we can iterate through all pairs or generate neighbors
    # Generating neighbors is O(N * Degree)
    
    # Helper to get neighbors
    def get_neighbors(idx, max_dist):
        w = idx_to_tuple(idx)
        res = set([idx])
        if max_dist >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]: continue
                    new_w = list(w)
                    new_w[pos] = val
                    res.add(tuple_to_idx(tuple(new_w)))
        if max_dist >= 2:
            # For dist 2, we can use the dist 1 neighbors of dist 1 neighbors
            # But for N2, we just need the set.
            # Since N is small (15 neighbors), we can just expand N
            n1_set = get_neighbors(idx, 1)
            for u in n1_set:
                if u == idx: continue
                n1_u = get_neighbors(u, 1)
                res.update(n1_u)
        return res

    # This is O(N * 15 * 15) which is fine.
    for i in range(total_words):
        n1 = get_neighbors(i, 1)
        n2 = get_neighbors(i, 2)
        bitset1 = 0
        bitset2 = 0
        for j in n1:
            bitset1 |= (1 << j)
        for j in n2:
            bitset2 |= (1 << j)
        N[i] = bitset1
        N2[i] = bitset2

    # Initial Solution using Greedy
    def greedy_init():
        uncovered = (1 << total_words) - 1
        code = []
        # Heuristic: Pick vertex covering most uncovered
        # For simplicity, random or sequential
        # Let's do a simple greedy: pick 0, then pick next best
        # Actually, let's just pick 0 to fix symmetry, then greedy
        code.append(0)
        uncovered &= ~N[0]
        
        while uncovered:
            # Find v in uncovered that covers most uncovered neighbors
            # This is O(N^2) per step, might be slow.
            # Just pick the first available one for speed in init?
            # Or pick one with max degree in uncovered?
            best_v = -1
            max_gain = -1
            # Sample a few candidates to speed up
            candidates = []
            # Get index of first set bit
            # Python int doesn't have ctz, but we can iterate or use a list
            # For 2187 bits, iterating is okay.
            u = uncovered
            while u:
                # Get lowest set bit
                lb = u & -u
                v = lb.bit_length() - 1
                u ^= lb
                if len(candidates) < 50:
                    candidates.append(v)
                if len(candidates) >= 50:
                    break
            
            for v in candidates:
                gain = (N[v] & uncovered).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_v = v
            
            if best_v == -1: break
            code.append(best_v)
            uncovered &= ~N[best_v]
            
        return code, uncovered

    init_code, _ = greedy_init()
    
    # Local Search to improve init_code
    def local_search(code):
        code = list(code)
        # Counts
        counts = [0] * total_words
        for c in code:
            # Add 1 to all neighbors of c
            # Bitset trick is hard for counts. Use array.
            # N[c] is bitset. We need to iterate set bits.
            u = N[c]
            while u:
                lb = u & -u
                idx = lb.bit_length() - 1
                counts[idx] += 1
                u ^= lb
        
        improved = True
        while improved:
            improved = False
            for i in range(len(code)):
                c = code[i]
                # Remove c
                u = N[c]
                while u:
                    lb = u & -u
                    idx = lb.bit_length() - 1
                    counts[idx] -= 1
                    u ^= lb
                
                # Check if any count is 0
                # If yes, we need to replace c
                # Find holes
                holes = []
                for j in range(total_words):
                    if counts[j] == 0:
                        holes.append(j)
                
                if not holes:
                    # Can remove c
                    code.pop(i)
                    improved = True
                    break
                else:
                    # Find candidate that covers all holes
                    # Holes are small usually
                    # For each hole h, candidate must be in N[h]
                    # Intersection of N[h] for all h in holes
                    # Start with N[holes[0]]
                    cand_set = N[holes[0]]
                    for h in holes[1:]:
                        cand_set &= N[h]
                    
                    # cand_set contains candidates that cover ALL holes
                    # We also need to prefer candidates that overlap less with existing code
                    # or just pick one
                    # Exclude c itself?
                    # If c is in cand_set, it covers the holes?
                    # Wait, we removed c. The holes are uncovered by the rest.
                    # c covers the holes. So c is a valid candidate.
                    # But we want to replace c.
                    # If we pick c, we restore it.
                    # Let's pick a candidate != c if possible.
                    
                    cand_list = []
                    u = cand_set
                    while u:
                        lb = u & -u
                        idx = lb.bit_length() - 1
                        if idx != c:
                            cand_list.append(idx)
                        u ^= lb
                    
                    if not cand_list:
                        # Must restore c
                        code[i] = c
                        # Add c back
                        u = N[c]
                        while u:
                            lb = u & -u
                            idx = lb.bit_length() - 1
                            counts[idx] += 1
                            u ^= lb
                        continue
                    else:
                        # Pick one (e.g., first or random)
                        # To improve, pick one that minimizes overlap?
                        # For now, pick first
                        new_c = cand_list[0]
                        code[i] = new_c
                        u = N[new_c]
                        while u:
                            lb = u & -u
                            idx = lb.bit_length() - 1
                            counts[idx] += 1
                            u ^= lb
                        improved = True
                        break
        return code

    best_code = local_search(init_code)
    best_size = len(best_code)
    
    # Branch and Bound
    # State: (uncovered_bitset, code_count, code_list)
    # LB: Greedy packing of uncovered
    
    def lb_packing(uncovered):
        # Greedy packing with dist >= 3
        # u_rem = uncovered
        # count = 0
        # while u_rem:
        #   v = first set bit
        #   count += 1
        #   u_rem &= ~N2[v] # Remove dist <= 2
        # return count
        u_rem = uncovered
        count = 0
        while u_rem:
            lb = u_rem & -u_rem
            v = lb.bit_length() - 1
            count += 1
            u_rem &= ~N2[v]
        return count

    import sys
    sys.setrecursionlimit(3000)
    
    deadline = time.time() + 110
    global_best = best_size
    global_best_code = list(best_code)
    
    # Symmetry breaking: 0 is in the code
    # Start state: 0 is selected
    start_uncovered = ((1 << total_words) - 1) & ~N[0]
    
    def bnb(uncovered, count, code_list):
        nonlocal global_best, global_best_code
        
        if time.time() > deadline:
            return
            
        if not uncovered:
            if count < global_best:
                global_best = count
                global_best_code = list(code_list)
            return
            
        # LB Pruning
        lb = lb_packing(uncovered)
        if count + lb >= global_best:
            return
            
        # Select vertex v
        # Heuristic: Max degree in uncovered
        # u = uncovered
        # best_v = -1
        # max_deg = -1
        # while u:
        #   lb = u & -u
        #   v = lb.bit_length() - 1
        #   deg = (N[v] & uncovered).bit_count()
        #   if deg > max_deg:
        #      max_deg = deg
        #      best_v = v
        #   u ^= lb
        
        # For speed, just pick the first set bit?
        # Or pick one with max (N[v] & uncovered).bit_count()
        # Let's try first set bit for speed, might be less effective but faster
        lb = uncovered & -uncovered
        v = lb.bit_length() - 1
        
        # Branch 1: Include v
        new_uncovered = uncovered & ~N[v]
        code_list.append(v)
        bnb(new_uncovered, count + 1, code_list)
        code_list.pop()
        
        # Branch 2: Exclude v
        # Must include a neighbor of v
        # Neighbors of v in uncovered: N[v] & uncovered
        # Note: v is in uncovered, so v is in N[v].
        # We want u in N(v) (dist 1) such that u is in uncovered.
        # Actually, if we exclude v, v must be covered by a neighbor u.
        # u must be in the code.
        # So we branch on each u in N[v] (dist <= 1) intersected with uncovered?
        # If u = v, that's Branch 1.
        # So we branch on u in (N[v] & uncovered) where u != v.
        
        u_cand = (N[v] & uncovered)
        # Remove v
        u_cand &= ~(1 << v)
        
        while u_cand:
            lb_u = u_cand & -u_cand
            u = lb_u.bit_length() - 1
            u_cand ^= lb_u
            
            new_uncovered = uncovered & ~N[u]
            code_list.append(u)
            bnb(new_uncovered, count + 1, code_list)
            code_list.pop()
            
            if time.time() > deadline:
                return

    # Run B&B
    bnb(start_uncovered, 1, [0])
    
    # Final result
    result = []
    for idx in global_best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))
        
    return result
