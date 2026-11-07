import time
import sys
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    candidates = list(range(2, N + 1))
    num_candidates = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute prime factorization
    factors = [None] * num_candidates
    for i, v in enumerate(candidates):
        x = v
        f = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                f[d] = f.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            f[x] = f.get(x, 0) + 1
        factors[i] = f
    
    # Precompute divs[a][b] = bitmask of c such that a | (b*c)
    divs = [[0] * num_candidates for _ in range(num_candidates)]
    
    for a_idx in range(num_candidates):
        a_val = candidates[a_idx]
        fa = factors[a_idx]
        for b_idx in range(num_candidates):
            if b_idx == a_idx:
                continue
            b_val = candidates[b_idx]
            fb = factors[b_idx]
            mask = 0
            for c_idx in range(num_candidates):
                if c_idx == a_idx or c_idx == b_idx:
                    continue
                fc = factors[c_idx]
                ok = True
                for p, exp_a in fa.items():
                    exp_bc = fb.get(p, 0) + fc.get(p, 0)
                    if exp_a > exp_bc:
                        ok = False
                        break
                if ok:
                    mask |= (1 << c_idx)
            divs[a_idx][b_idx] = mask
    
    # For each pair (b, c), precompute the set of x that would conflict with (b, c)
    # conflict_pair[b][c] = bitmask of x such that adding x creates a violation with (b,c)
    # Violation if: x|(b*c) or b|(x*c) or c|(b*x)
    # i.e., c in divs[x][b] or c in divs[b][x] or x in divs[c][b]
    
    # Actually, let's precompute for each x: 
    #   bad_pairs_for_x[x] = set of pairs (b,c) that conflict with x
    # But that's too much memory. Instead, let's precompute:
    #   For each x and b: conflict_bc[x][b] = bitmask of c such that (b,c) conflicts with x
    #   (b,c) conflicts with x if: c in divs[x][b] or c in divs[b][x] or x in divs[c][b]
    
    # Let's precompute conflict_mask[x][b] = bitmask of c where (b,c) would violate with x
    conflict_mask = [[0] * num_candidates for _ in range(num_candidates)]
    
    for x in range(num_candidates):
        for b in range(num_candidates):
            if b == x:
                continue
            mask = 0
            for c in range(num_candidates):
                if c == x or c == b:
                    continue
                # Check if (b, c) conflicts with x
                # x | (b*c): c in divs[x][b]
                if divs[x][b] & (1 << c):
                    mask |= (1 << c)
                # b | (x*c): c in divs[b][x]
                elif divs[b][x] & (1 << c):
                    mask |= (1 << c)
                # c | (b*x): x in divs[c][b]
                elif divs[c][b] & (1 << x):
                    mask |= (1 << c)
            conflict_mask[x][b] = mask
    
    # Now can_add(x, S_mask): for each b in S, check if any c in S is in conflict_mask[x][b]
    # i.e., (S_mask & conflict_mask[x][b]) should be 0 for all b in S
    # But we also need to exclude b=x and c=x, which are already handled since x not in S.
    
    def can_add_fast(x, S_mask):
        """Check if x can be added to S (represented as bitmask)."""
        for b in range(num_candidates):
            if S_mask & (1 << b):
                if b == x:
                    continue
                # Check if any c in S conflicts with (x, b)
                if S_mask & conflict_mask[x][b]:
                    return False
        return True
    
    # Precompute risk: for each x, total number of conflicting pairs in the full set
    risk = [0] * num_candidates
    for x in range(num_candidates):
        count = 0
        for b in range(num_candidates):
            if b == x:
                continue
            count += bin(conflict_mask[x][b] & ~(1 << x) & ~(1 << b)).count('1')
        risk[x] = count
    
    # Sort by risk ascending (safe first)
    sorted_by_risk = sorted(range(num_candidates), key=lambda i: risk[i])
    sorted_desc = sorted(range(num_candidates), key=lambda i: candidates[i], reverse=True)
    sorted_asc = sorted(range(num_candidates), key=lambda i: candidates[i])
    
    def greedy(order):
        S = []
        S_mask = 0
        for idx in order:
            if can_add_fast(idx, S_mask):
                S.append(idx)
                S_mask |= (1 << idx)
        return S, S_mask
    
    def local_search(S, S_mask, max_time):
        not_in_S_mask = ((1 << num_candidates) - 1) & ~S_mask
        not_in_S_list = [i for i in range(num_candidates) if not (S_mask & (1 << i))]
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try additions
            random.shuffle(not_in_S_list)
            for idx in not_in_S_list:
                if can_add_fast(idx, S_mask):
                    S.append(idx)
                    S_mask |= (1 << idx)
                    not_in_S_list.remove(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try single swaps
            if S and not_in_S_list:
                for _ in range(200):
                    if time.time() >= max_time:
                        break
                    i = random.choice(S)
                    j = random.choice(not_in_S_list)
                    S_minus_i_mask = S_mask & ~(1 << i)
                    if can_add_fast(j, S_minus_i_mask):
                        S.remove(i)
                        S.append(j)
                        S_mask = S_minus_i_mask | (1 << j)
                        not_in_S_list.remove(j)
                        not_in_S_list.append(i)
                        improved = True
                        break
            
            # Try 2-swap moves: remove i1, i2, add j1, j2
            if len(S) >= 2 and len(not_in_S_list) >= 2 and time.time() < max_time:
                for _ in range(50):
                    if time.time() >= max_time:
                        break
                    i1, i2 = random.sample(S, 2)
                    j1, j2 = random.sample(not_in_S_list, 2)
                    S_minus_mask = S_mask & ~(1 << i1) & ~(1 << i2)
                    # Check if both j1 and j2 can be added to S \ {i1, i2}
                    if can_add_fast(j1, S_minus_mask):
                        temp_mask = S_minus_mask | (1 << j1)
                        if can_add_fast(j2, temp_mask):
                            # Perform the swap
                            S.remove(i1)
                            S.remove(i2)
                            S.append(j1)
                            S.append(j2)
                            S_mask = temp_mask | (1 << j2)
                            not_in_S_list.remove(j1)
                            not_in_S_list.remove(j2)
                            not_in_S_list.append(i1)
                            not_in_S_list.append(i2)
                            improved = True
                            break
        
        return S, S_mask
    
    best_S = []
    best_size = 0
    rng = random.Random(12345)
    
    # Multiple restarts with different orderings
    num_restarts = 300
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r % 4 == 0:
            order = sorted_by_risk[:]
            k = min(40, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 1:
            order = sorted_desc[:]
            k = min(40, len(order))
            rng.shuffle(order[:k])
        elif r % 4 == 2:
            order = sorted_asc[:]
            k = min(40, len(order))
            rng.shuffle(order[:k])
        else:
            order = list(range(num_candidates))
            rng.shuffle(order)
        
        S, S_mask = greedy(order)
        
        time_left = deadline - time.time()
        search_time = max(0.5, time_left * 0.4)
        S, S_mask = local_search(S, S_mask, start_time + search_time)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
    
    # Final: try to improve best_S with more local search
    if best_S:
        S_mask = 0
        for idx in best_S:
            S_mask |= (1 << idx)
        time_left = deadline - time.time()
        if time_left > 2:
            best_S, S_mask = local_search(best_S, S_mask, deadline - 1)
    
    result = [candidates[i] for i in best_S]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
