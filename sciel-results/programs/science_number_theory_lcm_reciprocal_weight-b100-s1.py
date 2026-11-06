import time
import sys
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    N = 256
    
    def lcm(a, b):
        return a // gcd(a, b) * b

    # Precompute conflict graph for all pairs in [2, 256]
    adj_val = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj_val[a].add(b)
                adj_val[b].add(a)
                
    def solve_mwis(values, seed=0):
        if not values:
            return 0.0, []
        
        verts = sorted(values)
        n = len(verts)
        
        val_to_idx = {v: i for i, v in enumerate(verts)}
        
        adj = [set() for _ in range(n)]
        for i in range(n):
            vi = verts[i]
            for j in range(i + 1, n):
                vj = verts[j]
                if vj in adj_val[vi]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        weights = [1.0 / v for v in verts]
        
        # Randomized Greedy Initialization
        # We want to pick high weight (small value) nodes.
        # To diversify, we can shuffle the order of nodes with similar weights or just shuffle the whole list slightly?
        # Better: Pick the best available node, but with some probability pick the 2nd or 3rd best.
        # Or simply: Sort by weight, then apply a random permutation to the first K elements?
        # Let's try: Sort by weight descending. Then, for the first few picks, introduce randomness.
        
        # Actually, a common heuristic is to sort by weight/degree ratio, but here weights are fixed.
        # Let's just shuffle the candidates for the first few picks.
        
        rng = random.Random(seed)
        
        chosen_indices = []
        available = set(range(n))
        
        # Order by weight descending (value ascending)
        # We will iterate through a shuffled list of indices, but prioritize small values?
        # No, standard greedy is: while available, pick max weight node that is available.
        # To randomize: pick from top 5 available weights with uniform probability.
        
        while available:
            # Get sorted available by weight (value ascending)
            # Since n is small (<256), sorting each time is O(N^2 log N) total? No, N log N per step, N steps -> N^2 log N.
            # 256^2 * 8 ~ 500k ops. Fine.
            avail_list = sorted(available, key=lambda i: verts[i])
            
            # Pick from top min(5, len(avail_list))
            k = min(5, len(avail_list))
            # If k==1, pick the only one.
            # If k>1, pick randomly from first k.
            # But we must ensure we don't pick a node that conflicts with already chosen?
            # Wait, the greedy logic in the original code was:
            # for i in order: if i in available: pick i, remove neighbors.
            # This is a sequential scan.
            # To randomize this sequential scan, we can shuffle the `order` list?
            # If we shuffle `order`, we might pick a large value early, blocking small values.
            # That's bad.
            # We want to keep the bias towards small values but add diversity.
            
            # Alternative:
            # Maintain a list of candidates.
            # At each step, consider the top K smallest values in `available`.
            # Pick one randomly from them.
            # Add to chosen, remove it and its neighbors from available.
            
            if not avail_list:
                break
                
            # Top K smallest values
            candidates = avail_list[:k]
            # Pick one
            pick_idx = rng.choice(candidates)
            
            chosen_indices.append(pick_idx)
            available.remove(pick_idx)
            for j in adj[pick_idx]:
                if j in available:
                    available.remove(j)
                        
        best_sum = sum(weights[i] for i in chosen_indices)
        best_set = set(chosen_indices)
        current_set = set(chosen_indices)
        current_sum = best_sum
        
        # Local Search: 1-swap and 2-out-1-in
        while time.time() < deadline - 2.0:
            improved = False
            
            # 1. Try adding nodes (0-out, 1-in)
            for i in sorted(available, key=lambda x: verts[x]):
                if i in current_set:
                    continue
                if not (adj[i] & current_set):
                    current_set.add(i)
                    current_sum += weights[i]
                    improved = True
            
            if not improved:
                # 2. Try 1-swap (1-out, 1-in)
                best_swap_gain = 0
                best_swap_i = -1
                best_swap_j = -1
                
                for j in available:
                    if j in current_set:
                        continue
                    C_j = adj[j] & current_set
                    if not C_j:
                        continue
                    
                    min_weight_i = min(weights[i] for i in C_j)
                    gain = weights[j] - min_weight_i
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_j = j
                        for i in C_j:
                            if weights[i] == min_weight_i:
                                best_swap_i = i
                                break
                
                if best_swap_gain > 0:
                    current_set.remove(best_swap_i)
                    current_set.add(best_swap_j)
                    current_sum += best_swap_gain
                    improved = True
            
            if not improved:
                # 3. Try 2-out, 1-in
                # For each j not in current_set:
                # C_j = adj[j] & current_set
                # If len(C_j) == 2:
                #    Let C_j = {i1, i2}
                #    gain = weights[j] - weights[i1] - weights[i2]
                #    if gain > 0: swap
                best_2out_gain = 0
                best_2out_j = -1
                best_2out_remove = []
                
                for j in available:
                    if j in current_set:
                        continue
                    C_j = adj[j] & current_set
                    if len(C_j) == 2:
                        # Get the two nodes
                        it = iter(C_j)
                        i1 = next(it)
                        i2 = next(it)
                        gain = weights[j] - weights[i1] - weights[i2]
                        if gain > best_2out_gain:
                            best_2out_gain = gain
                            best_2out_j = j
                            best_2out_remove = [i1, i2]
                
                if best_2out_gain > 0:
                    for i in best_2out_remove:
                        current_set.remove(i)
                    current_set.add(best_2out_j)
                    current_sum += best_2out_gain
                    improved = True
            
            if not improved:
                break
                
        return current_sum, [verts[i] for i in current_set]

    V0 = set(range(17, 257))
    best_score = 0.0
    best_candidate = []
    
    # Run multiple times for V0
    # We have ~50s. Each run takes < 1s probably.
    # Let's do 10 restarts for V0 and 10 for each x.
    # Total 1 + 15 * 10 = 160 runs? Might be too many if each is slow.
    # Let's check time.
    
    num_restarts = 20
    
    # Case 0: No numbers from 2..16
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        sum0, set0 = solve_mwis(V0, seed=r)
        if sum0 > best_score:
            best_score = sum0
            best_candidate = list(set0)
            
    # Case x: Include exactly one small number x
    for x in range(2, 17):
        if time.time() > deadline - 10.0:
            break
            
        Vx = []
        for y in range(2, 257):
            if y == x:
                continue
            if lcm(x, y) > 256:
                Vx.append(y)
        
        base_score = 1.0 / x
        
        for r in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            sum_x, set_x = solve_mwis(Vx, seed=r)
            score = base_score + sum_x
            if score > best_score:
                best_score = score
                best_candidate = list(set_x)
                best_candidate.append(x)
                
    return best_candidate
