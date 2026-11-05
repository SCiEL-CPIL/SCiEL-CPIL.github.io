import time
from math import lcm
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 256
    
    # Precompute conflicts: conflicts[i] is a set of j such that lcm(i, j) <= 256
    conflicts = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        for j in range(i + 1, N + 1):
            if lcm(i, j) <= N:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Maximum Weight Closure formulation:
    # We want to select a subset S of [2,256] maximizing sum(1/x for x in S)
    # subject to: for all distinct a,b in S, lcm(a,b) > 256
    #
    # This is equivalent to: for each conflicting pair (a,b), we cannot select both.
    #
    # Maximum Weight Closure on a directed graph:
    # - Nodes: source s, sink t, and one node v_i for each i in [2,256]
    # - For each i with weight w_i = 1/i > 0: edge s -> v_i with capacity w_i
    # - For each i with weight w_i < 0: edge v_i -> t with capacity -w_i
    #   (Here all weights are positive, so no v_i -> t edges)
    # - For each conflicting pair (i,j): add edges v_i -> v_j and v_j -> v_i with infinite capacity
    #
    # The minimum s-t cut gives the optimal closure.
    # Nodes reachable from s in the residual graph after min-cut are the selected set.
    #
    # Since all weights are positive, we can simplify:
    # The problem becomes: find maximum weight independent set in the conflict graph.
    # This is NP-hard in general, but for our specific structure we can use min-cut.
    
    # Actually, let me reconsider. The standard reduction for maximum weight closure
    # with pairwise exclusion constraints:
    #
    # For a graph G=(V,E) with weights w_v, the maximum weight independent set can be
    # found via min-cut if the graph is bipartite. But our conflict graph may not be bipartite.
    #
    # Alternative approach: Use the fact that this is a "maximum weight closure" problem
    # where selecting a node forces us to NOT select its neighbors.
    #
    # Let me use a different formulation. Define binary variables x_i in {0,1}.
    # Maximize sum_i w_i * x_i
    # Subject to: x_i + x_j <= 1 for all (i,j) in E (conflict edges)
    #
    # This is the maximum weight independent set problem. For general graphs, this is NP-hard.
    # However, we can use a min-cut based approach if we reformulate.
    #
    # Actually, the correct reduction for maximum weight independent set via min-cut
    # works when the graph is bipartite. Let me check if our conflict graph is bipartite.
    #
    # Alternative: Use branch and bound or a smarter local search.
    #
    # Let me try a different approach: Start with a good initial solution and use
    # more sophisticated local search moves (2-swap, 3-swap, etc.) with simulated annealing.
    
    # Precompute reciprocal weights
    weights = {i: 1.0 / i for i in range(2, N + 1)}
    
    # Initial solution: all numbers in [129, 256]
    best_set = set(range(129, N + 1))
    best_score = sum(weights[x] for x in best_set)
    
    def is_compatible(x, current_set):
        for y in conflicts[x]:
            if y in current_set:
                return False
        return True
    
    # Simulated annealing with multiple move types
    import random
    
    current_set = set(best_set)
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    cooling_rate = 0.999
    
    T = T_start
    
    def get_neighbors_2swap(current_set):
        """Generate 2-swap moves: remove two elements, add two new elements."""
        moves = []
        current_list = list(current_set)
        n = len(current_list)
        
        # For each pair to remove, find compatible pairs to add
        for i in range(n):
            for j in range(i + 1, n):
                rem_i = current_list[i]
                rem_j = current_list[j]
                remaining = set(current_list) - {rem_i, rem_j}
                
                # Find all elements compatible with remaining
                candidates = []
                for x in range(2, N + 1):
                    if x in current_set:
                        continue
                    if is_compatible(x, remaining):
                        candidates.append(x)
                
                # Try pairs from candidates
                for a_idx in range(len(candidates)):
                    for b_idx in range(a_idx + 1, len(candidates)):
                        a = candidates[a_idx]
                        b = candidates[b_idx]
                        if is_compatible(a, remaining | {b}) and is_compatible(b, remaining | {a}):
                            gain = weights[a] + weights[b] - weights[rem_i] - weights[rem_j]
                            moves.append((rem_i, rem_j, a, b, gain))
        
        return moves
    
    # Main annealing loop
    iteration = 0
    while time.time() < deadline:
        improved = False
        
        # 1. Try single add
        best_add = None
        best_add_val = 0.0
        for x in range(2, N + 1):
            if x not in current_set:
                if is_compatible(x, current_set):
                    val = weights[x]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = x
        
        if best_add is not None:
            current_set.add(best_add)
            current_score += best_add_val
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue
        
        # 2. Try 1-swap (remove y, add x)
        best_swap_gain = 0.0
        best_swap_x = None
        best_swap_y = None
        
        for y in list(current_set):
            for x in range(2, y):
                if x in current_set:
                    continue
                valid = True
                for z in conflicts[x]:
                    if z in current_set and z != y:
                        valid = False
                        break
                if valid:
                    gain = weights[x] - weights[y]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_x = x
                        best_swap_y = y
        
        if best_swap_x is not None:
            current_set.remove(best_swap_y)
            current_set.add(best_swap_x)
            current_score += best_swap_gain
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue
        
        # 3. Try 2-swap (remove two, add two) - limited search for speed
        best_2swap_gain = 0.0
        best_2swap = None
        
        current_list = list(current_set)
        n = len(current_list)
        
        # Sample a limited number of 2-swap moves for speed
        num_samples = min(200, n * (n - 1) // 2)
        random.seed(iteration)
        sampled_pairs = random.sample(range(n * (n - 1) // 2), num_samples)
        
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx not in sampled_pairs:
                    pair_idx += 1
                    continue
                pair_idx += 1
                
                rem_i = current_list[i]
                rem_j = current_list[j]
                remaining = set(current_list) - {rem_i, rem_j}
                
                # Find compatible candidates
                candidates = []
                for x in range(2, N + 1):
                    if x in current_set:
                        continue
                    if is_compatible(x, remaining):
                        candidates.append(x)
                
                if len(candidates) < 2:
                    continue
                
                # Sample a few candidate pairs
                num_cand_samples = min(50, len(candidates) * (len(candidates) - 1) // 2)
                random.seed(iteration * 1000 + pair_idx)
                cand_pairs = random.sample(range(len(candidates) * (len(candidates) - 1) // 2), num_cand_samples)
                
                cand_pair_idx = 0
                for a_idx in range(len(candidates)):
                    for b_idx in range(a_idx + 1, len(candidates)):
                        if cand_pair_idx not in cand_pairs:
                            cand_pair_idx += 1
                            continue
                        cand_pair_idx += 1
                        
                        a = candidates[a_idx]
                        b = candidates[b_idx]
                        if is_compatible(a, remaining | {b}) and is_compatible(b, remaining | {a}):
                            gain = weights[a] + weights[b] - weights[rem_i] - weights[rem_j]
                            if gain > best_2swap_gain:
                                best_2swap_gain = gain
                                best_2swap = (rem_i, rem_j, a, b)
        
        if best_2swap is not None:
            rem_i, rem_j, a, b = best_2swap
            current_set.remove(rem_i)
            current_set.remove(rem_j)
            current_set.add(a)
            current_set.add(b)
            current_score += best_2swap_gain
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
            improved = True
            continue
        
        # 4. If no improving move, try a random perturbation (simulated annealing step)
        if T > T_end:
            # Try a random 1-swap or 2-swap, accept with probability exp(gain/T)
            move_type = random.choice([1, 2])
            
            if move_type == 1:
                # Random 1-swap
                y = random.choice(list(current_set))
                candidates = []
                for x in range(2, N + 1):
                    if x in current_set:
                        continue
                    valid = True
                    for z in conflicts[x]:
                        if z in current_set and z != y:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                
                if candidates:
                    x = random.choice(candidates)
                    gain = weights[x] - weights[y]
                    if gain > 0 or random.random() < pow(2.718281828, gain / T):
                        current_set.remove(y)
                        current_set.add(x)
                        current_score += gain
                        if current_score > best_score:
                            best_score = current_score
                            best_set = set(current_set)
                        improved = True
            
            if not improved and move_type == 2:
                # Random 2-swap
                if len(current_set) >= 2:
                    rem_i, rem_j = random.sample(list(current_set), 2)
                    remaining = set(current_set) - {rem_i, rem_j}
                    
                    candidates = []
                    for x in range(2, N + 1):
                        if x in current_set:
                            continue
                        if is_compatible(x, remaining):
                            candidates.append(x)
                    
                    if len(candidates) >= 2:
                        a, b = random.sample(candidates, 2)
                        if is_compatible(a, remaining | {b}) and is_compatible(b, remaining | {a}):
                            gain = weights[a] + weights[b] - weights[rem_i] - weights[rem_j]
                            if gain > 0 or random.random() < pow(2.718281828, gain / T):
                                current_set.remove(rem_i)
                                current_set.remove(rem_j)
                                current_set.add(a)
                                current_set.add(b)
                                current_score += gain
                                if current_score > best_score:
                                    best_score = current_score
                                    best_set = set(current_set)
                                improved = True
        
        if not improved:
            # Cool down
            T *= cooling_rate
            iteration += 1
            
            # If temperature is very low, try a few more iterations with different random seeds
            if T < T_end * 1.1:
                # Try a few random restarts from best_set
                for restart in range(3):
                    current_set = set(best_set)
                    current_score = best_score
                    random.seed(12345 + restart)
                    
                    # Do a few random perturbations
                    for _ in range(20):
                        y = random.choice(list(current_set))
                        candidates = []
                        for x in range(2, N + 1):
                            if x in current_set:
                                continue
                            valid = True
                            for z in conflicts[x]:
                                if z in current_set and z != y:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                        
                        if candidates:
                            x = random.choice(candidates)
                            current_set.remove(y)
                            current_set.add(x)
                            current_score = sum(weights[v] for v in current_set)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
                    
                    if time.time() > deadline - 2:
                        break
            
            # If we've been stuck for a while, break
            if T < T_end * 0.5:
                break
    
    return list(best_set)
