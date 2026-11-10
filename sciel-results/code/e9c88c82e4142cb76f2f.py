import math
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Nodes are 2..256
    nodes = list(range(2, N + 1))
    n_nodes = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Precompute conflicts
    # conflict[i][j] is True if lcm(nodes[i], nodes[j]) <= N
    conflict = [[False] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        a = nodes[i]
        for j in range(i + 1, n_nodes):
            b = nodes[j]
            # Optimization: if a*b > N * gcd(a,b) then lcm > N
            # lcm(a,b) = a*b//gcd(a,b)
            g = math.gcd(a, b)
            if a * b // g <= N:
                conflict[i][j] = True
                conflict[j][i] = True
                
    # Weights are 1/x
    weights = [1.0 / nodes[i] for i in range(n_nodes)]
    
    def get_sum(selected_indices):
        return sum(weights[i] for i in selected_indices)
        
    def is_valid_addition(selected_set, v_idx):
        # Check if v_idx conflicts with any in selected_set
        for u_idx in selected_set:
            if conflict[v_idx][u_idx]:
                return False
        return True
        
    def local_search(selected_set):
        # selected_set is a set of indices
        improved = True
        while improved:
            improved = False
            current_sum = get_sum(selected_set)
            
            # 1. Try Add
            # Sort candidates by weight descending to find best add first? 
            # Or just iterate.
            best_add_gain = 0
            best_add_idx = None
            for v_idx in range(n_nodes):
                if v_idx not in selected_set:
                    if is_valid_addition(selected_set, v_idx):
                        gain = weights[v_idx]
                        if gain > best_add_gain:
                            best_add_gain = gain
                            best_add_idx = v_idx
                            
            if best_add_idx is not None:
                selected_set.add(best_add_idx)
                improved = True
                continue
                
            # 2. Try Swap (remove u, add v)
            # We want max (weights[v] - weights[u]) such that v is valid wrt selected_set \ {u}
            best_swap_gain = 0
            best_swap_u = None
            best_swap_v = None
            
            # Iterate over u in selected_set
            for u_idx in list(selected_set):
                # Potential v's are those not in selected_set
                # To speed up, we can check all v not in selected_set
                # If n is small, this is fine.
                for v_idx in range(n_nodes):
                    if v_idx in selected_set or v_idx == u_idx:
                        continue
                    
                    # Check if v is compatible with all w in selected_set \ {u}
                    compatible = True
                    for w_idx in selected_set:
                        if w_idx == u_idx:
                            continue
                        if conflict[v_idx][w_idx]:
                            compatible = False
                            break
                            
                    if compatible:
                        gain = weights[v_idx] - weights[u_idx]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_u = u_idx
                            best_swap_v = v_idx
                            
            if best_swap_u is not None:
                selected_set.remove(best_swap_u)
                selected_set.add(best_swap_v)
                improved = True
                continue
                
            # 3. Try Remove (if it allows adding something better? 
            # This is covered by swap if we consider "adding nothing" as v=None? 
            # No, swap requires adding. 
            # Just removing is only useful if it allows a sequence of adds? 
            # Local search usually handles this via swaps. 
            # However, sometimes removing a low weight node allows adding TWO high weight nodes?
            # That's a 1-for-2 move.
            
            # Let's try 1-for-2: remove u, add v1, v2
            # Gain = w[v1] + w[v2] - w[u]
            # Constraints: v1, v2 not in S, v1!=v2, 
            # v1 compatible with S\{u}, v2 compatible with S\{u}, 
            # v1 compatible with v2.
            
            best_1for2_gain = 0
            best_1for2_u = None
            best_1for2_v1 = None
            best_1for2_v2 = None
            
            for u_idx in list(selected_set):
                S_minus_u = selected_set - {u_idx}
                # Find all v compatible with S_minus_u
                candidates = []
                for v_idx in range(n_nodes):
                    if v_idx in selected_set:
                        continue
                    if is_valid_addition(S_minus_u, v_idx):
                        candidates.append(v_idx)
                        
                # Try pairs from candidates
                # To keep it fast, limit candidates or just iterate
                # If candidates is large, this is O(K^2).
                # K is likely small for specific u.
                for i_cand in range(len(candidates)):
                    v1 = candidates[i_cand]
                    for j_cand in range(i_cand + 1, len(candidates)):
                        v2 = candidates[j_cand]
                        if conflict[v1][v2]:
                            continue
                        gain = weights[v1] + weights[v2] - weights[u_idx]
                        if gain > best_1for2_gain:
                            best_1for2_gain = gain
                            best_1for2_u = u_idx
                            best_1for2_v1 = v1
                            best_1for2_v2 = v2
                            
            if best_1for2_u is not None:
                selected_set.remove(best_1for2_u)
                selected_set.add(best_1for2_v1)
                selected_set.add(best_1for2_v2)
                improved = True
                continue
                
            # 4. Try 2-for-1: remove u1, u2, add v
            # Gain = w[v] - w[u1] - w[u2]
            best_2for1_gain = 0
            best_2for1_u1 = None
            best_2for1_u2 = None
            best_2for1_v = None
            
            # Only try if |S| > 1
            if len(selected_set) > 1:
                u_list = list(selected_set)
                for i_u1 in range(len(u_list)):
                    u1 = u_list[i_u1]
                    for i_u2 in range(i_u1 + 1, len(u_list)):
                        u2 = u_list[i_u2]
                        S_minus_u1u2 = selected_set - {u1, u2}
                        
                        for v_idx in range(n_nodes):
                            if v_idx in selected_set:
                                continue
                            if is_valid_addition(S_minus_u1u2, v_idx):
                                gain = weights[v_idx] - weights[u1] - weights[u2]
                                if gain > best_2for1_gain:
                                    best_2for1_gain = gain
                                    best_2for1_u1 = u1
                                    best_2for1_u2 = u2
                                    best_2for1_v = v_idx
                                    
            if best_2for1_u1 is not None:
                selected_set.remove(best_2for1_u1)
                selected_set.remove(best_2for1_u2)
                selected_set.add(best_2for1_v)
                improved = True
                continue
                
            # If no improvement in any move, stop
            if not improved:
                break
                
        return selected_set

    best_solution = set()
    best_score = 0.0
    
    # Restart loop
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization
        if r == 0:
            # Greedy by weight
            selected_set = set()
            sorted_indices = sorted(range(n_nodes), key=lambda i: weights[i], reverse=True)
            for idx in sorted_indices:
                if is_valid_addition(selected_set, idx):
                    selected_set.add(idx)
        elif r == 1:
            # Top half initialization
            selected_set = set()
            for i in range(n_nodes):
                if nodes[i] > N // 2:
                    selected_set.add(i)
        else:
            # Random valid subset initialization
            selected_set = set()
            random.shuffle(nodes)
            # Use original indices for shuffle? No, shuffle indices
            indices_to_try = list(range(n_nodes))
            random.shuffle(indices_to_try)
            for idx in indices_to_try:
                if is_valid_addition(selected_set, idx):
                    selected_set.add(idx)
        
        # Local Search
        optimized_set = local_search(selected_set)
        
        score = get_sum(optimized_set)
        if score > best_score:
            best_score = score
            best_solution = optimized_set
            
    # Convert indices back to values
    result = [nodes[i] for i in best_solution]
    
    if not result:
        # Fallback: just return a valid single element, e.g., N
        return [N]
        
    return result

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Nodes are 2..256
    nodes = list(range(2, N + 1))
    n_nodes = len(nodes)
    
    # Precompute conflicts
    conflict = [[False] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        a = nodes[i]
        for j in range(i + 1, n_nodes):
            b = nodes[j]
            g = math.gcd(a, b)
            if a * b // g <= N:
                conflict[i][j] = True
                conflict[j][i] = True
                
    weights = [1.0 / nodes[i] for i in range(n_nodes)]
    
    def get_sum(selected_indices):
        return sum(weights[i] for i in selected_indices)
        
    def is_valid_addition(selected_set, v_idx):
        for u_idx in selected_set:
            if conflict[v_idx][u_idx]:
                return False
        return True
        
    def local_search(selected_set):
        improved = True
        while improved:
            improved = False
            
            # 1. Try Add
            best_add_gain = 0
            best_add_idx = None
            for v_idx in range(n_nodes):
                if v_idx not in selected_set:
                    if is_valid_addition(selected_set, v_idx):
                        gain = weights[v_idx]
                        if gain > best_add_gain:
                            best_add_gain = gain
                            best_add_idx = v_idx
                            
            if best_add_idx is not None:
                selected_set.add(best_add_idx)
                improved = True
                continue
                
            # 2. Try Swap (remove u, add v)
            best_swap_gain = 0
            best_swap_u = None
            best_swap_v = None
            
            for u_idx in list(selected_set):
                for v_idx in range(n_nodes):
                    if v_idx in selected_set or v_idx == u_idx:
                        continue
                    compatible = True
                    for w_idx in selected_set:
                        if w_idx == u_idx:
                            continue
                        if conflict[v_idx][w_idx]:
                            compatible = False
                            break
                    if compatible:
                        gain = weights[v_idx] - weights[u_idx]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap_u = u_idx
                            best_swap_v = v_idx
                            
            if best_swap_u is not None:
                selected_set.remove(best_swap_u)
                selected_set.add(best_swap_v)
                improved = True
                continue
                
            # 3. Try 1-for-2
            best_1for2_gain = 0
            best_1for2_u = None
            best_1for2_v1 = None
            best_1for2_v2 = None
            
            for u_idx in list(selected_set):
                S_minus_u = selected_set - {u_idx}
                candidates = []
                for v_idx in range(n_nodes):
                    if v_idx not in selected_set and is_valid_addition(S_minus_u, v_idx):
                        candidates.append(v_idx)
                        
                for i_cand in range(len(candidates)):
                    v1 = candidates[i_cand]
                    for j_cand in range(i_cand + 1, len(candidates)):
                        v2 = candidates[j_cand]
                        if not conflict[v1][v2]:
                            gain = weights[v1] + weights[v2] - weights[u_idx]
                            if gain > best_1for2_gain:
                                best_1for2_gain = gain
                                best_1for2_u = u_idx
                                best_1for2_v1 = v1
                                best_1for2_v2 = v2
                                
            if best_1for2_u is not None:
                selected_set.remove(best_1for2_u)
                selected_set.add(best_1for2_v1)
                selected_set.add(best_1for2_v2)
                improved = True
                continue
                
            # 4. Try 2-for-1
            best_2for1_gain = 0
            best_2for1_u1 = None
            best_2for1_u2 = None
            best_2for1_v = None
            
            if len(selected_set) > 1:
                u_list = list(selected_set)
                for i_u1 in range(len(u_list)):
                    u1 = u_list[i_u1]
                    for i_u2 in range(i_u1 + 1, len(u_list)):
                        u2 = u_list[i_u2]
                        S_minus_u1u2 = selected_set - {u1, u2}
                        for v_idx in range(n_nodes):
                            if v_idx not in selected_set and is_valid_addition(S_minus_u1u2, v_idx):
                                gain = weights[v_idx] - weights[u1] - weights[u2]
                                if gain > best_2for1_gain:
                                    best_2for1_gain = gain
                                    best_2for1_u1 = u1
                                    best_2for1_u2 = u2
                                    best_2for1_v = v_idx
                                    
            if best_2for1_u1 is not None:
                selected_set.remove(best_2for1_u1)
                selected_set.remove(best_2for1_u2)
                selected_set.add(best_2for1_v)
                improved = True
                continue
                
        return selected_set

    best_solution = set()
    best_score = 0.0
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            selected_set = set()
            sorted_indices = sorted(range(n_nodes), key=lambda i: weights[i], reverse=True)
            for idx in sorted_indices:
                if is_valid_addition(selected_set, idx):
                    selected_set.add(idx)
        elif r == 1:
            selected_set = set()
            for i in range(n_nodes):
                if nodes[i] > N // 2:
                    selected_set.add(i)
        else:
            selected_set = set()
            indices_to_try = list(range(n_nodes))
            random.shuffle(indices_to_try)
            for idx in indices_to_try:
                if is_valid_addition(selected_set, idx):
                    selected_set.add(idx)
        
        optimized_set = local_search(selected_set)
        score = get_sum(optimized_set)
        if score > best_score:
            best_score = score
            best_solution = optimized_set
            
    result = [nodes[i] for i in best_solution]
    if not result:
        return [N]
    return result
# EVOLVE-BLOCK-END
