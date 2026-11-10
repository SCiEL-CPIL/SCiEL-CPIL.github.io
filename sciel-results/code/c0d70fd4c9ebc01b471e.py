import time
import random

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85.0
    
    # Precompute 3-APs
    aps_for = [[] for _ in range(N + 1)]
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            aps_for[a].append((b, c))
            aps_for[b].append((a, c))
            aps_for[c].append((a, b))
            
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    def greedy_forward():
        in_set = [False] * (N + 1)
        S = set()
        for x in range(1, N + 1):
            valid = True
            for u, v in aps_for[x]:
                if in_set[u] and in_set[v]:
                    valid = False
                    break
            if valid:
                S.add(x)
                in_set[x] = True
        return S, in_set, sum(weights[x] for x in S)
        
    S, in_set, score = greedy_forward()
    best_S = set(S)
    best_score = score
    
    def get_score(in_arr):
        # Faster than summing set if we maintain it, but summing set is fine for small N
        return sum(weights[x] for x in range(1, N+1) if in_arr[x])
        
    def try_add_repair(in_arr, x):
        # Returns (success, new_in_arr, delta_score)
        # We modify in_arr in place if successful, else return original
        
        if in_arr[x]:
            return False, in_arr, 0.0
            
        conflicts = []
        for u, v in aps_for[x]:
            if in_arr[u] and in_arr[v]:
                conflicts.append((u, v))
                
        if not conflicts:
            in_arr[x] = True
            return True, in_arr, weights[x]
            
        nodes = set()
        for u, v in conflicts:
            nodes.add(u)
            nodes.add(v)
            
        if len(nodes) > 15: # Cutoff for exact VC
            return False, in_arr, 0.0
            
        nodes_list = list(nodes)
        node_idx = {n: i for i, n in enumerate(nodes_list)}
        n_nodes = len(nodes_list)
        
        edge_nodes = []
        for u, v in conflicts:
            edge_nodes.append((node_idx[u], node_idx[v]))
            
        best_cover_weight = float('inf')
        best_cover_mask = 0 # Bitmask of selected nodes
        
        def dfs(cover_idx, current_weight, mask):
            nonlocal best_cover_weight, best_cover_mask
            if current_weight >= best_cover_weight:
                return
            if cover_idx == len(edge_nodes):
                if current_weight < best_cover_weight:
                    best_cover_weight = current_weight
                    best_cover_mask = mask
                return
                
            u_idx, v_idx = edge_nodes[cover_idx]
            if (mask >> u_idx) & 1 or (mask >> v_idx) & 1:
                dfs(cover_idx + 1, current_weight, mask)
            else:
                # Pick u
                dfs(cover_idx + 1, current_weight + weights[nodes_list[u_idx]], mask | (1 << u_idx))
                # Pick v
                dfs(cover_idx + 1, current_weight + weights[nodes_list[v_idx]], mask | (1 << v_idx))
                
        dfs(0, 0.0, 0)
        
        if best_cover_mask == 0 and best_cover_weight != 0: # Edge case
             pass
             
        if best_cover_weight == float('inf'):
            return False, in_arr, 0.0
            
        removal_weight = best_cover_weight
        gain = weights[x] - removal_weight
        
        if gain > 0:
            # Apply changes
            for i in range(n_nodes):
                if (best_cover_mask >> i) & 1:
                    in_arr[nodes_list[i]] = False
            in_arr[x] = True
            return True, in_arr, gain
            
        return False, in_arr, 0.0

    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        candidates = [x for x in range(1, N + 1) if not in_set[x]]
        if not candidates:
            break
            
        x = random.choice(candidates)
        
        # Save state to rollback if needed? 
        # try_add_repair modifies in_arr in place on success.
        # We need to rollback on failure? No, it returns original on failure.
        # But if it succeeds, we accept.
        
        old_score = score
        success, new_in, delta = try_add_repair(in_set, x)
        
        if success:
            score += delta
            in_set = new_in
            no_improve_count = 0
            if score > best_score:
                best_score = score
                best_S = set(x for x in range(1, N+1) if in_set[x])
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            # Perturb
            to_remove = random.sample([x for x in range(1, N+1) if in_set[x]], min(5, sum(in_set)))
            for y in to_remove:
                in_set[y] = False
                score -= weights[y]
            no_improve_count = 0
            max_no_improve = max(100, max_no_improve // 2)
            
        iteration += 1

    return list(best_S)
