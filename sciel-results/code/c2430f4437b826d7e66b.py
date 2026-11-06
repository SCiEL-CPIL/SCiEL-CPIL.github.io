import time
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    N = 256
    # Weights are 1/x
    weights = [0.0] * (N + 1)
    for i in range(2, N + 1):
        weights[i] = 1.0 / i
        
    # Split into A and B
    A = list(range(2, 17))  # 2 to 16
    B = list(range(17, N + 1)) # 17 to 256
    
    # Precompute LCM compatibility for A
    # C_A[i][j] = True if lcm(A[i], A[j]) > 256
    # We need to check all pairs in A for validity of S_A
    
    # Precompute incompatibility graph for B
    # Incompatibility: lcm(a, b) <= 256
    # We will map B indices to 0..len(B)-1 for bitmasks or lists
    B_indices = {x: i for i, x in enumerate(B)}
    n_B = len(B)
    
    # Adjacency list for incompatibility in B
    # adj[i] is a set of j such that B[i] and B[j] are incompatible
    adj_B = [set() for _ in range(n_B)]
    
    for i in range(n_B):
        a = B[i]
        for j in range(i + 1, n_B):
            b = B[j]
            g = gcd(a, b)
            l = a // g * b
            if l <= 256:
                adj_B[i].add(j)
                adj_B[j].add(i)
                
    # Find connected components in the incompatibility graph of B
    visited = [False] * n_B
    components = []
    for i in range(n_B):
        if not visited[i]:
            # BFS/DFS to find component
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj_B[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(comp)
            
    # Precompute all subsets for each component to allow fast MWIS query
    # For a component C, and a mask of allowed vertices (subset of C), 
    # we want max sum of weights of an independent set.
    # Since components are small, we can precompute best[allowed_mask]
    # But allowed_mask changes based on S_A.
    # Instead, for each component, we can just brute force the independent sets
    # that are subsets of the allowed vertices.
    
    # Let's precompute the list of all independent sets for each component?
    # No, just solve on the fly for the allowed subset.
    # Given a component C (list of indices in B) and a set of allowed indices Allowed_C (subset of C),
    # we find max weight independent set in the subgraph induced by Allowed_C.
    
    def solve_component_mwis(comp, allowed_mask):
        """
        comp: list of node indices in B
        allowed_mask: bitmask of which nodes in comp are allowed (bit i corresponds to comp[i])
        Returns max weight.
        """
        # Filter comp to only allowed nodes
        allowed_nodes = []
        for i in range(len(comp)):
            if allowed_mask & (1 << i):
                allowed_nodes.append(comp[i])
                
        if not allowed_nodes:
            return 0.0
            
        # Build local adjacency for allowed_nodes
        # Map global B index to local index
        local_adj = {}
        for u in allowed_nodes:
            local_adj[u] = []
        for u in allowed_nodes:
            for v in adj_B[u]:
                if v in local_adj:
                    local_adj[u].append(v)
                    
        # Recursive B&B for small graph
        # Sort nodes by weight descending? Or degree?
        # Given small size, simple DFS is fine.
        # To speed up, we can use a greedy upper bound.
        
        # Simple DFS
        best_val = 0.0
        
        def dfs(idx, current_sum, chosen):
            nonlocal best_val
            if idx == len(allowed_nodes):
                if current_sum > best_val:
                    best_val = current_sum
                return
                
            u = allowed_nodes[idx]
            
            # Option 1: Don't pick u
            dfs(idx + 1, current_sum, chosen)
            
            # Option 2: Pick u (if compatible with all chosen)
            # Check compatibility
            can_pick = True
            for v in chosen:
                if v in local_adj[u]:
                    can_pick = False
                    break
            if can_pick:
                dfs(idx + 1, current_sum + weights[u], chosen + [u])

        # Optimization: If the graph is a matching or disjoint edges, this is fast.
        # We can also just iterate all subsets of allowed_nodes if len is small (< 15)
        if len(allowed_nodes) <= 15:
            best_val = 0.0
            for mask in range(1 << len(allowed_nodes)):
                # Check if mask is independent set
                is_independent = True
                for i in range(len(allowed_nodes)):
                    if mask & (1 << i):
                        u = allowed_nodes[i]
                        for j in range(i + 1, len(allowed_nodes)):
                            if mask & (1 << j):
                                v = allowed_nodes[j]
                                if v in local_adj[u]:
                                    is_independent = False
                                    break
                    if not is_independent:
                        break
                if is_independent:
                    val = 0.0
                    for i in range(len(allowed_nodes)):
                        if mask & (1 << i):
                            val += weights[allowed_nodes[i]]
                    if val > best_val:
                        best_val = val
            return best_val
        else:
            dfs(0, 0.0, [])
            return best_val

    best_total = 0.0
    best_S = []
    
    # Iterate over all subsets of A
    # A has 15 elements. 2^15 = 32768.
    # For each subset, check if it is a valid clique in A.
    # Precompute compatibility in A
    n_A = len(A)
    # comp_A[i][j] is True if A[i] and A[j] are compatible
    comp_A = [[True] * n_A for _ in range(n_A)]
    for i in range(n_A):
        for j in range(i + 1, n_A):
            a = A[i]
            b = A[j]
            g = gcd(a, b)
            l = a // g * b
            if l <= 256:
                comp_A[i][j] = False
                comp_A[j][i] = False
                
    # Also precompute for each a in A, the mask of B nodes incompatible with a
    # mask_B_incompat_with_A[i] is a bitmask of n_B bits
    mask_B_incompat_with_A = [0] * n_A
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            g = gcd(a, b)
            l = a // g * b
            if l <= 256:
                mask_B_incompat_with_A[i] |= (1 << j)
                
    for mask_A in range(1 << n_A):
        # Check time
        if time.time() > deadline:
            break
            
        # Check if S_A (defined by mask_A) is a valid clique
        # i.e., all pairs in S_A are compatible
        valid = True
        for i in range(n_A):
            if mask_A & (1 << i):
                for j in range(i + 1, n_A):
                    if mask_A & (1 << j):
                        if not comp_A[i][j]:
                            valid = False
                            break
            if not valid:
                break
                
        if not valid:
            continue
            
        # Calculate sum for S_A
        sum_A = 0.0
        # Calculate incompatible mask for B
        # If any a in S_A is incompatible with b in B, then b is excluded.
        # So Excluded_B is union of mask_B_incompat_with_A[i] for i in S_A.
        excluded_B = 0
        for i in range(n_A):
            if mask_A & (1 << i):
                sum_A += weights[A[i]]
                excluded_B |= mask_B_incompat_with_A[i]
                
        # Allowed B mask
        # Full mask for B is (1 << n_B) - 1
        allowed_B_mask = ((1 << n_B) - 1) ^ excluded_B
        
        # Solve MWIS for each component in B with allowed_B_mask
        sum_B = 0.0
        for comp in components:
            # Create local mask for this component
            local_mask = 0
            for k, node_idx in enumerate(comp):
                if allowed_B_mask & (1 << node_idx):
                    local_mask |= (1 << k)
            sum_B += solve_component_mwis(comp, local_mask)
            
        total_sum = sum_A + sum_B
        if total_sum > best_total:
            best_total = total_sum
            # Reconstruct best_S for return
            # We need the actual set of numbers
            S_A_set = [A[i] for i in range(n_A) if mask_A & (1 << i)]
            
            S_B_set = []
            for comp in components:
                local_mask = 0
                for k, node_idx in enumerate(comp):
                    if allowed_B_mask & (1 << node_idx):
                        local_mask |= (1 << k)
                # We need to recover the specific independent set chosen
                # This requires the solver to return the set, not just the value.
                # For now, we just track the value. 
                # To return the set, we need to modify solve_component_mwis to return the set.
                # Given the time, I will assume the value is sufficient for the score, 
                # but the task requires returning the subset.
                # I must modify the solver to return the set.
            
            # Let's restart with a solver that returns the set.
            # This might be slow to re-implement in the loop, so I'll do it properly.
            
    # The above loop calculates the max value. 
    # To return the set, I need to store the best mask_A and then reconstruct B.
    # I will refactor to store best_mask_A and best_allowed_B_mask.
    
    # Let's redo the loop storing the best masks.
    best_total = 0.0
    best_mask_A = 0
    best_allowed_B_mask = 0
    
    for mask_A in range(1 << n_A):
        if time.time() > deadline:
            break
            
        valid = True
        for i in range(n_A):
            if mask_A & (1 << i):
                for j in range(i + 1, n_A):
                    if mask_A & (1 << j):
                        if not comp_A[i][j]:
                            valid = False
                            break
            if not valid:
                break
                
        if not valid:
            continue
            
        sum_A = 0.0
        excluded_B = 0
        for i in range(n_A):
            if mask_A & (1 << i):
                sum_A += weights[A[i]]
                excluded_B |= mask_B_incompat_with_A[i]
                
        allowed_B_mask = ((1 << n_B) - 1) ^ excluded_B
        
        sum_B = 0.0
        for comp in components:
            local_mask = 0
            for k, node_idx in enumerate(comp):
                if allowed_B_mask & (1 << node_idx):
                    local_mask |= (1 << k)
            sum_B += solve_component_mwis(comp, local_mask)
            
        total_sum = sum_A + sum_B
        if total_sum > best_total:
            best_total = total_sum
            best_mask_A = mask_A
            best_allowed_B_mask = allowed_B_mask
            
    # Reconstruct the set
    final_set = []
    for i in range(n_A):
        if best_mask_A & (1 << i):
            final_set.append(A[i])
            
    # For B, we need to find the specific independent set.
    # We can run the solver again for the best_allowed_B_mask, but this time returning the set.
    # Or we can just pick the best set from the components again.
    
    for comp in components:
        local_mask = 0
        for k, node_idx in enumerate(comp):
            if best_allowed_B_mask & (1 << node_idx):
                local_mask |= (1 << k)
        
        # Find best independent set in this component with local_mask
        # Reuse the logic from solve_component_mwis but return the set
        allowed_nodes = []
        for k in range(len(comp)):
            if local_mask & (1 << k):
                allowed_nodes.append(comp[k])
                
        if not allowed_nodes:
            continue
            
        local_adj = {}
        for u in allowed_nodes:
            local_adj[u] = []
        for u in allowed_nodes:
            for v in adj_B[u]:
                if v in local_adj:
                    local_adj[u].append(v)
                    
        best_val = 0.0
        best_subset = []
        
        if len(allowed_nodes) <= 15:
            for mask in range(1 << len(allowed_nodes)):
                is_independent = True
                for i in range(len(allowed_nodes)):
                    if mask & (1 << i):
                        u = allowed_nodes[i]
                        for j in range(i + 1, len(allowed_nodes)):
                            if mask & (1 << j):
                                v = allowed_nodes[j]
                                if v in local_adj[u]:
                                    is_independent = False
                                    break
                    if not is_independent:
                        break
                if is_independent:
                    val = 0.0
                    subset = []
                    for i in range(len(allowed_nodes)):
                        if mask & (1 << i):
                            val += weights[allowed_nodes[i]]
                            subset.append(allowed_nodes[i])
                    if val > best_val:
                        best_val = val
                        best_subset = subset
        else:
            # Fallback to DFS if component is large (unlikely)
            def dfs(idx, current_sum, chosen):
                nonlocal best_val, best_subset
                if idx == len(allowed_nodes):
                    if current_sum > best_val:
                        best_val = current_sum
                        best_subset = list(chosen)
                    return
                u = allowed_nodes[idx]
                dfs(idx + 1, current_sum, chosen)
                can_pick = True
                for v in chosen:
                    if v in local_adj[u]:
                        can_pick = False
                        break
                if can_pick:
                    chosen.append(u)
                    dfs(idx + 1, current_sum + weights[u], chosen)
                    chosen.pop()
            dfs(0, 0.0, [])
            
        for idx in best_subset:
            final_set.append(B[idx])
            
    return final_set

