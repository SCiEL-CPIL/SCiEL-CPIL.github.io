"""Optimized solver for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
def construct():
    from math import lcm
    from fractions import Fraction
    import sys

    N = 256
    nums = list(range(2, 257))
    n = len(nums)  # 255

    # Build conflict graph (complement of the desired clique graph)
    # Edge if lcm(a, b) <= 256
    comp_adj = [0] * n
    for i in range(n):
        ai = nums[i]
        for j in range(i + 1, n):
            bj = nums[j]
            if lcm(ai, bj) <= N:
                comp_adj[i] |= (1 << j)
                comp_adj[j] |= (1 << i)

    weights = [Fraction(1, nums[i]) for i in range(n)]

    # Find connected components in the conflict graph
    visited = [False] * n
    components = []
    
    for i in range(n):
        if not visited[i]:
            # BFS to find component
            stack = [i]
            visited[i] = True
            comp_nodes = []
            while stack:
                u = stack.pop()
                comp_nodes.append(u)
                neighbors = comp_adj[u]
                for v in range(n):
                    if (neighbors >> v) & 1:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
            components.append(comp_nodes)

    total_best_weight = Fraction(0, 1)
    best_clique_nodes = []

    for comp in components:
        k = len(comp)
        if k == 0:
            continue
        
        # If component is a single node, it's always in the independent set
        if k == 1:
            total_best_weight += weights[comp[0]]
            best_clique_nodes.append(comp[0])
            continue

        # Map global indices to local indices
        local_to_global = comp
        local_n = k
        
        # Build local adjacency bitsets
        local_adj = [0] * local_n
        for i in range(local_n):
            gi = local_to_global[i]
            for j in range(i + 1, local_n):
                gj = local_to_global[j]
                if comp_adj[gi] & (1 << gj):
                    local_adj[i] |= (1 << j)
                    local_adj[j] |= (1 << i)
        
        local_weights = [weights[local_to_global[i]] for i in range(local_n)]
        
        # Solve MWIS on local graph
        best_w = Fraction(0, 1)
        best_set = 0
        
        # Greedy initial solution
        def greedy_mwis(cands_bitset):
            res_w = Fraction(0, 1)
            res_set = 0
            tmp = cands_bitset
            while tmp:
                # Pick highest weight node
                best_idx = -1
                best_w_val = Fraction(0, 1)
                t = tmp
                while t:
                    lsb = t & (-t)
                    idx = lsb.bit_length() - 1
                    if local_weights[idx] > best_w_val:
                        best_w_val = local_weights[idx]
                        best_idx = idx
                    t ^= lsb
                if best_idx < 0:
                    break
                res_w += local_weights[best_idx]
                res_set |= (1 << best_idx)
                # Remove node and neighbors
                tmp &= ~(1 << best_idx)
                tmp &= ~local_adj[best_idx]
            return res_w, res_set

        init_w, init_set = greedy_mwis((1 << local_n) - 1)
        best_w = init_w
        best_set = init_set

        def solve_mwis(cands_bitset, current_w, current_set):
            nonlocal best_w, best_set
            
            # Reduction: Isolated nodes must be included
            # A node is isolated if it has no neighbors in cands_bitset
            # Check if (cands_bitset & local_adj[v]) == 0
            # If so, include it.
            
            # To avoid O(N^2) check every time, we can do this loop
            changed = True
            while changed:
                changed = False
                tmp = cands_bitset
                while tmp:
                    lsb = tmp & (-tmp)
                    v = lsb.bit_length() - 1
                    # Check if v is isolated in the current candidate set
                    if not (cands_bitset & local_adj[v]):
                        # Include v
                        current_w += local_weights[v]
                        current_set |= (1 << v)
                        # Remove v and its neighbors (neighbors are empty in cands, so just remove v)
                        cands_bitset &= ~(1 << v)
                        changed = True
                        # Update best if improved
                        if current_w + sum(local_weights[i] for i in range(local_n) if (cands_bitset >> i) & 1) > best_w:
                             # This sum is expensive. Just update best_w at the end or with a loose bound
                             pass
                    tmp ^= lsb
            
            if not cands_bitset:
                if current_w > best_w:
                    best_w = current_w
                    best_set = current_set
                return
            
            # Bound: current_w + sum of weights of all candidates
            bound = current_w
            tmp = cands_bitset
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                bound += local_weights[idx]
                tmp ^= lsb
            
            if bound <= best_w:
                return
            
            # Branch
            # Pick a node with high degree or high weight
            # Let's pick the node with the highest weight among candidates
            cands = []
            tmp = cands_bitset
            while tmp:
                lsb = tmp & (-tmp)
                idx = lsb.bit_length() - 1
                cands.append(idx)
                tmp ^= lsb
            
            cands.sort(key=lambda x: local_weights[x], reverse=True)
            v = cands[0]
            
            # Include v
            new_cands = cands_bitset & ~(1 << v) & ~local_adj[v]
            new_w = current_w + local_weights[v]
            new_set = current_set | (1 << v)
            solve_mwis(new_cands, new_w, new_set)
            
            # Exclude v
            new_cands2 = cands_bitset & ~(1 << v)
            solve_mwis(new_cands2, current_w, current_set)

        solve_mwis((1 << local_n) - 1, Fraction(0, 1), 0)
        
        total_best_weight += best_w
        for i in range(local_n):
            if best_set & (1 << i):
                best_clique_nodes.append(local_to_global[i])

    result = [nums[i] for i in best_clique_nodes]
    return result
# EVOLVE-BLOCK-END
