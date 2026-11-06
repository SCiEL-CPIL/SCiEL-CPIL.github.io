"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import numpy as np
    from math import lcm, gcd
    from scipy.optimize import milp, LinearConstraint, Bounds
    import scipy.sparse as sp

    start_time = time.time()
    deadline = 75.0  # Safety margin (90s limit)

    nodes = list(range(2, 257))
    n = len(nodes)
    weights = np.array([1.0 / x for x in nodes])
    
    # Precompute conflict bitsets for fast compatibility checks
    # conflict_mask[i] is a bitset of nodes that conflict with node i
    conflict_mask = [0] * n
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    for i in range(n):
        ai = nodes[i]
        for j in range(i + 1, n):
            aj = nodes[j]
            if lcm(ai, aj) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Build edge list for MILP
    edges = []
    for i in range(n):
        mask = conflict_mask[i]
        j = i + 1
        while j < n:
            if (mask >> j) & 1:
                edges.append((i, j))
            j += 1
    m = len(edges)
    
    # Greedy initialization: sort by weight descending
    greedy_selected = []
    greedy_indices = []
    for idx in sorted(range(n), key=lambda i: -weights[i]):
        # Check compatibility using bitset
        compatible = True
        for gidx in greedy_indices:
            if (conflict_mask[idx] >> gidx) & 1:
                compatible = False
                break
        if compatible:
            greedy_selected.append(nodes[idx])
            greedy_indices.append(idx)
            
    if not greedy_selected:
        greedy_selected = [256]
        greedy_indices = [node_to_idx[256]]
        
    best_score = sum(weights[i] for i in greedy_indices)
    best_solution = greedy_selected
    best_indices = set(greedy_indices)

    # Try MILP
    try:
        if time.time() - start_time > deadline:
            return best_solution
            
        c = -weights
        
        if m == 0:
            return nodes

        # Use sparse matrix for constraints
        row_indices = []
        col_indices = []
        data = []
        for k, (i, j) in enumerate(edges):
            row_indices.append(k)
            col_indices.append(i)
            data.append(1)
            row_indices.append(k)
            col_indices.append(j)
            data.append(1)
            
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
        
        cons = LinearConstraint(A, lb=0, ub=1)
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds)
        
        if res.success:
            x = res.x
            selected_indices = [i for i in range(n) if x[i] > 0.5]
            
            # Verify using bitsets
            valid = True
            for i in range(len(selected_indices)):
                for j in range(i + 1, len(selected_indices)):
                    if (conflict_mask[selected_indices[i]] >> selected_indices[j]) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and len(selected_indices) > 0:
                score = sum(weights[i] for i in selected_indices)
                if score > best_score:
                    best_score = score
                    best_solution = [nodes[i] for i in selected_indices]
                    best_indices = set(selected_indices)
                
    except Exception:
        pass

    # Enhanced Local Search with bitset-based compatibility
    current_indices = set(best_indices)
    current_score = best_score
    
    def is_compatible(idx, current_set):
        """Check if idx is compatible with all nodes in current_set using bitset."""
        mask = conflict_mask[idx]
        for cidx in current_set:
            if (mask >> cidx) & 1:
                return False
        return True
    
    improved = True
    while improved and (time.time() - start_time < deadline - 5.0):
        improved = False
        
        # Phase 1: Try adding a single node
        for idx in range(n):
            if idx in current_indices:
                continue
            if is_compatible(idx, current_indices):
                current_indices.add(idx)
                current_score += weights[idx]
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 1-for-1 swaps
        for rem in list(current_indices):
            for add in range(n):
                if add in current_indices:
                    continue
                # Check if add is compatible with all current nodes except rem
                compatible = True
                mask = conflict_mask[add]
                for cidx in current_indices:
                    if cidx == rem:
                        continue
                    if (mask >> cidx) & 1:
                        compatible = False
                        break
                if compatible:
                    delta = weights[add] - weights[rem]
                    if delta > 1e-12:
                        current_indices.discard(rem)
                        current_indices.add(add)
                        current_score += delta
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 3: Try 2-for-1 swaps (remove two, add one)
        for rem1 in list(current_indices):
            for rem2 in list(current_indices):
                if rem1 >= rem2:
                    continue
                for add in range(n):
                    if add in current_indices:
                        continue
                    # Check if add is compatible with all current nodes except rem1, rem2
                    compatible = True
                    mask = conflict_mask[add]
                    for cidx in current_indices:
                        if cidx == rem1 or cidx == rem2:
                            continue
                        if (mask >> cidx) & 1:
                            compatible = False
                            break
                    if compatible:
                        delta = weights[add] - weights[rem1] - weights[rem2]
                        if delta > 1e-12:
                            current_indices.discard(rem1)
                            current_indices.discard(rem2)
                            current_indices.add(add)
                            current_score += delta
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 4: Try 1-for-2 swaps (remove one, add two)
        for rem in list(current_indices):
            for add1 in range(n):
                if add1 in current_indices:
                    continue
                # Check if add1 is compatible with all current nodes except rem
                compatible1 = True
                mask1 = conflict_mask[add1]
                for cidx in current_indices:
                    if cidx == rem:
                        continue
                    if (mask1 >> cidx) & 1:
                        compatible1 = False
                        break
                if not compatible1:
                    continue
                for add2 in range(add1 + 1, n):
                    if add2 in current_indices:
                        continue
                    # Check if add2 is compatible with all current nodes except rem, and with add1
                    compatible2 = True
                    mask2 = conflict_mask[add2]
                    if (mask2 >> add1) & 1:
                        continue
                    for cidx in current_indices:
                        if cidx == rem:
                            continue
                        if (mask2 >> cidx) & 1:
                            compatible2 = False
                            break
                    if compatible2:
                        delta = weights[add1] + weights[add2] - weights[rem]
                        if delta > 1e-12:
                            current_indices.discard(rem)
                            current_indices.add(add1)
                            current_indices.add(add2)
                            current_score += delta
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 5: Try 2-for-2 swaps
        for rem1 in list(current_indices):
            for rem2 in list(current_indices):
                if rem1 >= rem2:
                    continue
                for add1 in range(n):
                    if add1 in current_indices:
                        continue
                    compatible1 = True
                    mask1 = conflict_mask[add1]
                    for cidx in current_indices:
                        if cidx == rem1 or cidx == rem2:
                            continue
                        if (mask1 >> cidx) & 1:
                            compatible1 = False
                            break
                    if not compatible1:
                        continue
                    for add2 in range(add1 + 1, n):
                        if add2 in current_indices:
                            continue
                        compatible2 = True
                        mask2 = conflict_mask[add2]
                        if (mask2 >> add1) & 1:
                            continue
                        for cidx in current_indices:
                            if cidx == rem1 or cidx == rem2:
                                continue
                            if (mask2 >> cidx) & 1:
                                compatible2 = False
                                break
                        if compatible2:
                            delta = weights[add1] + weights[add2] - weights[rem1] - weights[rem2]
                            if delta > 1e-12:
                                current_indices.discard(rem1)
                                current_indices.discard(rem2)
                                current_indices.add(add1)
                                current_indices.add(add2)
                                current_score += delta
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # If no improvement in any phase, we're at a local optimum
        # Try random restarts with perturbation
        if time.time() - start_time < deadline - 10.0:
            # Perturb: remove a random node and re-optimize
            if len(current_indices) > 1:
                import random
                random.seed(int(time.time() * 1000) % 10000)
                rem_node = random.choice(list(current_indices))
                perturbed = current_indices - {rem_node}
                # Re-add greedily
                for idx in sorted(range(n), key=lambda i: -weights[i]):
                    if idx in perturbed:
                        continue
                    if is_compatible(idx, perturbed):
                        perturbed.add(idx)
                pert_score = sum(weights[i] for i in perturbed)
                if pert_score > current_score - 0.01:  # Allow slight degradation for exploration
                    current_indices = perturbed
                    current_score = pert_score
                    improved = True
                    break
    
    # Final verification and return
    final_list = [nodes[i] for i in current_indices]
    if not final_list:
        final_list = [256]
    
    # Verify validity
    valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            if lcm(final_list[i], final_list[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        return greedy_selected
    
    return final_list
# EVOLVE-BLOCK-END
