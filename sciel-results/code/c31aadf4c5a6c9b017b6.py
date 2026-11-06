import time
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
import random

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit
    
    # Nodes are 2..256
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Weights: We want to maximize sum(1/x). MILP minimizes.
    # c_i = -1 / nodes[i]
    c = -np.array([1.0 / x for x in nodes])
    
    # Build Conflict Graph
    # Edge (i, j) exists if lcm(nodes[i], nodes[j]) <= 256
    # <=> nodes[i] * nodes[j] <= 256 * gcd(nodes[i], nodes[j])
    
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    # Optimization: Only check pairs where i < j
    # We can potentially prune if nodes[i] * nodes[j] is huge, 
    # but gcd calculation is cheap enough for 255^2/2 ~ 32k pairs.
    
    for i in range(n_vars):
        a = nodes[i]
        for j in range(i + 1, n_vars):
            b = nodes[j]
            # Check conflict
            if a * b <= 256 * gcd(a, b):
                # Add constraint x_i + x_j <= 1
                row_indices.append(constraint_idx)
                col_indices.append(i)
                data.append(1)
                
                row_indices.append(constraint_idx)
                col_indices.append(j)
                data.append(1)
                
                constraint_idx += 1
                
    num_constraints = constraint_idx
    
    if num_constraints == 0:
        # No conflicts, take all
        return nodes

    # Create sparse matrix A for Ax <= 1
    # Shape: (num_constraints, n_vars)
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_vars))
    
    # Constraints: 0 <= A x <= 1
    lb = np.zeros(num_constraints)
    ub = np.ones(num_constraints)
    
    constraint = LinearConstraint(A, lb, ub)
    
    # Bounds for variables: 0 <= x <= 1
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    
    # Integrality: 1 means integer
    integrality = np.ones(n_vars)
    
    best_subset = None
    best_value = -1.0
    
    # Try to solve with MILP
    try:
        # options can be tuned, but defaults are usually fine for this size
        # time_limit is not directly exposed in older scipy, but we have our own check
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0:  # Optimal
            x = res.x
            subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
            if subset:
                best_subset = subset
                best_value = sum(1.0 / v for v in subset)
        elif res.status in [1, 2]: # Locally optimal or infeasible? status 1 is locally optimal, 2 is infeasible?
            # SciPy status: 0 optimal, 1 locally optimal, 2 infeasible, 3 unbounded, 4 numerical failure, 5 failed to converge, 6 interrupted, 7 failed to solve
            if res.x is not None:
                 x = res.x
                 subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
                 if subset:
                     best_subset = subset
                     best_value = sum(1.0 / v for v in subset)
                     
    except Exception:
        pass
    
    # If MILP failed or didn't provide a good solution, use a heuristic fallback
    # Heuristic: Greedy + Local Search (1-swap)
    
    # If MILP didn't give us a great result, use enhanced local search
    if best_subset is None or (time.time() - start_time > 3 and best_value < 1.14):
        # Precompute conflict graph as bitsets for fast checking
        n_nodes = len(nodes)
        node_to_idx = {v: i for i, v in enumerate(nodes)}
        
        # Build adjacency: conflict[i] is a set of indices j where nodes[i] and nodes[j] conflict
        conflict = [set() for _ in range(n_nodes)]
        for i in range(n_nodes):
            a = nodes[i]
            for j in range(i + 1, n_nodes):
                b = nodes[j]
                if a * b <= 256 * gcd(a, b):
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        def is_valid_addition(selected_set_indices, node_idx):
            """Check if node_idx can be added to selected_set_indices"""
            for s_idx in selected_set_indices:
                if s_idx in conflict[node_idx]:
                    return False
            return True
        
        def greedy_construct(order):
            """Greedy construction given an ordering of node indices"""
            selected = []
            selected_set = set()
            for node_idx in order:
                if is_valid_addition(selected_set, node_idx):
                    selected.append(node_idx)
                    selected_set.add(node_idx)
            return selected, selected_set
        
        def local_search(selected, selected_set, time_limit):
            """Enhanced local search with 1-swap, 2-swap, and add moves"""
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                
                # 1. Try adding nodes
                for node_idx in range(n_nodes):
                    if node_idx in selected_set:
                        continue
                    if is_valid_addition(selected_set, node_idx):
                        selected.append(node_idx)
                        selected_set.add(node_idx)
                        improved = True
                
                # 2. Try 1-swap: remove s, add n
                non_selected = [i for i in range(n_nodes) if i not in selected_set]
                for s_idx_pos, s_idx in enumerate(selected):
                    s_val = nodes[s_idx]
                    for n_idx in non_selected:
                        n_val = nodes[n_idx]
                        if n_val >= s_val:
                            continue  # No value improvement
                        # Check if n_idx is compatible with (selected \ {s_idx})
                        ok = True
                        for k, other_idx in enumerate(selected):
                            if k == s_idx_pos:
                                continue
                            if other_idx in conflict[n_idx]:
                                ok = False
                                break
                        if ok:
                            selected[s_idx_pos] = n_idx
                            selected_set.remove(s_idx)
                            selected_set.add(n_idx)
                            improved = True
                            break
                
                # 3. Try 2-swap: remove s1, s2, add n1, n2
                if (time.time() - start_time) < time_limit - 5:
                    non_selected = [i for i in range(n_nodes) if i not in selected_set]
                    for s1_pos in range(len(selected)):
                        s1_idx = selected[s1_pos]
                        for s2_pos in range(s1_pos + 1, len(selected)):
                            s2_idx = selected[s2_pos]
                            s1_val = nodes[s1_idx]
                            s2_val = nodes[s2_idx]
                            for n1_idx in non_selected:
                                n1_val = nodes[n1_idx]
                                for n2_idx in non_selected:
                                    if n2_idx == n1_idx:
                                        continue
                                    n2_val = nodes[n2_idx]
                                    # Value improvement check
                                    if (1.0/n1_val + 1.0/n2_val) <= (1.0/s1_val + 1.0/s2_val):
                                        continue
                                    # Check compatibility
                                    ok = True
                                    # n1 must be compatible with selected \ {s1, s2}
                                    for k, other_idx in enumerate(selected):
                                        if k == s1_pos or k == s2_pos:
                                            continue
                                        if other_idx in conflict[n1_idx] or other_idx in conflict[n2_idx]:
                                            ok = False
                                            break
                                    if not ok:
                                        continue
                                    # n1 and n2 must not conflict
                                    if n2_idx in conflict[n1_idx]:
                                        continue
                                    # Perform swap
                                    new_selected = []
                                    for k, idx in enumerate(selected):
                                        if k != s1_pos and k != s2_pos:
                                            new_selected.append(idx)
                                    new_selected.append(n1_idx)
                                    new_selected.append(n2_idx)
                                    selected[:] = new_selected
                                    selected_set = set(selected)
                                    improved = True
                                    break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                            
            return selected, selected_set
        
        # Multiple restarts with different orderings
        best_local_subset = None
        best_local_value = -1.0
        
        # Restart 1: Greedy by ascending value (prefer small numbers)
        order1 = list(range(n_nodes))  # nodes are already sorted ascending
        sel1, set1 = greedy_construct(order1)
        sel1, set1 = local_search(sel1, set1, deadline - 3)
        val1 = sum(1.0 / nodes[i] for i in sel1)
        if val1 > best_local_value:
            best_local_value = val1
            best_local_subset = [nodes[i] for i in sel1]
        
        # Restart 2: Random ordering
        if (time.time() - start_time) < deadline - 5:
            random.seed(42)
            order2 = list(range(n_nodes))
            random.shuffle(order2)
            sel2, set2 = greedy_construct(order2)
            sel2, set2 = local_search(sel2, set2, deadline - 3)
            val2 = sum(1.0 / nodes[i] for i in sel2)
            if val2 > best_local_value:
                best_local_value = val2
                best_local_subset = [nodes[i] for i in sel2]
        
        # Restart 3: Greedy by descending value (prefer large numbers)
        if (time.time() - start_time) < deadline - 5:
            order3 = list(range(n_nodes - 1, -1, -1))
            sel3, set3 = greedy_construct(order3)
            sel3, set3 = local_search(sel3, set3, deadline - 3)
            val3 = sum(1.0 / nodes[i] for i in sel3)
            if val3 > best_local_value:
                best_local_value = val3
                best_local_subset = [nodes[i] for i in sel3]
        
        # Final polish: try to improve best_local_subset with more aggressive local search
        if best_local_subset is not None and (time.time() - start_time) < deadline - 2:
            # Convert back to indices
            best_idx_set = set(node_to_idx[v] for v in best_local_subset)
            best_idx_list = [node_to_idx[v] for v in best_local_subset]
            best_idx_list, best_idx_set = local_search(best_idx_list, best_idx_set, deadline - 1)
            final_val = sum(1.0 / nodes[i] for i in best_idx_list)
            if final_val > best_local_value:
                best_local_value = final_val
                best_local_subset = [nodes[i] for i in best_idx_list]
        
        if best_local_subset is not None:
            if best_subset is None or best_local_value > best_value:
                best_subset = best_local_subset
                best_value = best_local_value
    
    if best_subset is None:
        # Absolute fallback: largest element (256) is always valid alone
        return [256]
    
    # Final verification and cleanup
    # Ensure all values are in [2, 256] and pairwise lcm > 256
    verified = []
    verified_set = set()
    for v in best_subset:
        if v < 2 or v > 256:
            continue
        ok = True
        for u in verified:
            if v * u <= 256 * gcd(v, u):
                ok = False
                break
        if ok:
            verified.append(v)
            verified_set.add(v)
    
    if not verified:
        return [256]
    
    return verified
