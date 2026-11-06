import time
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit
    
    N = 256
    all_nodes = list(range(2, N + 1))
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # Precompute conflict masks: conflict if lcm(a,b) <= 256
    conflict_mask = [0] * n_total
    for i in range(n_total):
        a = all_nodes[i]
        mask = 0
        for j in range(n_total):
            if i == j:
                continue
            b = all_nodes[j]
            if a * b <= 256 * gcd(a, b):
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    conflict_count = [bin(conflict_mask[i]).count('1') for i in range(n_total)]
    
    # Weights: 1/v
    weights = [1.0 / v for v in all_nodes]
    
    # Helper to check validity and get value
    def get_value(selected):
        return sum(weights[node_to_idx[v]] for v in selected)
    
    def is_valid(selected):
        # Check pairwise lcm > 256
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                a, b = selected[i], selected[j]
                if a * b <= 256 * gcd(a, b):
                    return False
        return True

    # Multi-start Greedy Initialization
    def greedy_init(order):
        sel = []
        used = 0
        for idx in order:
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            used |= (1 << idx)
        return sel

    # Local Search: 1-swap, 2-swap, 3-swap
    def local_search(selected, deadline):
        selected = selected[:]
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])
        
        improved = True
        while improved and (time.time() - start_time) < deadline:
            improved = False
            
            # 1. Try adding nodes (if any fit)
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                used |= (1 << i)
                improved = True
            
            # 2. Try 1-swap (replace one with a better one)
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                s_i = node_to_idx[s]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = weights[n_i] - weights[s_i]
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break
            
            # 3. Try 2-swap
            if not improved and len(selected) >= 2:
                n_sel = len(selected)
                for i in range(n_sel):
                    if (time.time() - start_time) > deadline:
                        break
                    for j in range(i + 1, n_sel):
                        if (time.time() - start_time) > deadline:
                            break
                        si = node_to_idx[selected[i]]
                        sj = node_to_idx[selected[j]]
                        used_without = used & ~(1 << si) & ~(1 << sj)
                        removed_sum = weights[si] + weights[sj]
                        
                        best_pair = None
                        best_gain = 0
                        non_sel_list = [k for k in range(n_total) if not ((used_without >> k) & 1)]
                        
                        for a_idx in range(len(non_sel_list)):
                            if (time.time() - start_time) > deadline:
                                break
                            na = non_sel_list[a_idx]
                            if used_without & conflict_mask[na]:
                                continue
                            for b_idx in range(a_idx + 1, len(non_sel_list)):
                                nb = non_sel_list[b_idx]
                                if (used_without | (1 << na)) & conflict_mask[nb]:
                                    continue
                                gain = (weights[na] + weights[nb]) - removed_sum
                                if gain > best_gain:
                                    best_gain = gain
                                    best_pair = (na, nb)
                        
                        if best_pair is not None:
                            na, nb = best_pair
                            selected[i] = all_nodes[na]
                            selected[j] = all_nodes[nb]
                            used = used_without | (1 << na) | (1 << nb)
                            improved = True
                            break
                    if improved:
                        break

            # 4. Try 3-swap (remove 2, add 3 OR remove 3, add 2)
            # This is expensive, so only do it if we have time and no other improvement
            if not improved and (time.time() - start_time) < deadline - 5:
                # Remove 2, Add 3
                n_sel = len(selected)
                if n_sel >= 2:
                    for i in range(n_sel):
                        if (time.time() - start_time) > deadline:
                            break
                        for j in range(i + 1, n_sel):
                            if (time.time() - start_time) > deadline:
                                break
                            si = node_to_idx[selected[i]]
                            sj = node_to_idx[selected[j]]
                            used_without = used & ~(1 << si) & ~(1 << sj)
                            removed_sum = weights[si] + weights[sj]
                            
                            best_triple = None
                            best_gain = 0
                            non_sel_list = [k for k in range(n_total) if not ((used_without >> k) & 1)]
                            
                            # Optimized 3-swap search
                            for a_idx in range(len(non_sel_list)):
                                if (time.time() - start_time) > deadline:
                                    break
                                na = non_sel_list[a_idx]
                                if used_without & conflict_mask[na]:
                                    continue
                                for b_idx in range(a_idx + 1, len(non_sel_list)):
                                    nb = non_sel_list[b_idx]
                                    if (used_without | (1 << na)) & conflict_mask[nb]:
                                        continue
                                    for c_idx in range(b_idx + 1, len(non_sel_list)):
                                        nc = non_sel_list[c_idx]
                                        if (used_without | (1 << na) | (1 << nb)) & conflict_mask[nc]:
                                            continue
                                        gain = (weights[na] + weights[nb] + weights[nc]) - removed_sum
                                        if gain > best_gain:
                                            best_gain = gain
                                            best_triple = (na, nb, nc)
                            
                            if best_triple is not None:
                                na, nb, nc = best_triple
                                # Replace selected[i], selected[j] with na, nb, nc
                                # We need to append nc
                                selected[i] = all_nodes[na]
                                selected[j] = all_nodes[nb]
                                selected.append(all_nodes[nc])
                                used = used_without | (1 << na) | (1 << nb) | (1 << nc)
                                improved = True
                                break
                        if improved:
                            break

        return selected

    # Iterated Local Search
    best_subset = []
    best_value = 0.0
    
    # Initial Heuristic
    scores = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    order1 = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    scores2 = [(1.0 / all_nodes[i]) / max(1, conflict_count[i]) for i in range(n_total)]
    order2 = sorted(range(n_total), key=lambda i: scores2[i], reverse=True)
    
    # Run ILS for a portion of the time
    il_deadline = start_time + 40.0
    
    # Start with a few good heuristics
    candidates = []
    for order in [order1, order2]:
        sol = greedy_init(order)
        sol = local_search(sol, il_deadline)
        val = get_value(sol)
        if val > best_value:
            best_value = val
            best_subset = sol[:]
    
    # Random Restarts
    while (time.time() - start_time) < il_deadline:
        # Perturb the best solution
        current = best_subset[:]
        
        # Random perturbation: remove 1-3 random elements, add 1-3 random non-conflicting
        k_remove = random.randint(1, 3)
        k_remove = min(k_remove, len(current))
        to_remove = random.sample(current, k_remove)
        for v in to_remove:
            current.remove(v)
        
        # Add random non-conflicting elements
        used = 0
        for v in current:
            used |= (1 << node_to_idx[v])
        
        k_add = random.randint(1, 3)
        attempts = 0
        while k_add > 0 and attempts < 50:
            attempts += 1
            idx = random.randint(0, n_total - 1)
            if not ((used >> idx) & 1) and not (used & conflict_mask[idx]):
                current.append(all_nodes[idx])
                used |= (1 << idx)
                k_add -= 1
        
        # Local Search on perturbed solution
        polished = local_search(current, il_deadline)
        val = get_value(polished)
        
        if val > best_value:
            best_value = val
            best_subset = polished[:]

    # Final Polish on the best found
    final_deadline = start_time + 75.0
    polished = local_search(best_subset, final_deadline)
    val = get_value(polished)
    if val > best_value:
        best_value = val
        best_subset = polished[:]

    # Try MILP with the best solution as start
    n_vars = n_total
    c = -np.array([1.0 / x for x in all_nodes])
    
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    for i in range(n_vars):
        mask = conflict_mask[i]
        j = i + 1
        while j < n_vars:
            if (mask >> j) & 1:
                row_indices.append(constraint_idx)
                col_indices.append(i)
                data.append(1)
                row_indices.append(constraint_idx)
                col_indices.append(j)
                data.append(1)
                constraint_idx += 1
            j += 1
    
    num_constraints = constraint_idx
    
    if num_constraints > 0 and (time.time() - start_time) < 80.0:
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_vars))
        lb = np.zeros(num_constraints)
        ub = np.ones(num_constraints)
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        
        x0 = np.zeros(n_vars)
        for v in best_subset:
            x0[node_to_idx[v]] = 1.0
        
        remaining = 85.0 - (time.time() - start_time)
        if remaining > 1:
            try:
                res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                           options={"time_limit": max(1, remaining), "mip_gap": 0.0, "presolve": True, "threads": 4})
                
                if res.x is not None:
                    milp_subset = [all_nodes[i] for i in range(n_vars) if res.x[i] > 0.5]
                    if milp_subset:
                        milp_value = sum(1.0 / v for v in milp_subset)
                        if milp_value > best_value:
                            best_subset = milp_subset
                            best_value = milp_value
            except Exception:
                pass

    # Final check
    if not best_subset:
        return [256]
        
    return best_subset

