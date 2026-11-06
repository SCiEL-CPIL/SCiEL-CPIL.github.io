import time
from math import gcd
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin against 60s limit (evaluator kills at 90s)
    
    N = 256
    all_nodes = list(range(2, N + 1))
    
    # Number-theoretic reduction:
    # If x is selected, then no multiple kx (k>=2, kx<=256) can be selected.
    # Strategy: For each "base" number, we either select it (and exclude its multiples)
    # or exclude it (and its multiples may or may not be selected).
    # 
    # Key insight: We can identify a set of "independent" numbers where no one is
    # a multiple of another. Among these, we still have conflicts (lcm > 256 requirement).
    #
    # Let's identify which numbers are "dominated": if a | b and a < b, then selecting a
    # prevents selecting b. So we can think in terms of "maximal" chains.
    #
    # Better approach: Build the conflict graph only on a reduced set.
    # Two numbers conflict if lcm(a,b) <= 256.
    # If a | b, they always conflict (lcm = b <= 256).
    #
    # Reduction: For each number x, define its "conflict closure" as all numbers y where
    # lcm(x,y) <= 256. If we select x, we exclude all of these.
    #
    # We'll use a smarter initialization: start with high-value (small) numbers that have
    # fewer conflicts, then build up.
    
    # Precompute conflict sets using bitsets for speed
    n_total = len(all_nodes)
    node_to_idx = {v: i for i, v in enumerate(all_nodes)}
    
    # conflict_mask[i] is a bitset of all nodes that conflict with all_nodes[i]
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
    
    # Number-theoretic pruning:
    # If x divides y (x < y), they conflict. Selecting x is always at least as good as
    # selecting y alone (1/x > 1/y), and x may conflict with fewer things than y.
    # Actually not necessarily - x might conflict with more things.
    # But we can say: if x | y, we never need both. So we can fix that at most one from
    # each divisibility chain is selected.
    #
    # Let's identify "primitive" numbers: those not divisible by any smaller number in [2,256].
    # Actually, let's just use the MILP on the full problem but with a better initial solution.
    
    # Better heuristic: Greedy with lookahead
    # Sort by 1/x descending (small x first), but weight by "conflict cost"
    
    # Compute for each node: how many other nodes it conflicts with
    conflict_count = [bin(conflict_mask[i]).count('1') for i in range(n_total)]
    
    # Score: 1/x / (1 + conflict_count) - prefer high value, low conflict
    scores = [(1.0 / all_nodes[i]) / (1 + conflict_count[i]) for i in range(n_total)]
    
    # Greedy construction with scoring
    selected = []
    selected_set = set()
    used_mask = 0  # bitset of selected nodes
    
    # Try multiple orderings and keep the best
    best_heuristic_subset = []
    best_heuristic_value = 0.0
    
    # Ordering 1: by score descending
    order1 = sorted(range(n_total), key=lambda i: scores[i], reverse=True)
    # Ordering 2: by 1/x descending (original greedy)
    order2 = sorted(range(n_total), key=lambda i: 1.0 / all_nodes[i], reverse=True)
    # Ordering 3: by conflict_count ascending (fewest conflicts first)
    order3 = sorted(range(n_total), key=lambda i: conflict_count[i])
    
    for order in [order1, order2, order3]:
        sel = []
        sel_set = set()
        used = 0
        for idx in order:
            # Check if idx conflicts with any selected
            if used & conflict_mask[idx]:
                continue
            sel.append(all_nodes[idx])
            sel_set.add(all_nodes[idx])
            used |= (1 << idx)
        
        val = sum(1.0 / v for v in sel)
        if val > best_heuristic_value:
            best_heuristic_value = val
            best_heuristic_subset = sel[:]
    
    # Local search on the best heuristic solution
    selected = best_heuristic_subset[:]
    selected_set = set(selected)
    # Rebuild used mask
    used = 0
    for v in selected:
        used |= (1 << node_to_idx[v])
    
    improved = True
    while improved and (time.time() - start_time) < deadline - 10:
        improved = False
        
        # 1. Try adding nodes
        for i in range(n_total):
            if (used >> i) & 1:
                continue
            if used & conflict_mask[i]:
                continue
            # Can add
            selected.append(all_nodes[i])
            selected_set.add(all_nodes[i])
            used |= (1 << i)
            improved = True
        
        # 2. Try 1-swap: remove s, add n if improves
        non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
        
        for s_idx, s in enumerate(selected):
            s_i = node_to_idx[s]
            # Temporarily remove s
            used_without_s = used & ~(1 << s_i)
            
            # Find best n to add
            best_n = None
            best_gain = 0
            for n_i in non_selected:
                n_val = all_nodes[n_i]
                if n_val >= s:
                    continue  # no gain
                # Check if n conflicts with selected \ {s}
                if used_without_s & conflict_mask[n_i]:
                    continue
                gain = 1.0 / n_val - 1.0 / s
                if gain > best_gain:
                    best_gain = gain
                    best_n = n_i
            
            if best_n is not None:
                # Perform swap
                selected[s_idx] = all_nodes[best_n]
                selected_set.remove(s)
                selected_set.add(all_nodes[best_n])
                used = used_without_s | (1 << best_n)
                improved = True
                # Update non_selected
                non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                break  # restart swap phase
        
        # 3. Try 2-opt: remove two, add two
        if (time.time() - start_time) < deadline - 15 and not improved:
            sel_list = selected[:]
            sel_set = set(sel_list)
            non_sel = [i for i in range(n_total) if all_nodes[i] not in sel_set]
            
            found_2opt = False
            for s1_idx in range(len(sel_list)):
                if found_2opt:
                    break
                s1 = sel_list[s1_idx]
                s1_i = node_to_idx[s1]
                for s2_idx in range(s1_idx + 1, len(sel_list)):
                    if found_2opt:
                        break
                    s2 = sel_list[s2_idx]
                    s2_i = node_to_idx[s2]
                    
                    used_without = used & ~(1 << s1_i) & ~(1 << s2_i)
                    
                    # Find best pair to add
                    best_pair = None
                    best_pair_gain = 0
                    for n1_i in non_sel:
                        n1 = all_nodes[n1_i]
                        if n1 >= s1 and n1 >= s2:
                            continue
                        if used_without & conflict_mask[n1_i]:
                            continue
                        for n2_i in non_sel:
                            if n2_i <= n1_i:
                                continue
                            n2 = all_nodes[n2_i]
                            if n2 >= s1 and n2 >= s2:
                                continue
                            if used_without & conflict_mask[n2_i]:
                                continue
                            if (1 << n1_i) & conflict_mask[n2_i]:
                                continue
                            gain = (1.0/n1 + 1.0/n2) - (1.0/s1 + 1.0/s2)
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (n1_i, n2_i)
                    
                    if best_pair is not None:
                        n1_i, n2_i = best_pair
                        # Perform 2-opt
                        selected[s1_idx] = all_nodes[n1_i]
                        selected[s2_idx] = all_nodes[n2_i]
                        selected_set.remove(s1)
                        selected_set.remove(s2)
                        selected_set.add(all_nodes[n1_i])
                        selected_set.add(all_nodes[n2_i])
                        used = used_without | (1 << n1_i) | (1 << n2_i)
                        improved = True
                        found_2opt = True
                        break
        
        if (time.time() - start_time) > deadline - 10:
            break
    
    best_subset = selected
    best_value = sum(1.0 / v for v in selected)
    
    # Now try MILP with the heuristic solution as initial point
    # Build conflict matrix for MILP
    n_vars = n_total
    c = -np.array([1.0 / x for x in all_nodes])
    
    row_indices = []
    col_indices = []
    data = []
    constraint_idx = 0
    
    for i in range(n_vars):
        mask = conflict_mask[i]
        # Get all j > i that conflict with i
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
    
    if num_constraints > 0:
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, n_vars))
        lb = np.zeros(num_constraints)
        ub = np.ones(num_constraints)
        constraint = LinearConstraint(A, lb, ub)
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        integrality = np.ones(n_vars)
        
        # Set initial solution from heuristic
        x0 = np.zeros(n_vars)
        for v in best_subset:
            x0[node_to_idx[v]] = 1.0
        
        try:
            res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds,
                       options={"time_limit": max(1, deadline - (time.time() - start_time) - 5)})
            
            if res.x is not None:
                milp_subset = [all_nodes[i] for i in range(n_vars) if res.x[i] > 0.5]
                if milp_subset:
                    milp_value = sum(1.0 / v for v in milp_subset)
                    if milp_value > best_value:
                        best_subset = milp_subset
                        best_value = milp_value
        except Exception:
            pass
    
    # Final polish: local search on the best solution
    if (time.time() - start_time) < deadline - 3:
        selected = best_subset[:]
        selected_set = set(selected)
        used = 0
        for v in selected:
            used |= (1 << node_to_idx[v])
        
        improved = True
        while improved and (time.time() - start_time) < deadline - 3:
            improved = False
            # Add
            for i in range(n_total):
                if (used >> i) & 1:
                    continue
                if used & conflict_mask[i]:
                    continue
                selected.append(all_nodes[i])
                selected_set.add(all_nodes[i])
                used |= (1 << i)
                improved = True
            
            # Swap
            non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
            for s_idx, s in enumerate(selected):
                s_i = node_to_idx[s]
                used_without_s = used & ~(1 << s_i)
                best_n = None
                best_gain = 0
                for n_i in non_selected:
                    n_val = all_nodes[n_i]
                    if n_val >= s:
                        continue
                    if used_without_s & conflict_mask[n_i]:
                        continue
                    gain = 1.0 / n_val - 1.0 / s
                    if gain > best_gain:
                        best_gain = gain
                        best_n = n_i
                if best_n is not None:
                    selected[s_idx] = all_nodes[best_n]
                    selected_set.remove(s)
                    selected_set.add(all_nodes[best_n])
                    used = used_without_s | (1 << best_n)
                    improved = True
                    non_selected = [i for i in range(n_total) if not ((used >> i) & 1)]
                    break
        
        val = sum(1.0 / v for v in selected)
        if val > best_value:
            best_subset = selected
            best_value = val
    
    if not best_subset:
        return [256]
    
    return best_subset
