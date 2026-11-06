import time
from math import gcd
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    deadline = time.time() + 55
    
    N = 256
    nodes = list(range(2, N + 1))
    n_vars = len(nodes)
    
    # Weights
    c = -np.array([1.0 / x for x in nodes])
    
    # Build conflict matrix
    rows = []
    cols = []
    vals = []
    constraint_idx = 0
    
    for i_idx in range(n_vars):
        i = nodes[i_idx]
        for j_idx in range(i_idx + 1, n_vars):
            j = nodes[j_idx]
            g = gcd(i, j)
            if i * j <= 256 * g:
                rows.append(constraint_idx)
                cols.append(i_idx)
                vals.append(1.0)
                rows.append(constraint_idx)
                cols.append(j_idx)
                vals.append(1.0)
                constraint_idx += 1
    
    num_constraints = constraint_idx
    
    if num_constraints == 0:
        return nodes
    
    A = sp.csr_matrix((vals, (rows, cols)), shape=(num_constraints, n_vars))
    
    lb = np.zeros(num_constraints)
    ub = np.ones(num_constraints)
    
    constraint = LinearConstraint(A, lb, ub)
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
    integrality = np.ones(n_vars)
    
    best_subset = []
    
    try:
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            subset = [nodes[i] for i in range(n_vars) if x[i] > 0.5]
            if subset:
                best_subset = subset
            else:
                best_subset = [nodes[0]]
        else:
            greedy_set = []
            for i in nodes:
                ok = True
                for y in greedy_set:
                    if i * y <= 256 * gcd(i, y):
                        ok = False
                        break
                if ok:
                    greedy_set.append(i)
            best_subset = greedy_set
            
    except Exception:
        greedy_set = []
        for i in nodes:
            ok = True
            for y in greedy_set:
                if i * y <= 256 * gcd(i, y):
                    ok = False
                    break
            if ok:
                greedy_set.append(i)
        best_subset = greedy_set
    
    # Local search refinement
    if time.time() < deadline - 5:
        best_subset = _local_search(best_subset, nodes, deadline)
    
    # Final validation and cleanup
    if not best_subset:
        best_subset = [2]
    
    # Verify all pairs satisfy lcm > 256
    final = []
    for x in best_subset:
        ok = True
        for y in final:
            if lcm_val(x, y) <= 256:
                ok = False
                break
        if ok:
            final.append(x)
    
    return final if final else [2]

def lcm_val(a, b):
    return a * b // gcd(a, b)

def _local_search(initial_subset, nodes, deadline):
    """Local search: try to improve the subset by removing and adding elements."""
    current = set(initial_subset)
    
    # Build adjacency: which pairs conflict
    conflict = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                if i * j <= 256 * gcd(i, j):
                    conflict.setdefault(i, set()).add(j)
    
    def is_valid(s):
        for a in s:
            for b in s:
                if a != b and b in conflict.get(a, set()):
                    return False
        return True
    
    def exact_score(s):
        total = Fraction(0, 1)
        for x in s:
            total += Fraction(1, x)
        return total
    
    best = set(current)
    best_score = exact_score(current)
    
    # Try single removals (always valid, just check if score improves)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for elem in list(current):
            if len(current) <= 1:
                break
            candidate = current - {elem}
            s = exact_score(candidate)
            if s > best_score:
                best_score = s
                best = set(candidate)
                current = set(candidate)
                improved = True
    
    # Try single additions
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for elem in nodes:
            if elem in current:
                continue
            # Check if elem conflicts with any in current
            if any(elem in conflict.get(x, set()) for x in current):
                continue
            candidate = current | {elem}
            s = exact_score(candidate)
            if s > best_score:
                best_score = s
                best = set(candidate)
                current = set(candidate)
                improved = True
    
    # Try swaps: remove one, add another
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for rem in list(current):
            for add in nodes:
                if add in current or add == rem:
                    continue
                # Check validity: remove rem, add add
                # add must not conflict with any element in current \ {rem}
                if any(add in conflict.get(x, set()) for x in current if x != rem):
                    continue
                candidate = (current - {rem}) | {add}
                s = exact_score(candidate)
                if s > best_score:
                    best_score = s
                    best = set(candidate)
                    current = set(candidate)
                    improved = True
                    break
            if improved:
                break
    
    # Try 2-for-1: remove two, add one
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        current_list = list(current)
        for i_idx in range(len(current_list)):
            for j_idx in range(i_idx + 1, len(current_list)):
                rem1, rem2 = current_list[i_idx], current_list[j_idx]
                for add in nodes:
                    if add in current:
                        continue
                    # Check validity
                    if any(add in conflict.get(x, set()) for x in current if x != rem1 and x != rem2):
                        continue
                    candidate = (current - {rem1, rem2}) | {add}
                    s = exact_score(candidate)
                    if s > best_score:
                        best_score = s
                        best = set(candidate)
                        current = set(candidate)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    # Try 1-for-2: remove one, add two
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for rem in list(current):
            for add1 in nodes:
                if add1 in current:
                    continue
                if any(add1 in conflict.get(x, set()) for x in current if x != rem):
                    continue
                for add2 in nodes:
                    if add2 in current or add2 == add1:
                        continue
                    if any(add2 in conflict.get(x, set()) for x in current if x != rem):
                        continue
                    if add1 in conflict.get(add2, set()):
                        continue
                    candidate = (current - {rem}) | {add1, add2}
                    s = exact_score(candidate)
                    if s > best_score:
                        best_score = s
                        best = set(candidate)
                        current = set(candidate)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return list(best) if best else list(current)
