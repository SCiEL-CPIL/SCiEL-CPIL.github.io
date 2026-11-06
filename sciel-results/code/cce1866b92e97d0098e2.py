import time
from math import gcd
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np
from fractions import Fraction
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)

    # Build conflict graph: conflict[i][j] = True if lcm(nodes[i], nodes[j]) <= N
    conflict = [[False]*n for _ in range(n)]
    conflict_sets = [set() for _ in range(n)]
    for i in range(n):
        a = nodes[i]
        for j in range(i+1, n):
            b = nodes[j]
            g = gcd(a, b)
            lcm_ab = a * b // g
            if lcm_ab <= N:
                conflict[i][j] = True
                conflict[j][i] = True
                conflict_sets[i].add(j)
                conflict_sets[j].add(i)

    weights = np.array([1.0 / x for x in nodes])

    # Greedy initial solution
    remaining = set(range(n))
    greedy_solution = []
    greedy_value = 0.0
    while remaining:
        best_idx = max(remaining, key=lambda i: weights[i])
        greedy_solution.append(best_idx)
        greedy_value += weights[best_idx]
        to_remove = {best_idx}
        for j in remaining:
            if conflict[best_idx][j]:
                to_remove.add(j)
        remaining -= to_remove

    best_value = greedy_value
    best_solution = list(greedy_solution)

    # Local search: 1-swap and 1-add/1-remove
    def local_search(solution, value):
        sol_set = set(solution)
        improved = True
        while improved:
            improved = False
            # Try adding each non-conflicting node
            for i in range(n):
                if i in sol_set:
                    continue
                # Check if i conflicts with any node in solution
                if any(conflict[i][j] for j in sol_set):
                    continue
                new_val = value + weights[i]
                if new_val > value + 1e-12:
                    sol_set.add(i)
                    value = new_val
                    improved = True
            # Try swapping: remove one, add another
            for i in list(sol_set):
                for j in range(n):
                    if j in sol_set:
                        continue
                    # Check if j conflicts with (sol_set - {i})
                    if any(conflict[j][k] for k in sol_set if k != i):
                        continue
                    new_val = value - weights[i] + weights[j]
                    if new_val > value + 1e-12:
                        sol_set.remove(i)
                        sol_set.add(j)
                        value = new_val
                        improved = True
                        break
                if improved:
                    break
            # Try removing a low-weight node if it allows adding a better one
            for i in sorted(sol_set, key=lambda x: weights[x]):
                old_set = sol_set - {i}
                # Find best node we can add
                best_add = -1
                best_gain = 0
                for j in range(n):
                    if j in old_set:
                        continue
                    if any(conflict[j][k] for k in old_set):
                        continue
                    gain = weights[j] - weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_add = j
                if best_add >= 0:
                    sol_set.discard(i)
                    sol_set.add(best_add)
                    value += best_gain
                    improved = True
                    break
        return list(sol_set), value

    best_solution, best_value = local_search(best_solution, best_value)

    # Branch and bound
    # Upper bound: for a set of available nodes, use greedy coloring bound or
    # simple sum of weights of available nodes
    # Better: use the fact that among conflicting nodes, we can pick at most one
    # from each connected component of the conflict subgraph
    
    def branch_and_bound(available, current_val, current_set):
        nonlocal best_value, best_solution
        if time.time() > deadline - 2.0:
            return
        
        if not available:
            return
        
        # Upper bound: sum of all weights in available
        ub = current_val + sum(weights[i] for i in available)
        if ub <= best_value + 1e-12:
            return
        
        # Find a node to branch on: pick the one with highest weight
        # Or pick one with many conflicts for better pruning
        avail_list = list(available)
        # Sort by weight descending, break ties by degree
        avail_list.sort(key=lambda i: (-weights[i], -len(conflict_sets[i])))
        
        for idx in avail_list:
            if time.time() > deadline - 2.0:
                return
            
            # Option 1: include idx
            new_set = current_set | {idx}
            new_available = {j for j in available if j != idx and not conflict[idx][j]}
            new_val = current_val + weights[idx]
            
            # Quick check: if this already beats best, update
            if new_val > best_value + 1e-12:
                best_value = new_val
                best_solution = list(new_set)
            
            # Recurse
            branch_and_bound(new_available, new_val, new_set)
            
            # Option 2: exclude idx
            new_available2 = {j for j in available if j != idx}
            branch_and_bound(new_available2, current_val, current_set)
    
    # Start B&B from full set
    full_available = set(range(n))
    branch_and_bound(full_available, 0.0, set())
    
    # Final local search on best solution found
    best_solution, best_value = local_search(best_solution, best_value)
    
    # Verify
    result = [nodes[i] for i in best_solution]
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a, b = result[i], result[j]
            g = gcd(a, b)
            if a * b // g <= N:
                valid = False
                break
        if not valid:
            break
    
    if valid and len(result) > 0:
        return result
    
    # Fallback to greedy
    return [nodes[i] for i in greedy_solution]

if __name__ == "__main__":
    construct()
