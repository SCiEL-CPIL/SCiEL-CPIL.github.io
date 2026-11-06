import time
import random
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start = time.time()
    DEADLINE = 145.0  # Safety margin

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        Sset = set(S)
        S_list = sorted(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i, n):
                if a + S_list[j] in Sset:
                    return False
        return True

    def can_add(x, S, Sset):
        if 2 * x in Sset:
            return False
        for a in S:
            if (a + x) in Sset:
                return False
            if (a - x) in Sset:
                return False
        return True

    def heuristic_solve(V, time_limit, start_time):
        """Fallback heuristic if ILP fails or times out."""
        best_set = set()
        best_size = 0
        rng = random.Random(42)
        
        # Try a few random restarts with greedy + local search
        for _ in range(10):
            S = set()
            V_shuffled = V[:]
            rng.shuffle(V_shuffled)
            for x in V_shuffled:
                if can_add(x, S, S):
                    S.add(x)
            
            # Simple local search: try to add any element, then swap
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                # Try adding
                for x in V:
                    if x not in S and can_add(x, S, S):
                        S.add(x)
                        improved = True
                        break
                if improved:
                    continue
                # Try 1-for-2 swap
                S_list = list(S)
                rng.shuffle(S_list)
                for u in S_list:
                    S_tent = S - {u}
                    Sset_tent = S_tent
                    added = 0
                    new_S = set(S_tent)
                    for x in V:
                        if x not in Sset_tent and can_add(x, new_S, Sset_tent):
                            new_S.add(x)
                            Sset_tent.add(x)
                            added += 1
                    if added > 1:
                        S = new_S
                        improved = True
                        break
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        
        return sorted(best_set)

    def solve_instance_ilp(V, time_limit, start_time):
        n = len(V)
        if n == 0:
            return []
        
        # Map values to indices
        val_to_idx = {v: i for i, v in enumerate(V)}
        
        # Find all triples (a, b, c) such that a + b = c
        # a, b can be same. c must be distinct? No, a+b=c implies if a,b,c in V.
        # If a+b=a, then b=0. But 0 is not in V.
        # If a+b=b, then a=0.
        # If a+b=c, and a,b,c distinct? Not necessarily.
        # Example: 2+2=4. If 2,4 in V.
        
        triples = []
        V_set = set(V)
        for i, a in enumerate(V):
            for j, b in enumerate(V):
                c = a + b
                if c in V_set:
                    k = val_to_idx[c]
                    # We need to handle the case where a,b,c are not distinct carefully.
                    # The constraint is x_a + x_b + x_c <= 2.
                    # If a=b, it's x_a + x_a + x_c <= 2 => 2x_a + x_c <= 2.
                    # If a=c, then b=0 (impossible).
                    # If b=c, then a=0 (impossible).
                    # So a,b,c can only be equal if a=b and c=2a.
                    # Or if a=c, b=0.
                    # So the only overlap is a=b.
                    triples.append((i, j, k))
        
        # Remove duplicates? (i,j,k) and (j,i,k) are different constraints?
        # x_i + x_j + x_k <= 2 is symmetric in i,j.
        # So (i,j,k) and (j,i,k) are the same constraint.
        # We can deduplicate to speed up solver.
        unique_triples = set()
        for t in triples:
            # Sort i,j to deduplicate symmetric pairs?
            # The constraint is linear: x_i + x_j + x_k <= 2.
            # It doesn't matter which is which.
            # But if i=j, it's 2x_i + x_k <= 2.
            # We should store the tuple of indices involved.
            # If i==j, store (i, i, k).
            # If i!=j, store (min(i,j), max(i,j), k).
            if i == j:
                key = (i, i, k)
            else:
                key = (min(i, j), max(i, j), k)
            unique_triples.add(key)
        
        if not unique_triples:
            return V[:] # No constraints, take all
        
        # Build A matrix for constraints
        # Number of constraints m
        m = len(unique_triples)
        A = np.zeros((m, n))
        for idx, (i, j, k) in enumerate(unique_triples):
            A[idx, i] += 1
            A[idx, j] += 1
            A[idx, k] += 1
            
        # Objective: maximize sum x_i
        c = np.ones(n)
        
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=0, ub=1)
        
        # Integrality: all integer
        integrality = np.ones(n)
        
        # Constraint: A x <= m
        constraint = LinearConstraint(A, ub=np.full(m, 2))
        
        try:
            # Use HiGHS solver
            # time_limit is in seconds
            # We allocate a portion of the time
            res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0: # Optimal
                x = res.x
                # Threshold 0.5
                selected_indices = [i for i in range(n) if x[i] > 0.5]
                S = [V[i] for i in selected_indices]
                # Verify sum-free just in case
                if is_sum_free(S):
                    return S
                else:
                    # Fallback if numerical issues
                    return heuristic_solve(V, time_limit, start_time)
            else:
                # Infeasible or other status
                return heuristic_solve(V, time_limit, start_time)
        except Exception as e:
            return heuristic_solve(V, time_limit, start_time)

    results = []
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining_total = DEADLINE - elapsed
        
        if remaining_total < 5:
            # Fallback
            odds = [x for x in V if x % 2 != 0]
            results.append(sorted(odds))
            continue
        
        instances_left = 3 - len(results)
        # Allocate more time to larger instances?
        # ILP is fast for 60, slower for 240.
        # Let's just split remaining time evenly minus a buffer.
        time_per_instance = (remaining_total / instances_left) - 2
        if time_per_instance < 1:
            time_per_instance = 1
            
        S_best = solve_instance_ilp(V, time_per_instance, start)
        
        if not is_sum_free(S_best):
            odds = [x for x in V if x % 2 != 0]
            S_best = sorted(odds)
            
        results.append(S_best)
        
    return results

if __name__ == "__main__":
    res = construct()
    print(res)
