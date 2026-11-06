import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    """
    Returns a nonempty subset of [1, 180] with no three distinct elements a, b, c 
    satisfying 1/a = 1/b + 1/c. Maximizes cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    N = 180
    numbers = list(range(1, N + 1))

    # 1. Identify all forbidden triples (a, b, c) with distinct elements
    # 1/a = 1/b + 1/c  =>  a = bc / (b+c)
    triples = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            numerator = b * c
            denominator = b + c
            if numerator % denominator == 0:
                a = numerator // denominator
                if 1 <= a <= N and a != b and a != c:
                    triple = tuple(sorted([a, b, c]))
                    triples.add(triple)

    triples_list = list(triples)
    num_triples = len(triples_list)

    # Precompute for each element, which triples it participates in
    element_triples = {x: [] for x in numbers}
    for idx, t in enumerate(triples_list):
        for x in t:
            element_triples[x].append(idx)
    
    # Precompute degree
    degree = {x: len(element_triples[x]) for x in numbers}

    # Helper to check if adding x to current_set creates a conflict
    def can_add(x, current_set):
        for t_idx in element_triples[x]:
            t = triples_list[t_idx]
            others = [v for v in t if v != x]
            if others[0] in current_set and others[1] in current_set:
                return False
        return True

    def greedy_initial():
        current_set = set(numbers)
        changed = True
        while changed:
            changed = False
            violating = []
            for t in triples_list:
                if t[0] in current_set and t[1] in current_set and t[2] in current_set:
                    violating.append(t)
            
            if not violating:
                break
                
            conflict_count = {}
            for t in violating:
                for x in t:
                    conflict_count[x] = conflict_count.get(x, 0) + 1
            
            # Remove element with max conflicts, tie-break by degree
            max_conf = max(conflict_count.values())
            candidates = [x for x, c in conflict_count.items() if c == max_conf]
            to_remove = max(candidates, key=lambda x: degree[x])
            current_set.remove(to_remove)
            changed = True
            
        return current_set

    best_set = greedy_initial()
    best_size = len(best_set)

    # 2. Iterated Local Search (ILS)
    # We represent the state by the set of removed elements to save memory/time?
    # No, just use sets.
    
    def local_search(initial_set):
        current = set(initial_set)
        removed = set(numbers) - current
        improved = True
        while improved:
            improved = False
            # Try to add back each removed element
            # Sort removed by degree ascending (easier to add?) or random
            removed_list = list(removed)
            random.shuffle(removed_list)
            for x in removed_list:
                if time.time() > deadline - 5:
                    return current
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    removed.discard(x)
                    improved = True
        return current

    # Run ILS
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        # Perturb best_set
        current = set(best_set)
        # Remove some elements randomly
        num_remove = random.randint(2, min(10, len(current)))
        to_remove = random.sample(list(current), num_remove)
        for x in to_remove:
            current.discard(x)
        
        result = local_search(current)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
            # If we found a new best, we might want to restart the ILS loop count or just continue
            # We can also try to "shake" more aggressively if we get stuck

    # 3. Try ILP to find the optimal or near-optimal solution
    try:
        c = -np.ones(N)
        A = np.zeros((num_triples, N))
        for idx, t in enumerate(triples_list):
            A[idx, t[0] - 1] = 1
            A[idx, t[1] - 1] = 1
            A[idx, t[2] - 1] = 1

        integrality = np.ones(N)
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=2)

        # Add time limit to milp? scipy.optimize.milp doesn't have a direct time limit param in all versions,
        # but we can check time before and after.
        # Actually, HiGHS can be slow. Let's hope the heuristic is good enough.
        # If we have time, run MILP.
        if time.time() < deadline - 10:
            milp_result = milp(
                c=c,
                constraints=[lin_cons],
                integrality=integrality,
                bounds=bounds
            )

            if milp_result.success and milp_result.x is not None:
                x = milp_result.x
                x_rounded = np.round(x).astype(int)

                ilp_set = set()
                for i in range(N):
                    if x_rounded[i] == 1:
                        ilp_set.add(i + 1)

                # Verify
                valid = True
                for t in triples_list:
                    if t[0] in ilp_set and t[1] in ilp_set and t[2] in ilp_set:
                        valid = False
                        break

                if valid and len(ilp_set) >= best_size:
                    best_set = ilp_set
                    best_size = len(ilp_set)

    except Exception as e:
        pass

    # Final verification and repair if necessary
    # This loop is O(N * num_triples) which is fine
    for t in triples_list:
        if t[0] in best_set and t[1] in best_set and t[2] in best_set:
            # Remove the element with highest degree
            candidates = sorted([t[0], t[1], t[2]], key=lambda x: degree[x], reverse=True)
            best_set.remove(candidates[0])

    if not best_set:
        best_set = {1}

    return list(best_set)
