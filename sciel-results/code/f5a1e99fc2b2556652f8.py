import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0
    
    recip = [0.0] * (n + 1)
    for i in range(1, n + 1):
        recip[i] = 1.0 / i

    triples = []
    for a in range(1, n + 1):
        for d in range(1, n - a + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            triples.append((a, b, c))
    
    def get_forbidden_pair(x, y):
        forbidden = 0
        s = x + y
        if s % 2 == 0:
            mid = s // 2
            if mid != x and mid != y and 1 <= mid <= n:
                forbidden |= (1 << mid)
        z = 2 * x - y
        if z != x and z != y and 1 <= z <= n:
            forbidden |= (1 << z)
        z2 = 2 * y - x
        if z2 != x and z2 != y and 1 <= z2 <= n:
            forbidden |= (1 << z2)
        return forbidden

    def can_add_fast(x, s_set):
        for y in s_set:
            s = x + y
            if s % 2 == 0:
                mid = s // 2
                if mid != x and mid != y and mid in s_set:
                    return False
            z = 2 * x - y
            if z != x and z != y and 1 <= z <= n and z in s_set:
                return False
            z2 = 2 * y - x
            if z2 != x and z2 != y and 1 <= z2 <= n and z2 in s_set:
                return False
        return True

    def greedy_fill(s_list, s_set):
        for x in range(1, n + 1):
            if not s_list[x]:
                if can_add_fast(x, s_set):
                    s_list[x] = True
                    s_set.add(x)
        return s_list, s_set

    def get_sum(s_list):
        total = 0.0
        for i in range(1, n + 1):
            if s_list[i]:
                total += recip[i]
        return total

    def local_search(s_list, s_set):
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            current_sum = get_sum(s_list)
            for rem in range(1, n + 1):
                if not s_list[rem]:
                    continue
                trial_list = s_list[:]
                trial_set = set(s_set)
                trial_list[rem] = False
                trial_set.discard(rem)
                greedy_fill(trial_list, trial_set)
                new_sum = get_sum(trial_list)
                if new_sum > current_sum + 1e-12:
                    s_list = trial_list
                    s_set = trial_set
                    current_sum = new_sum
                    improved = True
                    break
        return s_list, s_set

    def bb_swap_improve(s_list, s_set, max_candidates=40, max_pair_depth=2):
        nonlocal best_set, best_set_set, best_sum
        current_sum = get_sum(s_list)
        in_set_list = [x for x in range(1, n+1) if s_list[x]]
        not_in_list = [x for x in range(1, n+1) if not s_list[x]]
        not_in_list.sort(key=lambda x: -recip[x])
        not_in_list = not_in_list[:max_candidates]
        
        for rem in in_set_list:
            if time.time() > deadline - 6.0:
                return
            new_set = set(s_set)
            new_set.discard(rem)
            base_sum = current_sum - recip[rem]
            
            # Try single additions
            best_add = 0.0
            best_add_elems = []
            for x in not_in_list:
                if can_add_fast(x, new_set):
                    if recip[x] > best_add:
                        best_add = recip[x]
                        best_add_elems = [x]
            
            # Try pairs
            if max_pair_depth >= 2:
                for i in range(len(not_in_list)):
                    for j in range(i+1, len(not_in_list)):
                        x1, x2 = not_in_list[i], not_in_list[j]
                        if can_add_fast(x1, new_set):
                            temp = new_set | {x1}
                            if can_add_fast(x2, temp):
                                pair_sum = recip[x1] + recip[x2]
                                if pair_sum > best_add:
                                    best_add = pair_sum
                                    best_add_elems = [x1, x2]
            
            new_total = base_sum + best_add
            if new_total > best_sum + 1e-12:
                new_s_list = s_list[:]
                new_s_list[rem] = False
                for x in best_add_elems:
                    new_s_list[x] = True
                best_set = new_s_list
                best_set_set = set(i for i in range(1, n+1) if new_s_list[i])
                best_sum = new_total

    # 1. Initial Greedy
    in_set = [False] * (n + 1)
    in_set_set = set()
    greedy_fill(in_set, in_set_set)
    best_set = in_set[:]
    best_set_set = in_set_set.copy()
    best_sum = get_sum(best_set)
    
    # 2. Local Search
    improved_list, improved_set = local_search(best_set[:], best_set_set.copy())
    if get_sum(improved_list) > best_sum:
        best_set = improved_list
        best_set_set = improved_set
        best_sum = get_sum(best_set)
    
    # 3. B&B swap improvement
    bb_swap_improve(best_set[:], best_set_set.copy())
    
    # 4. Randomized Restart
    random.seed(42)
    while time.time() < deadline - 8.0:
        perturbed_list = best_set[:]
        perturbed_set = set(best_set_set)
        num_removals = random.randint(1, 5)
        elements_in_set = list(perturbed_set)
        if not elements_in_set:
            break
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_list[rem_val] = False
            perturbed_set.discard(rem_val)
            elements_in_set.pop(idx)
        greedy_fill(perturbed_list, perturbed_set)
        improved_list, improved_set = local_search(perturbed_list, perturbed_set)
        new_sum = get_sum(improved_list)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_set = improved_list
            best_set_set = improved_set

    # --- MILP Attempt ---
    try:
        rows = []
        cols = []
        vals = []
        b_vec = []
        for a, b, c in triples:
            row_idx = len(b_vec)
            rows.append(row_idx); cols.append(a - 1); vals.append(1.0)
            rows.append(row_idx); cols.append(b - 1); vals.append(1.0)
            rows.append(row_idx); cols.append(c - 1); vals.append(1.0)
            b_vec.append(2.0)
        A = csr_matrix((vals, (rows, cols)), shape=(len(b_vec), n))
        c_obj = [-recip[i] for i in range(1, n + 1)]
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        constraint = LinearConstraint(A, lb=-np.inf, ub=b_vec)
        milp_time_limit = max(1.0, deadline - time.time() - 5.0)
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"time_limit": milp_time_limit, "presolve": "default"})
        if res.status == 0 and res.x is not None:
            milp_set_list = [False] * (n + 1)
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_set_list[i + 1] = True
            milp_set = set(i for i in range(1, n+1) if milp_set_list[i])
            valid = True
            for a, b, c in triples:
                if a in milp_set and b in milp_set and c in milp_set:
                    valid = False
                    break
            if valid:
                milp_sum = get_sum(milp_set_list)
                if milp_sum > best_sum:
                    best_set = milp_set_list
                    best_set_set = milp_set
                    best_sum = milp_sum
    except Exception:
        pass

    result = [i for i in range(1, n + 1) if best_set[i]]
    s_set = set(result)
    for a, b, c in triples:
        if a in s_set and b in s_set and c in s_set:
            raise AssertionError("3-AP found")
    return result
