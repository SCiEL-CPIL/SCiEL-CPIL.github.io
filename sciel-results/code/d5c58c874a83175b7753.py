import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix
from fractions import Fraction

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Precompute reciprocals as floats and exact fractions
    recip_f = [0.0] * (n + 1)
    recip_frac = [Fraction(0, 1)] * (n + 1)
    for i in range(1, n + 1):
        recip_f[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)

    # Precompute all 3-APs
    triples = []
    for a in range(1, n + 1):
        for d in range(1, n - a + 1):
            b = a + d
            c = a + 2 * d
            if c > n:
                break
            triples.append((a, b, c))

    # Precompute conflict sets: for each element x, which other elements y
    # would create a 3-AP if both x and y are in the set (with some third element)
    # conflict[x] = set of y such that {x,y} + some z forms a 3-AP
    # Actually, for can_add(x, S): x conflicts with S if there exists y in S
    # such that x,y form part of a 3-AP with a third element also in S.
    # 
    # More precisely, x cannot be added to S if there exist a,b in S such that
    # {x,a,b} is a 3-AP. This happens when:
    # 1. x is middle: a,b endpoints, 2x = a+b, so a+b-2x=0
    # 2. x is endpoint: a is other endpoint, b is middle: a+x=2b, so a+x-2b=0
    # 3. x is endpoint: b is other endpoint, a is middle: b+x=2a, so b+x-2a=0
    #
    # For fast checking, precompute for each pair (x,y) whether they're in a 3-AP together
    # But that's O(n^2) pairs. Instead, for each x, precompute the set of y such that
    # x and y are in some 3-AP (i.e., there exists z with {x,y,z} being a 3-AP).
    
    # conflict_mask[x] = bitmask of all y such that x,y are in a common 3-AP
    # But we need: x can't be added if there's y in S where the third element z is also in S.
    # 
    # Let's precompute: for each x, for each y, what is the third element z (if any)?
    # There are at most 3 possible z values for a given pair (x,y):
    # If x,y are endpoints: z = (x+y)/2 if integer
    # If x is first, y is second: z = 2y-x
    # If x is second, y is first: z = 2x-y
    # If x is first, y is third: z = (x+y)/2 if integer (same as endpoints)
    # If x is third, y is second: z = 2y-x (same as x first y second with roles swapped)
    # If x is second, y is third: z = 2x-y (same as x second y first)
    
    # So for pair (x,y) with x != y:
    # Possible third elements:
    #   z1 = 2*y - x (if x is middle or x is first,y is second)
    #   z2 = 2*x - y (if y is middle or y is first,x is second)  
    #   z3 = (x+y)//2 if (x+y) even (if x,y are endpoints)
    # Each z must be in [1,n] and z != x and z != y

    # Precompute third_elements[x][y] = set of possible z
    # Use a 2D array or dict for speed
    third_elems = {}
    for x in range(1, n + 1):
        third_elems[x] = {}
        for y in range(1, n + 1):
            if x == y:
                third_elems[x][y] = set()
                continue
            zs = set()
            # z = 2y - x
            z = 2 * y - x
            if 1 <= z <= n and z != x and z != y:
                zs.add(z)
            # z = 2x - y
            z = 2 * x - y
            if 1 <= z <= n and z != x and z != y:
                zs.add(z)
            # z = (x+y)/2
            s = x + y
            if s % 2 == 0:
                z = s // 2
                if 1 <= z <= n and z != x and z != y:
                    zs.add(z)
            third_elems[x][y] = zs

    def can_add(x, in_set_bool):
        """Check if x can be added to set represented by in_set_bool (list of bools)."""
        for y in range(1, n + 1):
            if in_set_bool[y]:
                zs = third_elems[x][y]
                for z in zs:
                    if in_set_bool[z]:
                        return False
        return True

    def can_add_fast(x, in_set_set):
        """Check if x can be added to set represented by in_set_set (a set of ints)."""
        for y in in_set_set:
            zs = third_elems[x][y]
            for z in zs:
                if z in in_set_set:
                    return False
        return True

    def greedy_fill(in_set_bool, in_set_set, order=None):
        """Add all valid elements. order is list of elements to try, or None for 1..n."""
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if not in_set_bool[x]:
                if can_add_fast(x, in_set_set):
                    in_set_bool[x] = True
                    in_set_set.add(x)
        return in_set_bool, in_set_set

    def get_sum_f(in_set_bool):
        total = 0.0
        for i in range(1, n + 1):
            if in_set_bool[i]:
                total += recip_f[i]
        return total

    def get_sum_frac(in_set_set):
        total = Fraction(0, 1)
        for i in in_set_set:
            total += recip_frac[i]
        return total

    def local_search(in_set_bool, in_set_set, time_limit):
        """Local search with removal-refill and swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            current_sum = get_sum_f(in_set_bool)
            
            # Try removing one element and refilling
            elements_in_set = list(in_set_set)
            random.shuffle(elements_in_set)
            for rem in elements_in_set:
                if time.time() > time_limit:
                    break
                
                trial_bool = in_set_bool[:]
                trial_set = set(in_set_set)
                trial_bool[rem] = False
                trial_set.discard(rem)
                
                greedy_fill(trial_bool, trial_set)
                new_sum = get_sum_f(trial_bool)
                
                if new_sum > current_sum + 1e-12:
                    in_set_bool = trial_bool
                    in_set_set = trial_set
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try swapping one element with another
            if not improved and time.time() < time_limit:
                elements_in_set = list(in_set_set)
                random.shuffle(elements_in_set)
                for a in elements_in_set:
                    if time.time() > time_limit:
                        break
                    for b in range(1, n + 1):
                        if in_set_bool[b]:
                            continue
                        trial_bool = in_set_bool[:]
                        trial_set = set(in_set_set)
                        trial_bool[a] = False
                        trial_set.discard(a)
                        if can_add_fast(b, trial_set):
                            trial_bool[b] = True
                            trial_set.add(b)
                            greedy_fill(trial_bool, trial_set)
                            new_sum = get_sum_f(trial_bool)
                            if new_sum > current_sum + 1e-12:
                                in_set_bool = trial_bool
                                in_set_set = trial_set
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        return in_set_bool, in_set_set

    # === Strategy 1: Greedy by index ===
    in_set_bool = [False] * (n + 1)
    in_set_set = set()
    greedy_fill(in_set_bool, in_set_set)
    best_bool = in_set_bool[:]
    best_set = in_set_set.copy()
    best_sum = get_sum_f(best_bool)
    
    # Local search
    improved_bool, improved_set = local_search(best_bool[:], best_set.copy(), deadline - 20.0)
    if get_sum_f(improved_bool) > best_sum:
        best_bool = improved_bool
        best_set = improved_set
        best_sum = get_sum_f(best_bool)

    # === Strategy 2: Greedy by reciprocal value (descending) ===
    in_set_bool2 = [False] * (n + 1)
    in_set_set2 = set()
    order_by_val = sorted(range(1, n + 1), key=lambda x: -recip_f[x])
    greedy_fill(in_set_bool2, in_set_set2, order_by_val)
    if get_sum_f(in_set_bool2) > best_sum:
        best_bool = in_set_bool2
        best_set = in_set_set2
        best_sum = get_sum_f(best_bool)
        improved_bool, improved_set = local_search(best_bool[:], best_set.copy(), deadline - 20.0)
        if get_sum_f(improved_bool) > best_sum:
            best_bool = improved_bool
            best_set = improved_set
            best_sum = get_sum_f(best_bool)

    # === Strategy 3: Randomized restarts ===
    random.seed(42)
    
    while time.time() < deadline - 15.0:
        # Perturb best set
        perturbed_bool = best_bool[:]
        perturbed_set = set(best_set)
        
        num_removals = random.randint(1, 4)
        elements_in_set = list(perturbed_set)
        if not elements_in_set:
            break
            
        for _ in range(num_removals):
            if not elements_in_set:
                break
            idx = random.randint(0, len(elements_in_set) - 1)
            rem_val = elements_in_set[idx]
            perturbed_bool[rem_val] = False
            perturbed_set.discard(rem_val)
            elements_in_set.pop(idx)
        
        # Refill with random order
        remaining = [x for x in range(1, n + 1) if not perturbed_bool[x]]
        random.shuffle(remaining)
        greedy_fill(perturbed_bool, perturbed_set, remaining)
        
        # Local search
        improved_bool, improved_set = local_search(perturbed_bool, perturbed_set, deadline - 15.0)
        
        new_sum = get_sum_f(improved_bool)
        if new_sum > best_sum + 1e-12:
            best_sum = new_sum
            best_bool = improved_bool
            best_set = improved_set

    # === MILP Attempt ===
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
        c_obj = [-recip_f[i] for i in range(1, n + 1)]
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(n)
        constraint = LinearConstraint(A, lb=-np.inf, ub=b_vec)
        
        milp_time_limit = max(1.0, deadline - time.time() - 3.0)
        
        res = milp(c=c_obj, constraints=constraint, integrality=integrality, bounds=bounds, 
                   options={"time_limit": milp_time_limit, "presolve": "default"})
        
        if res.status == 0 and res.x is not None:
            milp_bool = [False] * (n + 1)
            for i in range(n):
                if res.x[i] > 0.5:
                    milp_bool[i + 1] = True
            
            # Validate
            milp_set = set(i for i in range(1, n + 1) if milp_bool[i])
            valid = True
            for a, b, c in triples:
                if a in milp_set and b in milp_set and c in milp_set:
                    valid = False
                    break
            
            if valid:
                milp_sum = get_sum_f(milp_bool)
                if milp_sum > best_sum:
                    best_sum = milp_sum
                    best_bool = milp_bool
                    best_set = milp_set
                    
    except Exception:
        pass

    # Final polish: try one more local search
    if time.time() < deadline - 2.0:
        improved_bool, improved_set = local_search(best_bool[:], best_set.copy(), deadline - 1.0)
        if get_sum_f(improved_bool) > best_sum:
            best_bool = improved_bool
            best_set = improved_set
            best_sum = get_sum_f(best_bool)

    # Extract result
    result = [i for i in range(1, n + 1) if best_bool[i]]
    
    # Final validation
    s_set = set(result)
    for a, b, c in triples:
        if a in s_set and b in s_set and c in s_set:
            raise AssertionError("3-AP found: " + str((a, b, c)))
            
    return result
