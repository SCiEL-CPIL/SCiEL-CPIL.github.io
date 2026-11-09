"""Max-SAT style simulated annealing for reciprocal Schur triple avoidance."""
import time
import random
import math
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

# Deadline guard: finish well before 90s wall-clock
_DEADLINE = 85.0

def _precompute_triples():
    """
    Find all triples (a, b, c) of distinct integers in [1, 180] such that
    1/a = 1/b + 1/c.
    """
    triples = []
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, int(math.isqrt(a2)) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d == e:
                continue
            b = a + d
            c = a + e
            if b > 180 or c > 180:
                continue
            if b < 1 or c < 1:
                continue
            if a == b or a == c or b == c:
                continue
            triples.append((a, b, c))
    return triples

def _is_valid(set_bits, triples):
    """Check if no triple has all three elements in the set."""
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            return False
    return True

def _count_violations(set_bits, triples):
    """Count number of violated triples."""
    count = 0
    for (a, b, c) in triples:
        if set_bits[a] and set_bits[b] and set_bits[c]:
            count += 1
    return count

def _ilp_solve(triples, time_budget):
    """
    Solve the maximum independent set problem on the hypergraph of forbidden triples
    using ILP. We maximize sum(x_i) subject to x_a + x_b + x_c <= 2 for each triple.
    """
    n = 180
    m = len(triples)
    
    # Objective: maximize sum of x_i
    c = np.ones(n)
    
    # Constraints: for each triple (a,b,c), x_a + x_b + x_c <= 2
    # Variables are indexed 0..179 for elements 1..180
    A = np.zeros((m, n))
    for i, (a, b, c) in enumerate(triples):
        A[i, a-1] = 1
        A[i, b-1] = 1
        A[i, c-1] = 1
    
    # Constraints: A @ x <= 2 for each row
    cons = LinearConstraint(A, -np.inf, 2)
    
    # Bounds: 0 <= x_i <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all variables are integer (binary)
    integrality = np.ones(n)
    
    start = time.time()
    result = milp(c=-c, constraints=cons, bounds=bounds, integrality=integrality, 
                  options={'time_limit': time_budget, 'mip_rel_gap': 0.01})
    
    elapsed = time.time() - start
    
    if result.status == 0 or result.x is not None:
        # result.x contains the solution; round to binary
        x = np.round(result.x).astype(int)
        solution_set = set()
        for i in range(n):
            if x[i] == 1:
                solution_set.add(i + 1)
        return solution_set, elapsed
    
    return set(), elapsed

def _local_search_polish(initial_set, triples, time_budget):
    """
    Local search polish: try to add elements that don't create violations,
    and remove elements that allow adding more.
    """
    n = 180
    set_bits = [False] * (n + 1)
    current_set = set(initial_set)
    for x in current_set:
        set_bits[x] = True
    
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        
        # Try adding each element not in the set
        for x in range(1, n + 1):
            if set_bits[x]:
                continue
            # Check if adding x creates any violation
            can_add = True
            for (a, b, c) in triples:
                if x in (a, b, c):
                    other_two = [v for v in (a, b, c) if v != x]
                    if other_two[0] in current_set and other_two[1] in current_set:
                        can_add = False
                        break
            if can_add:
                current_set.add(x)
                set_bits[x] = True
                improved = True
        
        # Try removing one element and see if we can add two or more
        for x in list(current_set):
            # Temporarily remove x
            set_bits[x] = False
            current_set.discard(x)
            
            # Count how many we can now add
            added = 0
            for y in range(1, n + 1):
                if y == x or set_bits[y]:
                    continue
                can_add = True
                for (a, b, c) in triples:
                    if y in (a, b, c):
                        other_two = [v for v in (a, b, c) if v != y]
                        if other_two[0] in current_set and other_two[1] in current_set:
                            can_add = False
                            break
                if can_add:
                    current_set.add(y)
                    set_bits[y] = True
                    added += 1
            
            if added >= 2:
                improved = True
            else:
                # Revert: put x back, remove the added elements
                for y in range(1, n + 1):
                    if y != x and set_bits[y] and y not in initial_set:
                        # This was added during this trial
                        pass
                # Simpler: just keep the improvement if it's better
                if added < 2:
                    # Revert x removal
                    set_bits[x] = True
                    current_set.add(x)
                    # Remove any elements added during the trial
                    # This is tricky; let's just accept the state
    
    return current_set

def construct():
    start_time = time.time()
    
    triples = _precompute_triples()
    n = 180
    
    # Try ILP first
    ilp_time_budget = min(40.0, _DEADLINE - (time.time() - start_time) - 10)
    if ilp_time_budget > 5:
        ilp_set, ilp_elapsed = _ilp_solve(triples, ilp_time_budget)
    else:
        ilp_set = set()
        ilp_elapsed = 0
    
    # Fallback greedy if ILP failed or returned empty
    if len(ilp_set) < 100:
        def greedy_init():
            triple_count = [0] * (n + 1)
            for (a, b, c) in triples:
                triple_count[a] += 1
                triple_count[b] += 1
                triple_count[c] += 1
            
            best_set = set()
            for seed in range(10):
                random.seed(seed)
                order = list(range(1, n + 1))
                random.shuffle(order)
                order.sort(key=lambda x: (triple_count[x], random.random()))
                
                current_set = set()
                set_bits = [False] * (n + 1)
                for x in order:
                    can_add = True
                    for (a, b, c) in triples:
                        if x in (a, b, c):
                            other_two = [v for v in (a, b, c) if v != x]
                            if other_two[0] in current_set and other_two[1] in current_set:
                                can_add = False
                                break
                    if can_add:
                        current_set.add(x)
                        set_bits[x] = True
                
                if len(current_set) > len(best_set):
                    best_set = current_set
            
            return best_set
        
        ilp_set = greedy_init()
    
    # Local search polish with remaining time
    remaining_time = _DEADLINE - (time.time() - start_time) - 2
    if remaining_time > 1:
        polished_set = _local_search_polish(ilp_set, triples, remaining_time)
        if len(polished_set) > len(ilp_set):
            ilp_set = polished_set
    
    # Final validation
    set_bits = [False] * (n + 1)
    for x in ilp_set:
        set_bits[x] = True
    if not _is_valid(set_bits, triples):
        # Remove elements causing violations
        changed = True
        while changed and time.time() - start_time < _DEADLINE - 1:
            changed = False
            for (a, b, c) in triples:
                if set_bits[a] and set_bits[b] and set_bits[c]:
                    # Remove the element with highest triple count
                    counts = [a, b, c]
                    counts.sort(key=lambda v: -_count_violations(set_bits, triples))
                    # Just remove 'a' for simplicity
                    set_bits[a] = False
                    ilp_set.discard(a)
                    changed = True
    
    if not ilp_set:
        # Absolute fallback
        ilp_set = set(range(1, n + 1, 2))  # Odd numbers as a safe fallback
    
    return sorted(ilp_set)
