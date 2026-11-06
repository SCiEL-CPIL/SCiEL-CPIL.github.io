import time
import random
import itertools
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Task: Find a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
# satisfying 1/a = 1/b + 1/c. Maximize cardinality.
# 1/a = 1/b + 1/c  <=>  bc = a(b+c)

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute forbidden triples
    # A triple (a,b,c) is forbidden if 1/a = 1/b + 1/c for some permutation.
    # This is equivalent to checking if any element is the "sum" of the reciprocals of the other two.
    # Let the set be {u,v,w}. It is forbidden if:
    # 1. 1/u = 1/v + 1/w  => vw = u(v+w)
    # 2. 1/v = 1/u + 1/w  => uw = v(u+w)
    # 3. 1/w = 1/u + 1/v  => uv = w(u+v)
    
    forbidden_triples = []
    # Iterate over all triples u < v < w
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            for w in range(v + 1, N + 1):
                # Check if any permutation satisfies the reciprocal equation
                # Condition 1: w is the "small" one? No, 1/a = 1/b + 1/c implies a < b and a < c.
                # So the element with the largest reciprocal (smallest value) must be the sum of the other two.
                # Let's just check the three conditions directly.
                
                # Check if u is the 'a' in 1/a = 1/b + 1/c with b,c being v,w
                # 1/u = 1/v + 1/w  <=> v*w = u*(v+w)
                if v * w == u * (v + w):
                    forbidden_triples.append((u, v, w))
                    continue
                
                # Check if v is the 'a' in 1/a = 1/b + 1/c with b,c being u,w
                # 1/v = 1/u + 1/w  <=> u*w = v*(u+w)
                if u * w == v * (u + w):
                    forbidden_triples.append((u, v, w))
                    continue
                    
                # Check if w is the 'a' in 1/a = 1/b + 1/c with b,c being u,v
                # 1/w = 1/u + 1/v  <=> u*v = w*(u+v)
                if u * v == w * (u + w): # Wait, w is the largest index, so 1/w is smallest. 
                    # 1/w = 1/u + 1/v implies w > u and w > v.
                    # Equation: uv = w(u+v).
                    if u * v == w * (u + v):
                        forbidden_triples.append((u, v, w))
    
    # If no forbidden triples, return all
    if not forbidden_triples:
        return universe

    # Formulate as ILP
    # Variables x_1, ..., x_180 in {0, 1}
    # Maximize sum(x_i)
    # Subject to x_a + x_b + x_c <= 2 for each forbidden triple (a,b,c)
    
    c_obj = -np.ones(N)  # Minimize -sum(x) to maximize sum(x)
    
    # Constraints: A @ x <= b
    # A is sparse matrix of size M x N
    # Use dense for simplicity if M is not too large, or scipy.sparse
    # Let's count M
    M = len(forbidden_triples)
    
    # Build constraint matrix
    # Using scipy.sparse for efficiency
    import scipy.sparse as sp
    
    row_indices = []
    col_indices = []
    data = []
    
    for i, (a, b, c) in enumerate(forbidden_triples):
        # x_a + x_b + x_c <= 2
        # Indices are 0-based
        row_indices.extend([i, i, i])
        col_indices.extend([a-1, b-1, c-1])
        data.extend([1, 1, 1])
        
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(M, N))
    b = np.ones(M) * 2
    
    constraints = LinearConstraint(A, ub=b)
    integrality = np.ones(N)  # All variables are integer
    bounds = Bounds(lb=0, ub=1)
    
    # Solve using HiGHS
    try:
        # Time limit in seconds
        time_limit = 45 
        res = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds, 
                   options={"time_limit": time_limit})
        
        if res.status == 0 or res.status == 1: # 0 is optimal, 1 is feasible
            x = res.x
            # Convert to set
            selected = set()
            for i in range(N):
                if x[i] > 0.5:
                    selected.add(i + 1)
            
            if selected:
                return list(selected)
            else:
                # Fallback if ILP returns empty or fails
                pass
        else:
            # Fallback
            pass
    except Exception as e:
        # Fallback to heuristic if ILP fails
        pass

    # Fallback Heuristic (from original code)
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            num1 = u * v
            den1 = u + v
            if num1 % den1 == 0:
                x1 = num1 // den1
                if 1 <= x1 <= N and x1 != u and x1 != v:
                    conflict_pairs[x1].add(frozenset((u, v)))
            
            num2 = u * v
            den2 = v - u
            if den2 > 0 and num2 % den2 == 0:
                x2 = num2 // den2
                if 1 <= x2 <= N and x2 != u and x2 != v:
                    conflict_pairs[x2].add(frozenset((u, v)))
    
    def is_conflict_with_addition(S, x):
        for pair in conflict_pairs[x]:
            u, v = tuple(pair)
            if u in S and v in S:
                return True
        return False
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if not is_conflict_with_addition(S, x):
                S.add(x)
        return S
    
    def local_search(S):
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_conflict_with_addition(S, x):
                    continue
                S.add(x)
                improved = True
                break
            if not improved:
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if not is_conflict_with_addition(S, x):
                        continue
                    for y in list(S):
                        S.remove(y)
                        if not is_conflict_with_addition(S, x):
                            S.add(x)
                            improved = True
                            break
                        S.add(y)
                    if improved:
                        break
        return S
    
    best_set = set()
    best_size = 0
    orderings = [sorted(universe, reverse=True), sorted(universe)]
    for _ in range(3):
        orderings.append(random.sample(universe, N))
        
    for order in orderings:
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    while time.time() < deadline - 5:
        order = random.sample(universe, N)
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    for x in range(1, N + 1):
        if x not in best_set and not is_conflict_with_addition(best_set, x):
            best_set.add(x)
            
    return list(best_set)
