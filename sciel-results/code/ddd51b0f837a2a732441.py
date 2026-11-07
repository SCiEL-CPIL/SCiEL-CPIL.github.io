import math
import time
import random
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin under 90s
    
    N = 256
    
    # Precompute prime factors for 1..N
    F = [None] * (N + 1)
    F[1] = frozenset()
    for i in range(2, N + 1):
        x = i
        factors = set()
        d = 2
        while d * d <= x:
            if x % d == 0:
                factors.add(d)
                while x % d == 0:
                    x //= d
            else:
                d += 1
        if x > 1:
            factors.add(x)
        F[i] = frozenset(factors)
    
    # Precompute conflict graph: conflict[i][j] = True if gcd(i,j) > 1
    # Also precompute list of conflicting partners for each number
    conflict_partners = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if math.gcd(i, j) > 1:
                conflict_partners[i].append(j)
                conflict_partners[j].append(i)
    
    # Try MILP with tighter time and better formulation
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        
        # Variables x_1 ... x_256
        c = np.array([-i for i in range(1, N + 1)], dtype=float)
        
        # Build constraint matrix for conflicts
        M = sum(len(cp) for cp in conflict_partners) // 2  # each conflict counted twice
        A = lil_matrix((M, N), dtype=float)
        idx = 0
        for i in range(1, N + 1):
            for j in conflict_partners[i]:
                if j > i:
                    A[idx, i - 1] = 1
                    A[idx, j - 1] = 1
                    idx += 1
        
        A_csc = A.tocsc()
        lb = np.zeros(M)
        ub = np.ones(M)
        constraints = LinearConstraint(A_csc, lb, ub)
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        integrality = np.ones(N)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"presolve": True, "time_limit": 25.0})
        
        if res.status == 0:
            sol = res.x
            selected = [int(i + 1) for i in range(N) if sol[i] > 0.5]
            # Verify
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if math.gcd(selected[i], selected[j]) != 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [int(x) for x in selected]
                
    except Exception:
        pass
    
    # Heuristic approach: use maximum weight independent set on conflict graph
    # with greedy + local search + simulated annealing
    
    def greedy_initialization(strategy='descending'):
        if strategy == 'descending':
            candidates = list(range(N, 0, -1))
        elif strategy == 'ascending':
            candidates = list(range(1, N + 1))
        else:  # random
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        
        selected = []
        selected_set = set()
        selected_factors = set()
        
        for a in candidates:
            fa = F[a]
            if selected_factors.isdisjoint(fa):
                selected.append(a)
                selected_set.add(a)
                selected_factors.update(fa)
        
        return selected_set
    
    def can_add(x, selected_factors):
        return F[x].isdisjoint(selected_factors)
    
    def local_search(S, max_iterations=50):
        S = set(S)
        best_sum = sum(S)
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            if time.time() > deadline - 3:
                break
            
            # Compute factors
            S_factors = set()
            for x in S:
                S_factors.update(F[x])
            
            # Step 1: Greedy Addition
            added_any = True
            while added_any:
                added_any = False
                for x in range(N, 0, -1):
                    if x not in S and can_add(x, S_factors):
                        S.add(x)
                        S_factors.update(F[x])
                        best_sum += x
                        improved = True
                        added_any = True
                        break
            
            if time.time() > deadline - 3:
                break
            
            # Step 2: 1-for-1 Swaps
            S_list = sorted(S)
            for a in S_list:
                if time.time() > deadline - 3:
                    break
                rest = S - {a}
                rest_factors = set()
                for x in rest:
                    rest_factors.update(F[x])
                
                best_b = 0
                for b in range(N, a, -1):
                    if b not in S and can_add(b, rest_factors):
                        best_b = b
                        break
                
                if best_b > a:
                    S.remove(a)
                    S.add(best_b)
                    best_sum += (best_b - a)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Step 3: Remove-one-and-solve
            for a in sorted(S):
                if time.time() > deadline - 3:
                    return S
                
                rest = S - {a}
                rest_factors = set()
                for x in rest:
                    rest_factors.update(F[x])
                
                # Find best set of replacements
                candidates = [x for x in range(N, 0, -1) if x not in rest and can_add(x, rest_factors)]
                
                # Greedy pick replacements
                replacements = []
                temp_factors = set(rest_factors)
                for x in candidates:
                    if can_add(x, temp_factors):
                        replacements.append(x)
                        temp_factors.update(F[x])
                
                if sum(replacements) > a:
                    S = rest.union(replacements)
                    best_sum = sum(S)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Step 4: 2-exchange
            if time.time() > deadline - 6:
                break
                
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline - 3:
                    return S
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    rest = S - {a, b}
                    rest_factors = set()
                    for x in rest:
                        rest_factors.update(F[x])
                    
                    candidates = [x for x in range(N, 0, -1) if x not in rest and can_add(x, rest_factors)]
                    
                    replacements = []
                    temp_factors = set(rest_factors)
                    for x in candidates:
                        if can_add(x, temp_factors):
                            replacements.append(x)
                            temp_factors.update(F[x])
                    
                    if sum(replacements) > a + b:
                        S = rest.union(replacements)
                        best_sum = sum(S)
                        improved = True
                        break
                if improved:
                    break
        
        return S
    
    # Simulated annealing for better exploration
    def simulated_annealing(S, num_steps=2000):
        S = set(S)
        current_sum = sum(S)
        best_S = set(S)
        best_sum = current_sum
        
        T = 50.0
        cooling = 0.999
        
        S_factors = set()
        for x in S:
            S_factors.update(F[x])
        
        for step in range(num_steps):
            if time.time() > deadline - 3:
                break
            
            T *= cooling
            
            # Random move: either add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Add a random element
                candidates = [x for x in range(1, N + 1) if x not in S and can_add(x, S_factors)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_S = S | {x}
                delta = x
                if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                    S = new_S
                    S_factors.update(F[x])
                    current_sum += delta
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_S = set(S)
                        
            elif move_type < 0.6:
                # Remove a random element
                if not S:
                    continue
                x = random.choice(list(S))
                new_S = S - {x}
                delta = -x
                if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                    S = new_S
                    S_factors.difference_update(F[x])
                    current_sum += delta
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_S = set(S)
                        
            else:
                # Swap: remove one, add one
                if not S:
                    continue
                a = random.choice(list(S))
                rest = S - {a}
                rest_factors = set()
                for x in rest:
                    rest_factors.update(F[x])
                
                candidates = [x for x in range(1, N + 1) if x not in S and can_add(x, rest_factors)]
                if not candidates:
                    continue
                b = random.choice(candidates)
                delta = b - a
                if delta >= 0 or random.random() < math.exp(delta / max(T, 0.01)):
                    S = rest | {b}
                    S_factors.difference_update(F[a])
                    S_factors.update(F[b])
                    current_sum += delta
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_S = set(S)
        
        return best_S
    
    best_S = set()
    best_sum = 0
    
    # 1. Greedy Descending
    S1 = greedy_initialization('descending')
    if sum(S1) > best_sum:
        best_S = S1
        best_sum = sum(S1)
        
    # 2. Greedy Ascending
    S2 = greedy_initialization('ascending')
    if sum(S2) > best_sum:
        best_S = S2
        best_sum = sum(S2)

    # 3. Randomized Greedy (multiple seeds)
    random.seed(42)
    for i in range(10):
        if time.time() > deadline - 15:
            break
        Sr = greedy_initialization('random')
        if sum(Sr) > best_sum:
            best_S = Sr
            best_sum = sum(Sr)

    # 4. Local Search on Best
    if time.time() < deadline - 10:
        best_S = local_search(best_S)
        best_sum = sum(best_S)

    # 5. Simulated Annealing
    if time.time() < deadline - 5:
        sa_result = simulated_annealing(best_S)
        if sum(sa_result) > best_sum:
            best_S = sa_result
            best_sum = sum(best_S)

    # 6. More local search passes
    if time.time() < deadline - 5:
        best_S = local_search(best_S)
        best_sum = sum(best_S)

    # 7. Try local search on other initializations
    if time.time() < deadline - 8:
        for i in range(5):
            if time.time() > deadline - 4:
                break
            Sr = greedy_initialization('random')
            Sr = local_search(Sr)
            if sum(Sr) > best_sum:
                best_S = Sr
                best_sum = sum(Sr)

    # Final: try adding any missing coprime elements
    S_factors = set()
    for x in best_S:
        S_factors.update(F[x])
    for x in range(N, 0, -1):
        if x not in best_S and can_add(x, S_factors):
            best_S.add(x)
            S_factors.update(F[x])
    
    # Ensure return type is list of ints
    return [int(x) for x in best_S]
