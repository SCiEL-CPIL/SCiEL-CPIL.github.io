"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 84
    
    N = 128
    
    # Precompute gcd matrix
    gcd_mat = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
    
    # Precompute conflict pairs: for each x, pairs (a,b) such that {a,b,x} is bad
    conflict_pairs = [set() for _ in range(N+1)]
    for x in range(1, N+1):
        for a in range(1, x):
            for b in range(a+1, N+1):
                if b == x:
                    continue
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    conflict_pairs[x].add((a, b))
    
    def can_add(x, current_set):
        for a, b in conflict_pairs[x]:
            if a in current_set and b in current_set:
                return False
        return True
    
    def greedy(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
    
    def local_search(sol, max_iter=300):
        sol = set(sol)
        for _ in range(max_iter):
            if time.time() > DEADLINE:
                break
            improved = False
            for x in range(1, N+1):
                if x not in sol and can_add(x, sol):
                    sol.add(x)
                    improved = True
                    break
            if improved:
                continue
            for rem in list(sol):
                sol.remove(rem)
                for add in range(1, N+1):
                    if add not in sol and can_add(add, sol):
                        sol.add(add)
                        improved = True
                        break
                if improved:
                    break
                sol.add(rem)
            if not improved:
                break
        return sol
    
    best_sol = set()
    random.seed(777)
    
    orderings = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        sorted(range(1, N+1), key=lambda x: -len(conflict_pairs[x])),
        sorted(range(1, N+1), key=lambda x: len(conflict_pairs[x])),
    ]
    for _ in range(20):
        order = list(range(1, N+1))
        random.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > DEADLINE - 3:
            break
        sol = greedy(order)
        sol = local_search(sol, max_iter=300)
        if len(sol) > len(best_sol):
            best_sol = sol
    
    for _ in range(50):
        if time.time() > DEADLINE - 3:
            break
        sol = set()
        for x in random.sample(range(1, N+1), 30):
            if can_add(x, sol):
                sol.add(x)
        sol = local_search(sol, max_iter=300)
        if len(sol) > len(best_sol):
            best_sol = sol
    
    for x in range(1, N+1):
        if x not in best_sol and can_add(x, best_sol):
            best_sol.add(x)
    
    return list(best_sol)

# EVOLVE-BLOCK-END
