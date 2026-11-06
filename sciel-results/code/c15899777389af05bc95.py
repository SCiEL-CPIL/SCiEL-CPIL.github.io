import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))
    
    gcd_table = [[0] * (MOD_MAX + 1) for _ in range(MOD_MAX + 1)]
    for i in range(2, MOD_MAX + 1):
        for j in range(i, MOD_MAX + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def disjoint(r1, m1, r2, m2):
        g = gcd_table[m1][m2]
        if g <= 1:
            return False
        return (r1 % g) != (r2 % g)
    
    def can_add(sol, r, m):
        for r_k, m_k in sol:
            if not disjoint(r, m, r_k, m_k):
                return False
        return True
    
    def find_valid_residue(sol, m):
        for r in range(m):
            if can_add(sol, r, m):
                return r
        return -1
    
    def local_search(sol, max_rounds=100):
        sol = [list(x) for x in sol]
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            selected_mods = set(m for _, m in sol)
            
            for m in moduli:
                if m in selected_mods:
                    continue
                r = find_valid_residue(sol, m)
                if r >= 0:
                    sol.append([r, m])
                    improved = True
                    break
            if improved:
                continue
            
            for i in range(len(sol)):
                if time.time() > deadline:
                    break
                saved = sol.pop(i)
                added = False
                for m in moduli:
                    if m == saved[1]:
                        continue
                    r = find_valid_residue(sol, m)
                    if r >= 0:
                        sol.append([r, m])
                        added = True
                        break
                if not added:
                    sol.insert(i, saved)
                else:
                    improved = True
                    break
            if improved:
                continue
            break
        return sol
    
    best = []
    
    for p in range(2, MOD_MAX + 1):
        multiples = [m for m in moduli if m % p == 0]
        if len(multiples) < p:
            continue
        multiples.sort()
        sol = [[i, multiples[i]] for i in range(p)]
        valid = True
        for i in range(len(sol)):
            for j in range(i+1, len(sol)):
                if not disjoint(sol[i][0], sol[i][1], sol[j][0], sol[j][1]):
                    valid = False
                    break
            if not valid:
                break
        if valid and len(sol) > len(best):
            best = [list(x) for x in sol]
    
    best = local_search(best, max_rounds=200)
    
    for _ in range(1000):
        if time.time() > deadline - 2.0:
            break
        random.shuffle(moduli)
        sol = []
        for m in moduli:
            r = find_valid_residue(sol, m)
            if r >= 0:
                sol.append([r, m])
        sol_ls = local_search(sol, max_rounds=10)
        if len(sol_ls) > len(best):
            best = sol_ls
    
    return best
