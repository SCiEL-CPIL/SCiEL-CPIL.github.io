"""Optimal disjoint residue classes via clique + CSP with time guards."""
# EVOLVE-BLOCK-START
import math
import time
import random

_START_TIME = None

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _disjoint(r1, m1, r2, m2):
    g = _gcd(m1, m2)
    if g == 1:
        return False
    return (r2 - r1) % g != 0

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    deadline = _START_TIME + 85.0

    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mod = len(moduli)

    # Build non-coprime graph
    nc_adj = [0] * n_mod
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            if _gcd(moduli[i], moduli[j]) > 1:
                nc_adj[i] |= (1 << j)
                nc_adj[j] |= (1 << i)

    # Find all maximal cliques or use B&B to find maximum feasible clique
    # Strategy: enumerate cliques in descending size order, check CSP feasibility
    
    # First, find the maximum clique size
    def max_clique_size(adj, n):
        best = 0
        def expand(cand_mask, size):
            nonlocal best
            ub = cand_mask.bit_count()
            if size + ub <= best:
                return
            if cand_mask == 0:
                if size > best:
                    best = size
                return
            # Pick vertex with max degree in cand
            cm = cand_mask
            best_v = -1
            best_deg = -1
            while cm:
                u = (cm & -cm).bit_length() - 1
                deg = (adj[u] & cand_mask).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = u
                cm &= cm - 1
            if best_v == -1:
                return
            # Include
            expand(adj[best_v] & cand_mask, size + 1)
            # Exclude
            expand(cand_mask & ~(1 << best_v), size)
        expand((1 << n) - 1, 0)
        return best
    
    max_clique_sz = max_clique_size(nc_adj, n_mod)
    
    # Now enumerate cliques of size max_clique_sz down to max_clique_sz - 5
    # and check CSP feasibility
    
    def find_cliques_of_size(adj, n, target_size, max_cliques=1000):
        """Find cliques of exactly target_size."""
        cliques = []
        def expand(cand_mask, clique, depth):
            nonlocal cliques
            if len(cliques) >= max_cliques:
                return
            if len(clique) == target_size:
                cliques.append(list(clique))
                return
            remaining = target_size - len(clique)
            if cand_mask.bit_count() < remaining:
                return
            cm = cand_mask
            while cm and len(cliques) < max_cliques:
                v = (cm & -cm).bit_length() - 1
                cm &= cm - 1
                clique.append(v)
                expand(adj[v] & cand_mask, clique, depth + 1)
                clique.pop()
        expand((1 << n) - 1, [], 0)
        return cliques
    
    def solve_csp(mod_list, time_limit=5.0):
        """Assign residues with backtracking and time limit."""
        n = len(mod_list)
        if n == 0:
            return []
        start = time.time()
        
        gcds = [[1] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                gcds[i][j] = _gcd(mod_list[i], mod_list[j])
                gcds[j][i] = gcds[i][j]
        
        assignment = [None] * n
        
        def backtrack(idx):
            if time.time() - start > time_limit:
                return False
            if idx == n:
                return True
            m = mod_list[idx]
            # Try residues in order of least constraint first
            for r in range(m):
                ok = True
                for j in range(idx):
                    g = gcds[idx][j]
                    if (r - assignment[j]) % g == 0:
                        ok = False
                        break
                if ok:
                    assignment[idx] = r
                    if backtrack(idx + 1):
                        return True
                    assignment[idx] = None
            return False
        
        if backtrack(0):
            return [(assignment[i], mod_list[i]) for i in range(n)]
        return None
    
    best_solution = []
    
    # Try cliques from largest to smaller
    for sz in range(max_clique_sz, max(0, max_clique_sz - 10), -1):
        if time.time() > deadline - 10:
            break
        cliques = find_cliques_of_size(nc_adj, n_mod, sz, max_cliques=500)
        for clique in cliques:
            if time.time() > deadline - 5:
                break
            clique_moduli = [moduli[i] for i in clique]
            sol = solve_csp(clique_moduli, time_limit=3.0)
            if sol is not None and len(sol) > len(best_solution):
                best_solution = sol
    
    # Convert and try to extend
    if not best_solution:
        # Fallback: greedy
        best_solution = []
        for m in moduli:
            for r in range(m):
                if all(_disjoint(r, m, r2, m2) for r2, m2 in best_solution):
                    best_solution.append((r, m))
                    break
    
    result = [[r, m] for r, m in best_solution]
    used_moduli = set(m for _, m in result)
    
    # Try to add more moduli
    for m in moduli:
        if m in used_moduli:
            continue
        for r in range(m):
            if all(_disjoint(r, m, r2, m2) for r2, m2 in result):
                result.append([r, m])
                used_moduli.add(m)
                break
    
    return result if result else [[1, 2]]
# EVOLVE-BLOCK-END
