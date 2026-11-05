import time
import math
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    MAX_M = 128
    moduli = list(range(2, MAX_M + 1))
    n = len(moduli)

    # Precompute gcd for all pairs
    gcd_tab = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_tab[i][j] = g
            gcd_tab[j][i] = g

    # Build conflict graph: edge if gcd > 1
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if gcd_tab[i][j] > 1:
                conflict[i].add(j)
                conflict[j].add(i)

    # Degree for ordering
    degree = [len(conflict[i]) for i in range(n)]

    def check_residue(idx, r, selected_map):
        """Check if residue r for modulus idx is compatible with all selected."""
        for idx2, r2 in selected_map.items():
            g = gcd_tab[idx][idx2]
            if (r - r2) % g == 0:
                return False
        return True

    def try_assign_residues(order):
        """Given an order of moduli (independent set), assign residues via backtracking."""
        selected_map = {}
        
        def backtrack(pos):
            if pos == len(order):
                return True
            idx = order[pos]
            m = moduli[idx]
            # Try residues in smart order
            residues = [1, 0] + [r for r in range(2, m)]
            random.shuffle(residues[2:])
            for r in residues:
                if check_residue(idx, r, selected_map):
                    selected_map[idx] = r
                    if backtrack(pos + 1):
                        return True
                    del selected_map[idx]
            return False
        
        if backtrack(0):
            return selected_map
        return None

    def try_greedy_residues(order):
        """Greedy residue assignment - faster but may fail where backtracking succeeds."""
        selected_map = {}
        for idx in order:
            m = moduli[idx]
            residues = [1, 0] + [r for r in range(2, m)]
            random.shuffle(residues[2:])
            for r in residues:
                if check_residue(idx, r, selected_map):
                    selected_map[idx] = r
                    break
            else:
                return None
        return selected_map

    def find_max_independent_set(order, budget):
        """Branch and bound for maximum independent set."""
        best_set = []
        best_size = 0
        
        # Use the given order, try to build independent set
        # With pruning: if remaining vertices can't beat best, stop
        def branch(available, current):
            nonlocal best_set, best_size
            if time.time() > budget:
                return
            if not available:
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
                return
            
            # Upper bound: current size + number of available vertices
            if len(current) + len(available) <= best_size:
                return
            
            # Pick a vertex from available
            # Choose vertex with highest degree in the induced subgraph for better pruning
            v = max(available, key=lambda x: sum(1 for y in available if y in conflict[x]))
            
            # Option 1: Include v
            new_available = set(available)
            new_available.discard(v)
            new_available -= conflict[v]
            current.append(v)
            branch(new_available, current)
            current.pop()
            
            # Option 2: Exclude v
            new_available = set(available)
            new_available.discard(v)
            branch(new_available, current)
        
        branch(set(range(n)), [])
        return best_set

    best_solution = []
    best_len = 0

    # Strategy 1: Greedy independent set with multiple orderings
    orderings = [
        sorted(range(n), key=lambda i: -degree[i]),
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: moduli[i]),
        sorted(range(n), key=lambda i: -moduli[i]),
        sorted(range(n), key=lambda i: (-degree[i], moduli[i])),
        sorted(range(n), key=lambda i: (degree[i], -moduli[i])),
    ]

    for order in orderings:
        # Greedy independent set
        selected = []
        available = set(range(n))
        for idx in order:
            if idx not in available:
                continue
            selected.append(idx)
            available.discard(idx)
            available -= conflict[idx]
        
        # Try to assign residues
        res_map = try_greedy_residues(selected)
        if res_map is None:
            res_map = try_assign_residues(selected)
        if res_map and len(res_map) > best_len:
            best_len = len(res_map)
            best_solution = [(idx, r) for idx, r in res_map.items()]

    # Strategy 2: Branch and bound for larger independent sets
    if time.time() < deadline - 20:
        # Try B&B with different orderings
        for order in orderings[:3]:
            budget = min(deadline - 15, start_time + 30)
            iset = find_max_independent_set(order, budget)
            if len(iset) > best_len:
                # Sort by the ordering for better residue assignment
                iset_sorted = sorted(iset, key=lambda x: order.index(x))
                res_map = try_greedy_residues(iset_sorted)
                if res_map is None:
                    res_map = try_assign_residues(iset_sorted)
                if res_map and len(res_map) > best_len:
                    best_len = len(res_map)
                    best_solution = [(idx, r) for idx, r in res_map.items()]

    # Strategy 3: Local search - try to add more moduli by swapping residues
    if time.time() < deadline - 10:
        selected_idx = set(idx for idx, _ in best_solution)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(n):
                if idx in selected_idx:
                    continue
                m = moduli[idx]
                # Try to add idx by possibly changing existing residues
                # First try direct addition
                residues = [1, 0] + [r for r in range(2, m)]
                random.shuffle(residues[2:])
                for r in residues:
                    if check_residue(idx, r, {i: r2 for i, r2 in best_solution}):
                        best_solution.append((idx, r))
                        selected_idx.add(idx)
                        improved = True
                        break
                if improved:
                    break

    # Strategy 4: More aggressive local search with 1-swap
    if time.time() < deadline - 5:
        selected_idx = set(idx for idx, _ in best_solution)
        swap_improved = True
        while swap_improved and time.time() < deadline - 5:
            swap_improved = False
            for new_idx in range(n):
                if new_idx in selected_idx:
                    continue
                m_new = moduli[new_idx]
                
                # Try 1-swap: change one existing residue to allow new_idx
                for pos in range(len(best_solution)):
                    old_idx, old_r = best_solution[pos]
                    old_m = moduli[old_idx]
                    
                    for new_r_old in range(old_m):
                        if new_r_old == old_r:
                            continue
                        # Check if new_r_old is compatible with all others
                        ok_old = True
                        for qpos in range(len(best_solution)):
                            if qpos == pos:
                                continue
                            other_idx, other_r = best_solution[qpos]
                            g = gcd_tab[old_idx][other_idx]
                            if (new_r_old - other_r) % g == 0:
                                ok_old = False
                                break
                        if not ok_old:
                            continue
                        
                        # Check if new_idx can be added
                        found_new = False
                        for r_new in range(m_new):
                            ok_new = True
                            for qpos in range(len(best_solution)):
                                if qpos == pos:
                                    if (r_new - new_r_old) % gcd_tab[new_idx][old_idx] == 0:
                                        ok_new = False
                                        break
                                else:
                                    other_idx, other_r = best_solution[qpos]
                                    if (r_new - other_r) % gcd_tab[new_idx][other_idx] == 0:
                                        ok_new = False
                                        break
                            if ok_new:
                                found_new = True
                                break
                        
                        if found_new:
                            best_solution[pos] = (old_idx, new_r_old)
                            best_solution.append((new_idx, r_new))
                            selected_idx.add(new_idx)
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                if swap_improved:
                    break

    # Convert to required format
    return [[r, moduli[idx]] for idx, r in best_solution]
