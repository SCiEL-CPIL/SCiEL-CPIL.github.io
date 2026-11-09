"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict, deque

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _find_max_classes(max_mod=128, deadline=None):
    if deadline is None:
        deadline = time.time() + 85

    # Pre-compute gcd table
    gcd_table = [[0]*(max_mod+1) for _ in range(max_mod+1)]
    for m1 in range(2, max_mod+1):
        for m2 in range(2, max_mod+1):
            gcd_table[m1][m2] = gcd(m1, m2)

    all_moduli = list(range(2, max_mod+1))

    # Build constraint graph
    adj = defaultdict(set)
    for i, m1 in enumerate(all_moduli):
        for m2 in all_moduli[i+1:]:
            if gcd_table[m1][m2] > 1:
                adj[m1].add(m2)
                adj[m2].add(m1)

    # Find connected components
    visited = set()
    components = []
    for m in all_moduli:
        if m not in visited:
            comp = []
            queue = deque([m])
            visited.add(m)
            while queue:
                node = queue.popleft()
                comp.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)

    components.sort(key=len, reverse=True)

    best_set = []
    best_size = 0

    def solve_component(comp, deadline):
        if time.time() > deadline:
            return None
        if not comp:
            return {}
        if len(comp) == 1:
            return {comp[0]: 0}

        assigned = {}
        unassigned = list(comp)

        def get_domain(m, assigned):
            domain = list(range(m))
            for m_old, r_old in assigned.items():
                g = gcd_table[m][m_old]
                if g > 1:
                    bad = set()
                    start = r_old % g
                    for r in range(start, m, g):
                        bad.add(r)
                    if bad:
                        domain = [r for r in domain if r not in bad]
            return domain

        def backtrack(assigned, unassigned, deadline):
            if time.time() > deadline:
                return None
            if not unassigned:
                return dict(assigned)

            min_var = None
            min_domain = None
            min_len = float('inf')

            for m in unassigned:
                dom = get_domain(m, assigned)
                if len(dom) == 0:
                    return None
                if len(dom) < min_len:
                    min_len = len(dom)
                    min_var = m
                    min_domain = dom
                    if min_len <= 2:
                        break

            if min_domain is None:
                return None

            random.shuffle(min_domain)
            new_unassigned = [m for m in unassigned if m != min_var]

            for r in min_domain:
                assigned[min_var] = r
                result = backtrack(assigned, new_unassigned, deadline)
                if result is not None:
                    return result
                del assigned[min_var]

            return None

        return backtrack(assigned, unassigned, deadline)

    # Solve each component
    full_solution = {}
    all_solvable = True

    for comp in components:
        if time.time() > deadline:
            all_solvable = False
            break
        if len(comp) == 1:
            full_solution[comp[0]] = 0
            continue
        result = solve_component(comp, deadline)
        if result is None:
            all_solvable = False
            break
        full_solution.update(result)

    if all_solvable and full_solution:
        return [[r, m] for m, r in full_solution.items()]

    # Fallback: greedy with multiple orderings
    if not full_solution:
        for iteration in range(2000):
            if time.time() > deadline:
                break
            order = all_moduli[:]
            random.shuffle(order)
            current_dict = {}
            for m in order:
                constraints = []
                for m_old, r_old in current_dict.items():
                    g = gcd_table[m][m_old]
                    if g > 1:
                        constraints.append((g, r_old))
                valid = []
                for r in range(m):
                    ok = True
                    for g, r_old in constraints:
                        if (r - r_old) % g == 0:
                            ok = False
                            break
                    if ok:
                        valid.append(r)
                if valid:
                    current_dict[m] = random.choice(valid)
            if len(current_dict) > best_size:
                best_size = len(current_dict)
                best_set = [[r, m] for m, r in current_dict.items()]
    else:
        current_dict = dict(full_solution)
        remaining = [m for m in all_moduli if m not in current_dict]
        random.shuffle(remaining)
        for m in remaining:
            if time.time() > deadline:
                break
            constraints = []
            for m_old, r_old in current_dict.items():
                g = gcd_table[m][m_old]
                if g > 1:
                    constraints.append((g, r_old))
            valid = []
            for r in range(m):
                ok = True
                for g, r_old in constraints:
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    valid.append(r)
            if valid:
                current_dict[m] = random.choice(valid)
        if len(current_dict) > best_size:
            best_size = len(current_dict)
            best_set = [[r, m] for m, r in current_dict.items()]

    # Local search extension
    for passes in range(200):
        if time.time() > deadline:
            break
        if not best_set:
            break
        current_dict = {m: r for r, m in best_set}
        remaining = [m for m in all_moduli if m not in current_dict]
        if not remaining:
            break
        added_any = False
        for m in remaining:
            if time.time() > deadline:
                break
            constraints = []
            for m_old, r_old in current_dict.items():
                g = gcd_table[m][m_old]
                if g > 1:
                    constraints.append((g, r_old))
            valid = []
            for r in range(m):
                ok = True
                for g, r_old in constraints:
                    if (r - r_old) % g == 0:
                        ok = False
                        break
                if ok:
                    valid.append(r)
            if valid:
                current_dict[m] = random.choice(valid)
                added_any = True
                if len(current_dict) > best_size:
                    best_size = len(current_dict)
                    best_set = [[r, m] for m, r in current_dict.items()]
        if not added_any:
            break

    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
