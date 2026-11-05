import numpy as np
import time
from itertools import combinations
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True

    def build_structure(n):
        """Build all the structural data needed for constraint propagation."""
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = []
        triple_to_k4s = defaultdict(list)
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_to_idx[(a,b,c)]
            t2 = triple_to_idx[(a,b,d)]
            t3 = triple_to_idx[(a,c,d)]
            t4 = triple_to_idx[(b,c,d)]
            k4_idx = len(k4s)
            k4s.append((t1, t2, t3, t4))
            for t in (t1, t2, t3, t4):
                triple_to_k4s[t].append(k4_idx)
        
        k5s = []
        triple_to_k5s = defaultdict(list)
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5_idx = len(k5s)
            k5s.append(tuple(idxs))
            for t in idxs:
                triple_to_k5s[t].append(k5_idx)
        
        return triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s

    def quadratic_residue_coloring(n, mod=None):
        if mod is None:
            mod = n
            while not is_prime(mod):
                mod += 1
        
        qr = set()
        for i in range(1, mod):
            qr.add((i * i) % mod)
        
        triples = list(combinations(range(n), 3))
        colors = []
        for (a, b, c) in triples:
            d1 = (b - a) % mod
            d2 = (c - b) % mod
            if d1 in qr and d2 in qr:
                colors.append(0)
            else:
                colors.append(1)
        return colors

    def count_violations(colors, k4s, k5s):
        cost = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                cost += 1
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                cost += 1
        return cost

    def compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s):
        delta = 0
        old_color = colors[t]
        
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            red_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 0:
                        red_count += 1
                else:
                    if colors[i] == 0:
                        red_count += 1
            
            was_violation = (red_count == 4)
            
            if new_color == 0:
                new_red_count = red_count + (1 if old_color != 0 else 0)
            else:
                new_red_count = red_count - (1 if old_color == 0 else 0)
            
            new_violation = (new_red_count == 4)
            
            if new_violation and not was_violation:
                delta += 1
            elif was_violation and not new_violation:
                delta -= 1
        
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            blue_count = 0
            for i in idxs:
                if i == t:
                    if old_color == 1:
                        blue_count += 1
                else:
                    if colors[i] == 1:
                        blue_count += 1
            
            was_violation = (blue_count == 10)
            
            if new_color == 1:
                new_blue_count = blue_count + (1 if old_color != 1 else 0)
            else:
                new_blue_count = blue_count - (1 if old_color == 1 else 0)
            
            new_violation = (new_blue_count == 10)
            
            if new_violation and not was_violation:
                delta += 1
            elif was_violation and not new_violation:
                delta -= 1
        
        return delta

    def simulate_annealing(n, time_limit, init_colors=None):
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        if init_colors is None:
            candidates = []
            for mod_offset in range(0, 15):
                mod = n + mod_offset
                while not is_prime(mod):
                    mod += 1
                qr = set()
                for i in range(1, mod):
                    qr.add((i * i) % mod)
                
                c = np.zeros(num_triples, dtype=np.int8)
                for idx, (a, b, cc) in enumerate(triples):
                    d1 = (b - a) % mod
                    d2 = (cc - b) % mod
                    if d1 in qr and d2 in qr:
                        c[idx] = 0
                    else:
                        c[idx] = 1
                candidates.append(c)
            
            patterns = [
                lambda a,b,c: 0 if (a+b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a+b+c) % 2 == 0 else 1,
                lambda a,b,c: 0 if (a*b+c) % 3 == 0 else 1,
                lambda a,b,c: 0 if (a^b^c) % 2 == 0 else 1,
            ]
            for pat in patterns:
                c = np.array([pat(a,b,c) for a,b,c in triples], dtype=np.int8)
                candidates.append(c)
            
            candidates.append(np.random.randint(0, 2, num_triples, dtype=np.int8))
            
            best_init = None
            best_init_cost = float('inf')
            for c in candidates:
                cost = count_violations(c, k4s, k5s)
                if cost < best_init_cost:
                    best_init_cost = cost
                    best_init = c
            
            colors = best_init.copy()
        else:
            colors = init_colors.copy()
        
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        
        temp = 2.0
        cool_rate = 0.9995
        min_temp = 0.01
        no_improve_steps = 0
        max_no_improve = 30000
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_cost = compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + delta_cost
            
            if new_cost <= current_cost:
                accept = True
            elif temp > min_temp:
                accept = np.random.rand() < np.exp(-delta_cost / max(temp, 1e-10))
            else:
                accept = False
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve_steps = 0
                    if best_cost == 0:
                        return best_colors.tolist()
                else:
                    no_improve_steps += 1
            else:
                no_improve_steps += 1
            
            if no_improve_steps > max_no_improve:
                temp *= 0.95
                no_improve_steps = 0
            else:
                temp *= cool_rate
            
            if temp < min_temp:
                temp = min_temp
        
        return best_colors.tolist()

    def backtrack_search(n, time_limit):
        """
        Backtracking search with proper constraint propagation and state management.
        """
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        
        colors = [-1] * num_triples
        
        k4_red_count = [0] * len(k4s)
        k4_unassigned = [4] * len(k4s)
        
        k5_blue_count = [0] * len(k5s)
        k5_unassigned = [10] * len(k5s)
        
        # Proper assignment stack: list of (triple_idx, color, affected_k4s, affected_k5s)
        assignment_stack = []
        
        def is_feasible_k4(k4_idx):
            return k4_red_count[k4_idx] < 4
        
        def is_feasible_k5(k5_idx):
            return k5_blue_count[k5_idx] < 10
        
        def assign(triple_idx, color):
            colors[triple_idx] = color
            
            affected_k4s = triple_to_k4s[triple_idx]
            affected_k5s = triple_to_k5s[triple_idx]
            
            for k4_idx in affected_k4s:
                k4_unassigned[k4_idx] -= 1
                if color == 0:
                    k4_red_count[k4_idx] += 1
                if k4_red_count[k4_idx] >= 4:
                    # Rollback
                    for k4i in affected_k4s:
                        k4_unassigned[k4i] += 1
                        if color == 0:
                            k4_red_count[k4i] -= 1
                    for k5i in affected_k5s:
                        k5_unassigned[k5i] += 1
                        if color == 1:
                            k5_blue_count[k5i] -= 1
                    colors[triple_idx] = -1
                    return False
            
            for k5_idx in affected_k5s:
                k5_unassigned[k5_idx] -= 1
                if color == 1:
                    k5_blue_count[k5_idx] += 1
                if k5_blue_count[k5_idx] >= 10:
                    # Rollback
                    for k4i in affected_k4s:
                        k4_unassigned[k4i] += 1
                        if color == 0:
                            k4_red_count[k4i] -= 1
                    for k5i in affected_k5s:
                        k5_unassigned[k5i] += 1
                        if color == 1:
                            k5_blue_count[k5i] -= 1
                    colors[triple_idx] = -1
                    return False
            
            assignment_stack.append((triple_idx, color, affected_k4s, affected_k5s))
            return True
        
        def unassign_last():
            if not assignment_stack:
                return
            triple_idx, color, affected_k4s, affected_k5s = assignment_stack.pop()
            colors[triple_idx] = -1
            
            for k4_idx in affected_k4s:
                k4_unassigned[k4_idx] += 1
                if color == 0:
                    k4_red_count[k4_idx] -= 1
            
            for k5_idx in affected_k5s:
                k5_unassigned[k5_idx] += 1
                if color == 1:
                    k5_blue_count[k5_idx] -= 1
        
        def get_forced_assignments():
            forced = {}
            
            for t in range(num_triples):
                if colors[t] != -1:
                    continue
                
                for k4_idx in triple_to_k4s[t]:
                    if k4_unassigned[k4_idx] == 1 and k4_red_count[k4_idx] == 3:
                        forced[t] = 1
                        break
                
                if t in forced:
                    continue
                
                for k5_idx in triple_to_k5s[t]:
                    if k5_unassigned[k5_idx] == 1 and k5_blue_count[k5_idx] == 9:
                        forced[t] = 0
                        break
            
            return forced
        
        # Precompute constraint counts for variable ordering
        constraint_counts = [len(triple_to_k4s[t]) + len(triple_to_k5s[t]) for t in range(num_triples)]
        
        def search(time_limit):
            if time.time() > time_limit:
                return None
            
            # Check for contradictions
            for k4_idx in range(len(k4s)):
                if k4_red_count[k4_idx] >= 4:
                    return None
            for k5_idx in range(len(k5s)):
                if k5_blue_count[k5_idx] >= 10:
                    return None
            
            # Apply forced assignments
            forced = get_forced_assignments()
            
            while forced:
                t, color = forced.popitem()
                if colors[t] != -1:
                    continue
                if not assign(t, color):
                    # Contradiction - need to backtrack
                    # Find how many forced assignments we made in this round
                    # and undo them
                    return None
                new_forced = get_forced_assignments()
                forced.update(new_forced)
            
            # Find the next unassigned triple using most-constrained-first
            unassigned = [i for i in range(num_triples) if colors[i] == -1]
            if not unassigned:
                return colors[:]
            
            # Pick the most constrained unassigned variable
            best_t = max(unassigned, key=lambda t: constraint_counts[t])
            
            # Try both colors, but prefer the one that's less likely to cause conflicts
            # Heuristic: try the color that has fewer conflicts first
            color_order = [0, 1]
            
            for color in color_order:
                if time.time() > time_limit:
                    return None
                
                if assign(best_t, color):
                    result = search(time_limit)
                    if result is not None:
                        return result
                    unassign_last()
            
            return None
        
        result = search(time_limit)
        if result is not None:
            return result
        return None

    def local_polish(colors, n, time_limit):
        """Local search to reduce violations."""
        triples, num_triples, triple_to_idx, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        colors = np.array(colors, dtype=np.int8)
        
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()
        
        temp = 0.5
        cool_rate = 0.999
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 10000
        
        while time.time() < time_limit:
            t = np.random.randint(num_triples)
            old_color = colors[t]
            new_color = 1 - old_color
            
            delta_cost = compute_delta(t, new_color, colors, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + delta_cost
            
            if new_cost <= current_cost:
                accept = True
            elif temp > min_temp:
                accept = np.random.rand() < np.exp(-delta_cost / max(temp, 1e-10))
            else:
                accept = False
            
            if accept:
                colors[t] = new_color
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors.tolist()
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > max_no_improve:
                temp *= 0.9
                no_improve = 0
            else:
                temp *= cool_rate
            
            if temp < min_temp:
                temp = min_temp
        
        return best_colors.tolist()

    # Main search logic
    best_n = 9
    best_colors = quadratic_residue_coloring(9)
    
    # First, try to find valid colorings for increasing n using backtracking
    # Backtracking is feasible for small n due to constraint propagation
    for n in range(10, 20):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time() - 15
        if remaining < 10:
            break
        
        # Try backtracking with a time limit
        bt_time_limit = time.time() + min(remaining * 0.6, 30)
        bt_result = backtrack_search(n, bt_time_limit)
        
        if bt_result is not None:
            # Verify
            triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
            colors_np = np.array(bt_result, dtype=np.int8)
            if count_violations(colors_np, k4s, k5s) == 0:
                best_n = n
                best_colors = bt_result
                continue
        
        # If backtracking didn't find a solution, try SA
        remaining = deadline - time.time() - 15
        if remaining < 10:
            break
        
        qr_colors = quadratic_residue_coloring(n)
        sa_time_limit = time.time() + min(remaining * 0.8, 40)
        sa_result = simulate_annealing(n, sa_time_limit, init_colors=qr_colors)
        
        triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
        colors_np = np.array(sa_result, dtype=np.int8)
        if count_violations(colors_np, k4s, k5s) == 0:
            best_n = n
            best_colors = sa_result
            continue
        
        # Try local polish on the SA result
        remaining = deadline - time.time() - 15
        if remaining > 5:
            polish_time_limit = time.time() + min(remaining * 0.7, 20)
            polished = local_polish(sa_result, n, polish_time_limit)
            colors_np = np.array(polished, dtype=np.int8)
            if count_violations(colors_np, k4s, k5s) == 0:
                best_n = n
                best_colors = polished
                continue
        
        # Check if we should continue
        remaining = deadline - time.time() - 15
        if remaining < 15:
            break
    
    # For larger n, use SA with multiple restarts
    for n in range(20, min(best_n + 5, 30)):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 8
        if remaining < 10:
            break
        
        # Try multiple SA runs with different initializations
        best_sa_result = None
        best_sa_cost = float('inf')
        
        num_restarts = min(3, int(remaining / 10))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            sa_time_limit = time.time() + min(remaining * 0.4, 30)
            
            if restart == 0:
                init = quadratic_residue_coloring(n)
            elif restart == 1:
                triples, num_triples, _, _, _, _, _ = build_structure(n)
                init = np.random.randint(0, 2, num_triples, dtype=np.int8).tolist()
            else:
                triples, num_triples, _, _, _, _, _ = build_structure(n)
                init = np.random.randint(0, 2, num_triples, dtype=np.int8).tolist()
            
            sa_result = simulate_annealing(n, sa_time_limit, init_colors=init)
            
            triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
            colors_np = np.array(sa_result, dtype=np.int8)
            cost = count_violations(colors_np, k4s, k5s)
            
            if cost < best_sa_cost:
                best_sa_cost = cost
                best_sa_result = sa_result
            
            if cost == 0:
                best_n = n
                best_colors = sa_result
                break
        
        if best_n >= n:
            continue
        
        # Try local polish on the best SA result
        if best_sa_result is not None and best_sa_cost > 0:
            remaining = deadline - time.time() - 8
            if remaining > 5:
                polish_time_limit = time.time() + min(remaining * 0.6, 15)
                polished = local_polish(best_sa_result, n, polish_time_limit)
                triples, num_triples, _, k4s, k5s, _, _ = build_structure(n)
                colors_np = np.array(polished, dtype=np.int8)
                if count_violations(colors_np, k4s, k5s) == 0:
                    best_n = n
                    best_colors = polished
    
    return {'n': best_n, 'colors': best_colors}
