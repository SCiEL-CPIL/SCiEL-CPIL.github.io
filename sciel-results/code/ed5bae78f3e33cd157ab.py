import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 180
    elements = list(range(1, N + 1))
    
    # 1. Precompute all forbidden triples
    # Equation: 1/a = 1/b + 1/c => c = ab / (b-a)
    # Assume a < b. Then c = a + a^2 / (b-a).
    # Let d = b-a. d must divide a^2.
    # b = a + d, c = a + a^2/d.
    # Conditions: b <= N, c <= N, b != c, a, b, c distinct.
    
    triples = []
    elem_to_triples = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        a2 = a * a
        # d must be a divisor of a^2
        # b = a + d <= N => d <= N - a
        max_d = N - a
        for d in range(1, max_d + 1):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if c <= N and b != c:
                    # a, b, c are distinct because c > a and b > a, and b != c
                    t = tuple(sorted((a, b, c)))
                    idx = len(triples)
                    triples.append(t)
                    for x in t:
                        elem_to_triples[x].append(idx)
                        
    num_triples = len(triples)
    
    # 2. Smart Initialization with multiple strategies
    def run_smart_greedy():
        # Base set: [87, 180] is known to be valid (no distinct triples)
        active = set(range(87, N + 1))
        
        # Try to add elements from 86 down to 1
        for x in range(86, 0, -1):
            # Check if x can be added
            conflict = False
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                # If the other two elements are in active, we have a conflict
                others = [y for y in (a, b, c) if y != x]
                if all(y in active for y in others):
                    conflict = True
                    break
            
            if not conflict:
                active.add(x)
        
        return active
    
    def run_greedy_random():
        """Greedy with random element ordering"""
        active = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        for x in order:
            conflict = False
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in active for y in others):
                    conflict = True
                    break
            
            if not conflict:
                active.add(x)
        
        return active
    
    def run_greedy_weighted():
        """Greedy that prioritizes elements with fewer conflicts"""
        active = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            # Find element with fewest conflicts with active set
            best_x = None
            best_conflicts = float('inf')
            
            for x in list(remaining):
                conflicts = 0
                for idx in elem_to_triples[x]:
                    a, b, c = triples[idx]
                    others = [y for y in (a, b, c) if y != x]
                    if all(y in active for y in others):
                        conflicts += 1
                
                if conflicts == 0:
                    best_x = x
                    best_conflicts = 0
                    break
                elif conflicts < best_conflicts:
                    best_x = x
                    best_conflicts = conflicts
            
            if best_x is not None and best_conflicts == 0:
                active.add(best_x)
                remaining.remove(best_x)
            else:
                # No element can be added without conflict, stop
                break
        
        return active

    # 3. Simulated Annealing with improved parameters
    def simulate_annealing(initial_set, time_budget, T_initial=10.0, cooling=0.9995):
        present = [False] * N
        for x in initial_set:
            present[x-1] = True
            
        triple_violation = [0] * num_triples
        total_violations = 0
        
        for idx, (a, b, c) in enumerate(triples):
            if present[a-1] and present[b-1] and present[c-1]:
                triple_violation[idx] = 1
                total_violations += 1
                
        current_score = sum(present) - 1000 * total_violations
        best_score = current_score
        best_present = present[:]
        
        T = T_initial
        start_sa = time.time()
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() - start_sa < time_budget:
            idx = random.randint(0, N - 1)
            elem = idx + 1
            old_val = present[idx]
            present[idx] = not old_val
            
            for t_idx in elem_to_triples[elem]:
                a, b, c = triples[t_idx]
                was_violation = triple_violation[t_idx]
                is_violation = present[a-1] and present[b-1] and present[c-1]
                if was_violation and not is_violation:
                    total_violations -= 1
                    triple_violation[t_idx] = 0
                elif not was_violation and is_violation:
                    total_violations += 1
                    triple_violation[t_idx] = 1
            
            new_score = sum(present) - 1000 * total_violations
            delta = new_score - current_score
            
            if delta > 0:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_present = present[:]
            else:
                if T > 0.01 and random.random() < math.exp(delta / T):
                    current_score = new_score
                    no_improve_count += 1
                else:
                    present[idx] = old_val
                    for t_idx in elem_to_triples[elem]:
                        a, b, c = triples[t_idx]
                        was_violation = triple_violation[t_idx]
                        is_violation = present[a-1] and present[b-1] and present[c-1]
                        if was_violation and not is_violation:
                            total_violations -= 1
                            triple_violation[t_idx] = 0
                        elif not was_violation and is_violation:
                            total_violations += 1
                            triple_violation[t_idx] = 1
                    no_improve_count += 1
            
            T *= cooling
            if T < 0.01:
                break
            
            # Restart if stuck too long
            if no_improve_count > max_no_improve:
                T = T_initial * 0.5
                no_improve_count = 0
                
        return best_present

    # 4. Cleanup with re-addition attempt
    def smart_cleanup(initial_set):
        active = set(initial_set)
        violations = []
        for idx, (a, b, c) in enumerate(triples):
            if a in active and b in active and c in active:
                violations.append((a, b, c))
        
        removed = []
        while violations:
            # Find element in max violations
            counts = {x: 0 for x in active}
            for (a, b, c) in violations:
                counts[a] += 1
                counts[b] += 1
                counts[c] += 1
            
            max_count = max(counts.values())
            candidates = [x for x in active if counts[x] == max_count]
            victim = min(candidates)
            active.remove(victim)
            removed.append(victim)
            
            new_violations = []
            for (a, b, c) in violations:
                if a != victim and b != victim and c != victim:
                    new_violations.append((a, b, c))
            violations = new_violations
        
        # Try to re-add removed elements
        for x in removed:
            conflict = False
            for idx in elem_to_triples[x]:
                a, b, c = triples[idx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in active for y in others):
                    conflict = True
                    break
            
            if not conflict:
                active.add(x)
        
        return active
    
    # 5. Local search polish
    def local_search_polish(initial_set, time_budget):
        """Try to add elements one by one, and remove/add pairs"""
        active = set(initial_set)
        start_time = time.time()
        
        # First, try to add any missing elements
        for x in range(1, N + 1):
            if x not in active:
                conflict = False
                for idx in elem_to_triples[x]:
                    a, b, c = triples[idx]
                    others = [y for y in (a, b, c) if y != x]
                    if all(y in active for y in others):
                        conflict = True
                        break
                
                if not conflict:
                    active.add(x)
        
        # Then try swap operations: remove one, add another
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            for remove_x in list(active):
                for add_x in range(1, N + 1):
                    if add_x in active:
                        continue
                    
                    # Check if we can remove remove_x and add add_x
                    new_active = active - {remove_x}
                    new_active.add(add_x)
                    
                    # Check for violations
                    valid = True
                    for idx, (a, b, c) in enumerate(triples):
                        if a in new_active and b in new_active and c in new_active:
                            valid = False
                            break
                    
                    if valid and len(new_active) > len(active):
                        active = new_active
                        improved = True
                        break
                if improved:
                    break
        
        return active

    best_set = set()
    best_size = 0
    
    # Strategy: Try multiple initializations and SA runs
    initializations = []
    
    # 1. Smart greedy
    greedy_set = run_smart_greedy()
    initializations.append(greedy_set)
    
    # 2. Random greedy (if time permits)
    if time.time() < deadline - 20:
        random_set = run_greedy_random()
        initializations.append(random_set)
    
    # 3. Weighted greedy (if time permits)
    if time.time() < deadline - 15:
        weighted_set = run_greedy_weighted()
        initializations.append(weighted_set)
    
    # Run SA on each initialization
    for init_set in initializations:
        if time.time() >= deadline - 10:
            break
        
        sa_budget = min(deadline - time.time() - 10, 15)
        if sa_budget < 2:
            continue
        
        best_present = simulate_annealing(init_set, sa_budget)
        sa_set = {i+1 for i, val in enumerate(best_present) if val}
        sa_set = smart_cleanup(sa_set)
        
        if len(sa_set) > best_size:
            best_size = len(sa_set)
            best_set = sa_set
    
    # Final polish with local search
    if time.time() < deadline - 5 and best_size > 0:
        polish_budget = deadline - time.time() - 5
        polished_set = local_search_polish(best_set, polish_budget)
        polished_set = smart_cleanup(polished_set)
        
        if len(polished_set) > best_size:
            best_size = len(polished_set)
            best_set = polished_set

    # Final cleanup
    final_set = smart_cleanup(best_set)
    return list(final_set)
