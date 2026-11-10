import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: for each element x, store set of frozenset({a,b})
    # such that {x,a,b} is a bad triple (all three pairwise LCMs equal)
    conflict = [set() for _ in range(n + 1)]
    
    # Direct triple enumeration
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict[a].add(frozenset((b, c)))
                    conflict[b].add(frozenset((a, c)))
                    conflict[c].add(frozenset((a, b)))
    
    def count_conflicts(x, selected_set):
        """Count how many bad triples would be created by adding x to selected_set."""
        cnt = 0
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                cnt += 1
        return cnt
    
    def is_valid(x, selected_set):
        """Check if x can be added to selected_set without creating a bad triple."""
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def count_conflicts_excluding(x, selected_set, excluded_y):
        """Check if adding x to (selected_set - {excluded_y}) creates a bad triple."""
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                if a != excluded_y and b != excluded_y:
                    return 1
        return 0
    
    def count_conflicts_excluding2(x, selected_set, excluded_y1, excluded_y2):
        """Check if adding x to (selected_set - {y1, y2}) creates a bad triple."""
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a in selected_set and b in selected_set:
                if a != excluded_y1 and b != excluded_y1 and a != excluded_y2 and b != excluded_y2:
                    return 1
        return 0
    
    best_result = set()
    
    # Phase A: Greedy-add with multiple orderings
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: -x.bit_count()),
        sorted(range(1, n + 1), key=lambda x: x.bit_count()),
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
        sorted(range(1, n + 1), key=lambda x: len(conflict[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict[x])),
        sorted(range(1, n + 1), key=lambda x: (len(conflict[x]), x)),
        [2**k for k in range(0, 8) if 2**k <= n] + [x for x in range(1, n+1) if x not in {2**k for k in range(0, 8) if 2**k <= n}],
        sorted(range(1, n + 1), key=lambda x: (x & -x), reverse=True),  # by 2-adic valuation
        sorted(range(1, n + 1), key=lambda x: (x & -x)),
    ]
    
    for order in orders:
        if time.time() > deadline - 25:
            break
        selected = set()
        for x in order:
            if is_valid(x, selected):
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Randomized greedy restarts
    rng = random.Random(42)
    for restart in range(80):
        if time.time() > deadline - 25:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = set()
        for x in order:
            if is_valid(x, selected):
                selected.add(x)
        if len(selected) > len(best_result):
            best_result = selected.copy()
    
    # Phase B: Simulated annealing on the 3-uniform hypergraph
    # State: subset S of [1,n]. Objective: maximize |S| - penalty * (number of bad triples in S)
    # We allow invalid intermediate states and use a penalty to guide back to validity.
    
    def count_bad_triples(selected_set):
        """Count the number of bad triples within selected_set."""
        cnt = 0
        sel_list = list(selected_set)
        for i, a in enumerate(sel_list):
            for b in sel_list[i+1:]:
                for c in sel_list[i+1:]:
                    if c <= b:
                        continue
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        cnt += 1
        return cnt
    
    def count_bad_triples_fast(selected_set):
        """Count bad triples using conflict structure."""
        cnt = 0
        for x in selected_set:
            for pair in conflict[x]:
                a, b = tuple(pair)
                if a in selected_set and b in selected_set:
                    # Each triple counted 3 times (once per element)
                    pass
        # More careful: iterate over all triples
        sel_list = sorted(selected_set)
        cnt = 0
        for i in range(len(sel_list)):
            a = sel_list[i]
            for j in range(i+1, len(sel_list)):
                b = sel_list[j]
                L = lcm_table[a][b]
                for k in range(j+1, len(sel_list)):
                    c = sel_list[k]
                    if lcm_table[a][c] == L and lcm_table[b][c] == L:
                        cnt += 1
        return cnt
    
    def count_bad_triples_conflict(selected_set):
        """Count bad triples using the conflict structure (each triple counted 3x, divide by 3)."""
        cnt = 0
        for x in selected_set:
            for pair in conflict[x]:
                a, b = tuple(pair)
                if a in selected_set and b in selected_set:
                    cnt += 1
        return cnt // 3
    
    # Simulated annealing
    if time.time() < deadline - 5:
        current = best_result.copy()
        current_size = len(current)
        current_bad = count_bad_triples_conflict(current)
        
        # Objective: size - penalty * bad_triples
        # We want to maximize size while minimizing bad triples
        penalty = 5.0
        T_start = 3.0
        T_end = 0.01
        cooling_rate = 0.995
        
        best_sa = current.copy()
        best_sa_score = len(current) - penalty * current_bad
        
        T = T_start
        sa_iter = 0
        max_sa_iters = 50000
        
        while T > T_end and sa_iter < max_sa_iters and time.time() < deadline - 5:
            # Random move: add, remove, or swap
            move_type = rng.random()
            if move_type < 0.3:
                # Add a random element not in current
                candidates = [x for x in range(1, n+1) if x not in current]
                if not candidates:
                    sa_iter += 1
                    continue
                x = rng.choice(candidates)
                new_set = current | {x}
            elif move_type < 0.6:
                # Remove a random element from current
                if not current:
                    sa_iter += 1
                    continue
                x = rng.choice(list(current))
                new_set = current - {x}
            else:
                # Swap: remove one, add one
                if not current:
                    sa_iter += 1
                    continue
                y = rng.choice(list(current))
                candidates = [x for x in range(1, n+1) if x not in current]
                if not candidates:
                    sa_iter += 1
                    continue
                x = rng.choice(candidates)
                new_set = (current - {y}) | {x}
            
            new_size = len(new_set)
            new_bad = count_bad_triples_conflict(new_set)
            
            current_score = current_size - penalty * current_bad
            new_score = new_size - penalty * new_bad
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and rng.random() < pow(2.718281828, delta / T)):
                current = new_set
                current_size = new_size
                current_bad = new_bad
                
                if len(current) - penalty * current_bad > best_sa_score:
                    best_sa_score = len(current) - penalty * current_bad
                    best_sa = current.copy()
            
            T *= cooling_rate
            sa_iter += 1
        
        # After SA, if best_sa is valid, use it
        if count_bad_triples_conflict(best_sa) == 0 and len(best_sa) > len(best_result):
            best_result = best_sa.copy()
        
        # Also try SA starting from other good solutions
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            order = list(range(1, n + 1))
            rng.shuffle(order)
            selected = set()
            for x in order:
                if is_valid(x, selected):
                    selected.add(x)
            
            current = selected.copy()
            current_size = len(current)
            current_bad = count_bad_triples_conflict(current)
            
            best_sa_local = current.copy()
            best_sa_local_score = len(current) - penalty * current_bad
            
            T = T_start
            sa_iter = 0
            max_sa_iters = 20000
            
            while T > T_end and sa_iter < max_sa_iters and time.time() < deadline - 5:
                move_type = rng.random()
                if move_type < 0.3:
                    candidates = [x for x in range(1, n+1) if x not in current]
                    if not candidates:
                        sa_iter += 1
                        continue
                    x = rng.choice(candidates)
                    new_set = current | {x}
                elif move_type < 0.6:
                    if not current:
                        sa_iter += 1
                        continue
                    x = rng.choice(list(current))
                    new_set = current - {x}
                else:
                    if not current:
                        sa_iter += 1
                        continue
                    y = rng.choice(list(current))
                    candidates = [x for x in range(1, n+1) if x not in current]
                    if not candidates:
                        sa_iter += 1
                        continue
                    x = rng.choice(candidates)
                    new_set = (current - {y}) | {x}
                
                new_size = len(new_set)
                new_bad = count_bad_triples_conflict(new_set)
                
                current_score = current_size - penalty * current_bad
                new_score = new_size - penalty * new_bad
                
                delta = new_score - current_score
                
                if delta > 0 or (T > 0 and rng.random() < pow(2.718281828, delta / T)):
                    current = new_set
                    current_size = new_size
                    current_bad = new_bad
                    
                    if len(current) - penalty * current_bad > best_sa_local_score:
                        best_sa_local_score = len(current) - penalty * current_bad
                        best_sa_local = current.copy()
                
                T *= cooling_rate
                sa_iter += 1
            
            if count_bad_triples_conflict(best_sa_local) == 0 and len(best_sa_local) > len(best_result):
                best_result = best_sa_local.copy()
    
    # Phase C: Local search refinement on best_result
    current = best_result.copy()
    
    # Phase C1: Try to add any element that fits (multiple passes)
    improved = True
    while improved and time.time() < deadline - 8:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if is_valid(x, current):
                current.add(x)
                improved = True
                break
    
    # Phase C2: 1-for-1 Swaps
    swap_attempts = 0
    max_swap_attempts = 30000
    while time.time() < deadline - 6 and swap_attempts < max_swap_attempts:
        swap_found = False
        for y in sorted(list(current), key=lambda y: -len(conflict[y])):
            if swap_attempts >= max_swap_attempts or time.time() > deadline - 6:
                break
            for x in range(1, n + 1):
                if x in current:
                    continue
                swap_attempts += 1
                if swap_attempts > max_swap_attempts or time.time() > deadline - 6:
                    break
                if count_conflicts_excluding(x, current, y) == 0:
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found or swap_attempts >= max_swap_attempts:
                break
        if not swap_found:
            break
    
    # Phase C3: Try 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 5:
        for x1 in range(1, n + 1):
            if x1 in current or time.time() > deadline - 5:
                continue
            for x2 in range(x1 + 1, n + 1):
                if x2 in current or time.time() > deadline - 5:
                    continue
                for y in list(current):
                    if count_conflicts_excluding(x1, current, y) == 0 and count_conflicts_excluding(x2, current - {y}, y) == 0:
                        current.remove(y)
                        current.add(x1)
                        current.add(x2)
                        break
    
    # Phase C4: Try 2-for-2 swaps (remove 2, add 2)
    if time.time() < deadline - 4:
        for y1 in range(1, n + 1):
            if y1 not in current or time.time() > deadline - 4:
                continue
            for y2 in range(y1 + 1, n + 1):
                if y2 not in current or time.time() > deadline - 4:
                    continue
                for x1 in range(1, n + 1):
                    if x1 in current or time.time() > deadline - 4:
                        continue
                    for x2 in range(x1 + 1, n + 1):
                        if x2 in current or time.time() > deadline - 4:
                            continue
                        temp = current - {y1, y2}
                        if count_conflicts(x1, temp) == 0 and count_conflicts(x2, temp | {x1}) == 0:
                            current = temp | {x1, x2}
                            break
                    if time.time() > deadline - 4:
                        break
                if time.time() > deadline - 4:
                    break
            if time.time() > deadline - 4:
                break
    
    # Phase C5: Final add pass
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if is_valid(x, current):
                current.add(x)
                improved = True
                break
    
    # Phase C6: More aggressive 1-for-1 swaps after additions
    if time.time() < deadline - 1:
        swap_found = True
        while swap_found and time.time() < deadline - 1:
            swap_found = False
            for y in list(current):
                if time.time() > deadline - 1:
                    break
                for x in range(1, n + 1):
                    if x in current:
                        continue
                    if count_conflicts_excluding(x, current, y) == 0:
                        current.remove(y)
                        current.add(x)
                        swap_found = True
                        break
                if swap_found:
                    break
    
    # Phase D: One more round of SA on the refined solution
    if time.time() < deadline - 2:
        current = best_result.copy()
        current_size = len(current)
        current_bad = count_bad_triples_conflict(current)
        
        best_sa2 = current.copy()
        best_sa2_score = len(current) - penalty * current_bad
        
        T = 1.5
        sa_iter = 0
        max_sa_iters = 30000
        
        while T > 0.05 and sa_iter < max_sa_iters and time.time() < deadline - 2:
            move_type = rng.random()
            if move_type < 0.3:
                candidates = [x for x in range(1, n+1) if x not in current]
                if not candidates:
                    sa_iter += 1
                    continue
                x = rng.choice(candidates)
                new_set = current | {x}
            elif move_type < 0.6:
                if not current:
                    sa_iter += 1
                    continue
                x = rng.choice(list(current))
                new_set = current - {x}
            else:
                if not current:
                    sa_iter += 1
                    continue
                y = rng.choice(list(current))
                candidates = [x for x in range(1, n+1) if x not in current]
                if not candidates:
                    sa_iter += 1
                    continue
                x = rng.choice(candidates)
                new_set = (current - {y}) | {x}
            
            new_size = len(new_set)
            new_bad = count_bad_triples_conflict(new_set)
            
            current_score = current_size - penalty * current_bad
            new_score = new_size - penalty * new_bad
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and rng.random() < pow(2.718281828, delta / T)):
                current = new_set
                current_size = new_size
                current_bad = new_bad
                
                if len(current) - penalty * current_bad > best_sa2_score:
                    best_sa2_score = len(current) - penalty * current_bad
                    best_sa2 = current.copy()
            
            T *= 0.997
            sa_iter += 1
        
        if count_bad_triples_conflict(best_sa2) == 0 and len(best_sa2) > len(current):
            current = best_sa2.copy()
        
        # Final validity check and add pass
        if count_bad_triples_conflict(current) > 0:
            # Remove elements to make valid
            changed = True
            while changed and time.time() < deadline - 0.5:
                changed = False
                for x in list(current):
                    if count_conflicts(x, current - {x}) > 0:
                        current.remove(x)
                        changed = True
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for x in range(1, n + 1):
                if x in current:
                    continue
                if is_valid(x, current):
                    current.add(x)
                    improved = True
                    break
    
    # Final validity check
    if count_bad_triples_conflict(current) > 0:
        # Fallback: use best_result which should be valid
        current = best_result.copy()
    
    # Final add pass
    improved = True
    while improved and time.time() < deadline - 0.3:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if is_valid(x, current):
                current.add(x)
                improved = True
                break
    
    return list(current) if current else list(best_result)
