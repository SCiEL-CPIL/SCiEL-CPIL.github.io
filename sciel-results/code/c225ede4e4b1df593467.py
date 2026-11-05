def construct():
    import time
    import random
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 180
    
    # Precompute all forbidden triples and build efficient conflict structures
    # constraint[x] = set of pairs (b, c) such that {x, b, c} is a forbidden triple
    # For fast checking: for each x, precompute which pairs in the set would conflict
    
    constraint = [set() for _ in range(N + 1)]
    
    for b in range(2, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    # a = bc/(b+c) < min(b,c) when b != c, so a is smallest
                    constraint[a].add((b, c))
                    constraint[b].add((a, c))
                    constraint[c].add((a, b))
    
    def is_valid(s):
        """Check if set s has no forbidden triple."""
        s_list = sorted(s)
        n = len(s_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = s_list[i], s_list[j], s_list[k]
                    if b * c == a * (b + c):
                        return False
        return True
    
    def can_add(x, s):
        """Check if x can be added to set s without creating a forbidden triple."""
        for (b, c) in constraint[x]:
            if b in s and c in s:
                return False
        return True
    
    def count_violations(x, s):
        """Count how many pairs (b,c) in constraint[x] both exist in s."""
        count = 0
        for (b, c) in constraint[x]:
            if b in s and c in s:
                count += 1
        return count
    
    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy from large end
    current_set = set()
    # Start with elements that have fewest constraints
    elems = list(range(1, N + 1))
    elems.sort(key=lambda x: len(constraint[x]))
    
    for x in elems:
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x, current_set):
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
    
    # Strategy 2: Greedy from small end
    current_set = set()
    for x in range(1, N + 1):
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x, current_set):
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
    
    # Strategy 3: Start with upper half, greedily add lower
    current_set = set(range(91, N + 1))
    candidates = list(range(1, 91))
    candidates.sort(key=lambda x: count_violations(x, current_set))
    
    for x in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if can_add(x, current_set):
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
    
    # Local search: hill climbing with swaps
    current_set = set(best_set)
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        for remove_elem in list(current_set):
            if time.time() - start_time > DEADLINE:
                break
            trial_set = current_set - {remove_elem}
            # Try all possible additions
            for add_elem in range(1, N + 1):
                if add_elem in trial_set:
                    continue
                if can_add(add_elem, trial_set):
                    new_size = len(trial_set) + 1
                    if new_size > len(current_set):
                        current_set = trial_set | {add_elem}
                        improved = True
                        if len(current_set) > best_size:
                            best_set = set(current_set)
                            best_size = len(current_set)
                        break
    
    # Strategy 4: Randomized restarts with simulated annealing
    random.seed(42)
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random initialization
        current_set = set()
        elems = list(range(1, N + 1))
        random.shuffle(elems)
        
        for x in elems:
            if can_add(x, current_set):
                current_set.add(x)
        
        # Simulated annealing
        T = 10.0
        cooling = 0.95
        for step in range(500):
            if time.time() - start_time > DEADLINE:
                break
            
            if T < 0.1:
                break
            
            # Random move: add, remove, or swap
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Try to add a random element not in set
                not_in_set = [x for x in range(1, N + 1) if x not in current_set]
                if not not_in_set:
                    continue
                x = random.choice(not_in_set)
                if can_add(x, current_set):
                    current_set.add(x)
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(current_set)
                else:
                    # Accept with probability based on violation count
                    v = count_violations(x, current_set)
                    if v > 0 and random.random() < (1.0 / (1.0 + v)) * T:
                        current_set.add(x)
            
            elif move_type == 'remove':
                if current_set:
                    x = random.choice(list(current_set))
                    current_set.remove(x)
            
            elif move_type == 'swap':
                if current_set:
                    remove_x = random.choice(list(current_set))
                    not_in_set = [x for x in range(1, N + 1) if x not in current_set]
                    if not_in_set:
                        add_x = random.choice(not_in_set)
                        trial_set = (current_set - {remove_x}) | {add_x}
                        # Check validity
                        valid = True
                        for (b, c) in constraint[add_x]:
                            if b in trial_set and c in trial_set:
                                valid = False
                                break
                        if valid:
                            current_set = trial_set
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
            
            T *= cooling
        
        # Local search polish
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for remove_elem in list(current_set):
                if time.time() - start_time > DEADLINE:
                    break
                trial_set = current_set - {remove_elem}
                for add_elem in range(1, N + 1):
                    if add_elem in trial_set:
                        continue
                    if can_add(add_elem, trial_set):
                        if len(trial_set) + 1 > len(current_set):
                            current_set = trial_set | {add_elem}
                            improved = True
                            if len(current_set) > best_size:
                                best_set = set(current_set)
                                best_size = len(current_set)
                            break
    
    # Final verification
    if not is_valid(best_set):
        # Fallback to a known valid set
        best_set = set(range(91, N + 1))
    
    return sorted(best_set)
