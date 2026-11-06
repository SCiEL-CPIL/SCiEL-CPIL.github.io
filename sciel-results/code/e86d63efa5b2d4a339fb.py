"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    from math import gcd
    import random
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    N = 512
    # Precompute score contribution
    score_contrib = [0.0] * N
    for a in range(1, N):
        score_contrib[a] = 1.0 / (N - a)
    
    def is_coprime_to_all(a, chosen_list):
        for b in chosen_list:
            if gcd(a, b) != 1:
                return False
        return True
    
    # Phase 1: Greedy initialization - pick from high to low
    chosen = []
    chosen_set = set()
    for a in range(511, 0, -1):
        if is_coprime_to_all(a, chosen):
            chosen.append(a)
            chosen_set.add(a)
    
    def current_score():
        return sum(score_contrib[a] for a in chosen)
    
    best_score = current_score()
    best_chosen = chosen[:]
    
    def try_addition(chosen, chosen_set, best_score, best_chosen):
        """Try to add any unchosen element that improves score."""
        for a in range(511, 0, -1):
            if a not in chosen_set:
                if is_coprime_to_all(a, chosen):
                    new_score = current_score() + score_contrib[a]
                    if new_score > best_score + 1e-15:
                        chosen.append(a)
                        chosen_set.add(a)
                        best_score = new_score
                        best_chosen = chosen[:]
                        return True, best_score, best_chosen
        return False, best_score, best_chosen
    
    def try_1swap(chosen, chosen_set, best_score, best_chosen):
        """Try to swap one chosen element with one unchosen element."""
        for i in range(len(chosen)):
            if time.time() >= deadline:
                break
            old_a = chosen[i]
            for new_a in range(511, 0, -1):
                if new_a in chosen_set:
                    continue
                # Check if new_a is coprime to all other chosen elements
                coprime = True
                for j in range(len(chosen)):
                    if j == i:
                        continue
                    if gcd(new_a, chosen[j]) != 1:
                        coprime = False
                        break
                if not coprime:
                    continue
                delta = score_contrib[new_a] - score_contrib[old_a]
                if delta > 1e-15:
                    # Perform swap
                    chosen[i] = new_a
                    chosen_set.remove(old_a)
                    chosen_set.add(new_a)
                    best_score += delta
                    best_chosen = chosen[:]
                    return True, best_score, best_chosen
        return False, best_score, best_chosen
    
    def try_2swap(chosen, chosen_set, best_score, best_chosen):
        """Try to swap two chosen elements with two unchosen elements."""
        n = len(chosen)
        for i in range(n):
            if time.time() >= deadline:
                break
            for j in range(i+1, n):
                if time.time() >= deadline:
                    break
                old_a = chosen[i]
                old_b = chosen[j]
                # Build list of other chosen elements
                others = [chosen[k] for k in range(n) if k != i and k != j]
                for new_a in range(511, 0, -1):
                    if new_a in chosen_set:
                        continue
                    if gcd(new_a, old_b) != 1:
                        continue
                    if not is_coprime_to_all(new_a, others):
                        continue
                    for new_b in range(511, 0, -1):
                        if new_b in chosen_set or new_b == new_a:
                            continue
                        if gcd(new_b, old_a) != 1:
                            continue
                        if gcd(new_b, new_a) != 1:
                            continue
                        if not is_coprime_to_all(new_b, others):
                            continue
                        delta = (score_contrib[new_a] + score_contrib[new_b]) - (score_contrib[old_a] + score_contrib[old_b])
                        if delta > 1e-15:
                            # Perform 2-swap
                            chosen[i] = new_a
                            chosen[j] = new_b
                            chosen_set.remove(old_a)
                            chosen_set.remove(old_b)
                            chosen_set.add(new_a)
                            chosen_set.add(new_b)
                            best_score += delta
                            best_chosen = chosen[:]
                            return True, best_score, best_chosen
        return False, best_score, best_chosen
    
    # Phase 2: Local search with 1-swaps and 2-swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try additions
        add_ok, best_score, best_chosen = try_addition(chosen, chosen_set, best_score, best_chosen)
        if add_ok:
            improved = True
            continue
        
        # Try 1-swaps
        swap1_ok, best_score, best_chosen = try_1swap(chosen, chosen_set, best_score, best_chosen)
        if swap1_ok:
            improved = True
            continue
        
        # Try 2-swaps
        swap2_ok, best_score, best_chosen = try_2swap(chosen, chosen_set, best_score, best_chosen)
        if swap2_ok:
            improved = True
            continue
    
    # Phase 3: Simulated annealing to escape local optima
    if time.time() < deadline - 20:  # Leave 20s safety margin
        current_temp = 0.01
        cooling_rate = 0.999
        min_temp = 1e-6
        random.seed(42)
        
        while current_temp > min_temp and time.time() < deadline - 10:
            # Randomly select a move type
            move_type = random.random()
            if move_type < 0.1 and len(chosen) > 0:
                # Random removal
                idx = random.randint(0, len(chosen)-1)
                old_a = chosen[idx]
                chosen.pop(idx)
                chosen_set.remove(old_a)
                new_score = current_score()
                delta = new_score - best_score
                if delta > 0 or random.random() < pow(2.718281828, delta / current_temp):
                    best_score = new_score
                    best_chosen = chosen[:]
                else:
                    # Revert
                    chosen.append(old_a)
                    chosen_set.add(old_a)
            elif move_type < 0.5:
                # Random 1-swap
                if len(chosen) > 0:
                    i = random.randint(0, len(chosen)-1)
                    candidates = [a for a in range(1, N) if a not in chosen_set]
                    if candidates:
                        new_a = random.choice(candidates)
                        old_a = chosen[i]
                        # Check coprime
                        valid = True
                        for j in range(len(chosen)):
                            if j != i and gcd(new_a, chosen[j]) != 1:
                                valid = False
                                break
                        if valid:
                            chosen[i] = new_a
                            chosen_set.remove(old_a)
                            chosen_set.add(new_a)
                            new_score = current_score()
                            delta = new_score - best_score
                            if delta > 0 or random.random() < pow(2.718281828, delta / current_temp):
                                best_score = new_score
                                best_chosen = chosen[:]
                            else:
                                # Revert
                                chosen[i] = old_a
                                chosen_set.remove(new_a)
                                chosen_set.add(old_a)
            else:
                # Random 2-swap
                if len(chosen) >= 2:
                    i, j = random.sample(range(len(chosen)), 2)
                    candidates = [a for a in range(1, N) if a not in chosen_set]
                    if len(candidates) >= 2:
                        new_a, new_b = random.sample(candidates, 2)
                        old_a, old_b = chosen[i], chosen[j]
                        others = [chosen[k] for k in range(len(chosen)) if k != i and k != j]
                        valid = (gcd(new_a, new_b) == 1 and 
                                 gcd(new_a, old_b) == 1 and 
                                 gcd(new_b, old_a) == 1 and
                                 is_coprime_to_all(new_a, others) and
                                 is_coprime_to_all(new_b, others))
                        if valid:
                            chosen[i] = new_a
                            chosen[j] = new_b
                            chosen_set.remove(old_a)
                            chosen_set.remove(old_b)
                            chosen_set.add(new_a)
                            chosen_set.add(new_b)
                            new_score = current_score()
                            delta = new_score - best_score
                            if delta > 0 or random.random() < pow(2.718281828, delta / current_temp):
                                best_score = new_score
                                best_chosen = chosen[:]
                            else:
                                # Revert
                                chosen[i] = old_a
                                chosen[j] = old_b
                                chosen_set.remove(new_a)
                                chosen_set.remove(new_b)
                                chosen_set.add(old_a)
                                chosen_set.add(old_b)
            
            current_temp *= cooling_rate
    
    # Final polish: run local search again on best solution
    if time.time() < deadline - 5:
        chosen = best_chosen[:]
        chosen_set = set(chosen)
        best_score = current_score()
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            add_ok, best_score, best_chosen = try_addition(chosen, chosen_set, best_score, best_chosen)
            if add_ok:
                improved = True
                continue
            swap1_ok, best_score, best_chosen = try_1swap(chosen, chosen_set, best_score, best_chosen)
            if swap1_ok:
                improved = True
                continue
            swap2_ok, best_score, best_chosen = try_2swap(chosen, chosen_set, best_score, best_chosen)
            if swap2_ok:
                improved = True
                continue
    
    # Return the best solution found
    return best_chosen
# EVOLVE-BLOCK-END
