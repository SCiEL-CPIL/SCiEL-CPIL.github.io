import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 128
    
    def is_valid(S):
        """Check if set S (as a set of ints) has no forbidden triple."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a >= b:
                    continue
                for c in S:
                    if c <= b:
                        continue
                    # Check if a+b, a+c, b+c are all in S
                    if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                        return False
        return True
    
    def check_add(x, L_set, U_set, S_set):
        """Check if adding x to L creates any forbidden triple."""
        new_L = L_set | {x}
        new_S = S_set | {x}
        
        # Check Type A: three from new_L
        L_list = sorted(new_L)
        for i in range(len(L_list)):
            a = L_list[i]
            for j in range(i+1, len(L_list)):
                b = L_list[j]
                ab = a + b
                if ab not in new_S:
                    continue
                for k in range(j+1, len(L_list)):
                    c = L_list[k]
                    if (a + c) in new_S and (b + c) in new_S:
                        return False
        
        # Check Type B: two from new_L, one from U_set
        for i in range(len(L_list)):
            a = L_list[i]
            for j in range(i+1, len(L_list)):
                b = L_list[j]
                ab = a + b
                if ab not in new_S:
                    continue
                for c in U_set:
                    if (a + c) in new_S and (b + c) in new_S:
                        return False
        
        return True

    def local_search_polish(current, max_iters=500):
        """Local search to try to increase size by swapping elements."""
        current = set(current)
        L_set = current & set(range(1, 65))
        U_set = current & set(range(65, 129))
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start_time < DEADLINE:
            improved = False
            iters += 1
            # Try each L candidate not in current
            for x in range(1, 65):
                if x in current:
                    continue
                if check_add(x, L_set, U_set, current):
                    current.add(x)
                    L_set.add(x)
                    improved = True
                    break
            
            if not improved:
                # Try removing one U element and adding one or more L elements
                for u in sorted(U_set):
                    if time.time() - start_time > DEADLINE:
                        break
                    new_U = U_set - {u}
                    new_L = set(L_set)
                    new_S = current - {u}
                    added = True
                    while added and time.time() - start_time < DEADLINE:
                        added = False
                        for x in range(1, 65):
                            if x in new_S:
                                continue
                            if check_add(x, new_L, new_U, new_S):
                                new_L.add(x)
                                new_S.add(x)
                                added = True
                                break
                    if len(new_S) > len(current):
                        current = new_S
                        L_set = new_L
                        U_set = new_U
                        improved = True
                        break
            
            if not improved:
                # Try removing two U elements and adding L elements
                U_list = sorted(U_set)
                for i in range(len(U_list)):
                    if time.time() - start_time > DEADLINE:
                        break
                    for j in range(i+1, len(U_list)):
                        if time.time() - start_time > DEADLINE:
                            break
                        u1, u2 = U_list[i], U_list[j]
                        new_U = U_set - {u1, u2}
                        new_L = set(L_set)
                        new_S = current - {u1, u2}
                        added = True
                        while added and time.time() - start_time < DEADLINE:
                            added = False
                            for x in range(1, 65):
                                if x in new_S:
                                    continue
                                if check_add(x, new_L, new_U, new_S):
                                    new_L.add(x)
                                    new_S.add(x)
                                    added = True
                                    break
                        if len(new_S) > len(current):
                            current = new_S
                            L_set = new_L
                            U_set = new_U
                            improved = True
                            break
                    if improved:
                        break
        
        return current

    best_result = set(range(65, 129))
    best_size = 64
    
    candidates = list(range(1, 65))
    
    # Phase 1: Multiple random greedy constructions
    for trial in range(400):
        if time.time() - start_time > DEADLINE:
            break
        
        random.shuffle(candidates)
        L_set = set()
        U_set = set(range(65, 129))
        S_set = set(range(65, 129))
        
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if check_add(x, L_set, U_set, S_set):
                L_set.add(x)
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
    
    # Phase 2: Try with reduced U sets
    for trial in range(100):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove 2-5 random U elements
        n_remove = random.randint(2, 5)
        U_remove = set(random.sample(range(65, 129), n_remove))
        U_set = set(range(65, 129)) - U_remove
        S_set = set(U_set)
        L_set = set()
        
        random.shuffle(candidates)
        for x in candidates:
            if time.time() - start_time > DEADLINE:
                break
            if check_add(x, L_set, U_set, S_set):
                L_set.add(x)
                S_set.add(x)
        
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_result = set(S_set)
    
    # Phase 3: Local search polish on best result
    if time.time() - start_time < DEADLINE:
        polished = local_search_polish(best_result)
        if len(polished) > best_size:
            best_size = len(polished)
            best_result = polished
    
    # Phase 4: Simulated annealing-like refinement
    # Try random perturbations of the best set
    if time.time() - start_time < DEADLINE:
        current_best = set(best_result)
        temperature = 10.0
        cooling = 0.95
        
        while temperature > 0.1 and time.time() - start_time < DEADLINE:
            # Pick a random element to remove
            elements = list(current_best)
            if not elements:
                break
            remove_elem = random.choice(elements)
            
            # Try to add 1-2 random L elements
            new_set = set(current_best) - {remove_elem}
            L_set = new_set & set(range(1, 65))
            U_set = new_set & set(range(65, 129))
            
            added_count = 0
            for _ in range(random.randint(1, 2)):
                candidates = [x for x in range(1, 65) if x not in new_set]
                if not candidates:
                    break
                x = random.choice(candidates)
                if check_add(x, L_set, U_set, new_set):
                    new_set.add(x)
                    L_set.add(x)
                    added_count += 1
            
            # Accept if better, or with probability based on temperature
            if len(new_set) > len(current_best):
                current_best = new_set
                if len(current_best) > best_size:
                    best_size = len(current_best)
                    best_result = set(current_best)
            elif len(new_set) == len(current_best):
                # Equal size, accept with some probability to escape local optima
                if random.random() < 0.1:
                    current_best = new_set
            else:
                # Worse, accept with probability exp(-delta/temperature)
                delta = len(current_best) - len(new_set)
                if random.random() < pow(2.71828, -delta / temperature):
                    current_best = new_set
            
            temperature *= cooling
    
    result = sorted(best_result)
    # Ensure all are Python ints
    result = [int(x) for x in result]
    return result

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Set: {result[:20]}...{result[-20:]}")
    
    # Verify
    S = set(result)
    valid = True
    for a in result:
        for b in result:
            if a >= b:
                continue
            for c in result:
                if c <= b:
                    continue
                if (a+b) in S and (a+c) in S and (b+c) in S:
                    valid = False
                    print(f"Forbidden triple: {a},{b},{c}")
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid: {valid}")
