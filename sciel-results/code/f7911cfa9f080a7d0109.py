<<<<<<< SEARCH
    def hill_climb(initial_set, max_iters=3000, aggressive=False):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            used_primes = set()
            for a in current:
                used_primes |= prime_factors[a]
            
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                improved = True
                continue
            
            # 2. Try 1-for-2 swap: remove one, add two
            found_12 = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_pair = find_best_pair(rem_used, rem_set)
                if best_pair is not None:
                    new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                    if new_score > current_score + 1e-15:
                        current = remaining + list(best_pair)
                        current_set = rem_set | set(best_pair)
                        current_score = new_score
                        improved = True
                        found_12 = True
                        break
            
            if improved:
                continue
            
            # 3. Try 1-for-1 swap
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = set()
                for a in remaining:
                    rem_used |= prime_factors[a]
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # 4. Try 2-for-1 swap (MERGE): remove two, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = set()
                    for a in remaining:
                        rem_used |= prime_factors[a]
                    
                    best_add = find_best_add(rem_used, rem_set)
                    if best_add is not None:
                        new_score = rem_score + scores[best_add]
                        if new_score > current_score + 1e-15:
                            current = remaining + [best_add]
                            current_set = rem_set | {best_add}
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            # 5. Try 2-for-2 swap
            if aggressive and not improved:
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        
                        rem_used = set()
                        for a in remaining:
                            rem_used |= prime_factors[a]
                        
                        best_pair = find_best_pair(rem_used, rem_set)
                        if best_pair is not None:
                            new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                            if new_score > current_score + 1e-15:
                                current = remaining + list(best_pair)
                                current_set = rem_set | set(best_pair)
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_score
=======
    def compute_used_primes(lst):
        """Compute union of prime factors for a list of numbers."""
        used = set()
        for a in lst:
            used |= prime_factors[a]
        return used
    
    def hill_climb(initial_set, max_iters=3000, aggressive=False):
        current = list(initial_set)
        current_score = get_score(current)
        current_set = set(current)
        used_primes = compute_used_primes(current)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try adding any a not in current that is coprime
            best_add = find_best_add(used_primes, current_set)
            if best_add is not None:
                current.append(best_add)
                current_set.add(best_add)
                current_score += scores[best_add]
                used_primes |= prime_factors[best_add]
                improved = True
                continue
            
            # 2. Try 1-for-2 swap: remove one, add two
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                # Recompute used primes excluding 'removed'
                rem_used = compute_used_primes(remaining)
                
                best_pair = find_best_pair(rem_used, rem_set)
                if best_pair is not None:
                    new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                    if new_score > current_score + 1e-15:
                        current = remaining + list(best_pair)
                        current_set = rem_set | set(best_pair)
                        current_score = new_score
                        used_primes = rem_used | prime_factors[best_pair[0]] | prime_factors[best_pair[1]]
                        improved = True
                        break
            
            if improved:
                continue
            
            # 3. Try 1-for-1 swap
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_set = set(remaining)
                rem_score = current_score - scores[removed]
                
                rem_used = compute_used_primes(remaining)
                
                best_add = find_best_add(rem_used, rem_set)
                if best_add is not None:
                    new_score = rem_score + scores[best_add]
                    if new_score > current_score + 1e-15:
                        current = remaining + [best_add]
                        current_set = rem_set | {best_add}
                        current_score = new_score
                        used_primes = rem_used | prime_factors[best_add]
                        improved = True
                        break
            
            if improved:
                continue
            
            # 4. Try 2-for-1 swap (MERGE): remove two, add one
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        break
                    removed1 = current[i]
                    removed2 = current[j]
                    remaining = [x for k, x in enumerate(current) if k != i and k != j]
                    rem_set = set(remaining)
                    rem_score = current_score - scores[removed1] - scores[removed2]
                    
                    rem_used = compute_used_primes(remaining)
                    
                    best_add = find_best_add(rem_used, rem_set)
                    if best_add is not None:
                        new_score = rem_score + scores[best_add]
                        if new_score > current_score + 1e-15:
                            current = remaining + [best_add]
                            current_set = rem_set | {best_add}
                            current_score = new_score
                            used_primes = rem_used | prime_factors[best_add]
                            improved = True
                            break
                if improved:
                    break
            
            # 5. Try 2-for-2 swap
            if aggressive and not improved:
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline:
                            break
                        removed1 = current[i]
                        removed2 = current[j]
                        remaining = [x for k, x in enumerate(current) if k != i and k != j]
                        rem_set = set(remaining)
                        rem_score = current_score - scores[removed1] - scores[removed2]
                        
                        rem_used = compute_used_primes(remaining)
                        
                        best_pair = find_best_pair(rem_used, rem_set)
                        if best_pair is not None:
                            new_score = rem_score + scores[best_pair[0]] + scores[best_pair[1]]
                            if new_score > current_score + 1e-15:
                                current = remaining + list(best_pair)
                                current_set = rem_set | set(best_pair)
                                current_score = new_score
                                used_primes = rem_used | prime_factors[best_pair[0]] | prime_factors[best_pair[1]]
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_score
>>>>>>> REPLACE

