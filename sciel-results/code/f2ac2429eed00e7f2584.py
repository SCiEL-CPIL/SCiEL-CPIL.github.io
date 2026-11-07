import time
import random

def construct():
    deadline = time.time() + 55
    candidates = list(range(2, 41))
    
    # Precompute prime exponent vectors for each candidate
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    num_primes = len(primes)
    
    def factorize(n):
        vec = [0] * num_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                vec[i] += 1
                n //= p
        return tuple(vec)
    
    exp_vecs = {x: factorize(x) for x in candidates}
    zero_vec = (0,) * num_primes
    
    # Direct construction with primes as initial best
    direct_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    best_set = direct_primes[:]
    best_size = len(best_set)
    
    def vec_add(a, b):
        return tuple(ai + bi for ai, bi in zip(a, b))
    
    def is_valid_with_vecs(current_vecs, x_vec):
        """Check if x_vec can be added given current set of subset sum vectors."""
        for v in current_vecs:
            if vec_add(v, x_vec) in current_vecs:
                return False
        return True
    
    def add_to_vecs(current_vecs, x_vec):
        """Add x_vec to the set, returning new vector set."""
        new_vecs = {vec_add(v, x_vec) for v in current_vecs}
        current_vecs.update(new_vecs)
        return current_vecs
    
    # Greedy with multiple orderings to get good initial solutions
    def greedy_search(ordering):
        current_set = []
        current_vecs = {zero_vec}
        
        for x in ordering:
            x_vec = exp_vecs[x]
            if is_valid_with_vecs(current_vecs, x_vec):
                current_set.append(x)
                add_to_vecs(current_vecs, x_vec)
        
        return current_set
    
    random.seed(42)
    
    # Try different orderings
    orderings = [
        sorted(candidates),
        sorted(candidates, reverse=True),
        candidates[:],
    ]
    
    for ordering in orderings:
        result = greedy_search(ordering)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Randomized greedy restarts - more iterations since vector ops are faster
    for restart in range(15000):
        if time.time() > deadline - 5:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Backtracking search from best greedy solution
    if time.time() < deadline - 3:
        current_vecs = {zero_vec}
        for elem in best_set:
            add_to_vecs(current_vecs, exp_vecs[elem])
        
        remaining = [x for x in candidates if x not in best_set]
        
        def backtrack(current_set, current_vecs, remaining):
            nonlocal best_set, best_size
            
            max_possible = len(current_set) + len(remaining)
            if max_possible <= best_size:
                return
            
            for i, x in enumerate(remaining):
                x_vec = exp_vecs[x]
                if not is_valid_with_vecs(current_vecs, x_vec):
                    continue
                
                new_vecs = set(current_vecs)
                add_to_vecs(new_vecs, x_vec)
                
                new_set = current_set + [x]
                new_remaining = remaining[:i] + remaining[i+1:]
                
                if len(new_set) > best_size:
                    best_size = len(new_set)
                    best_set = new_set[:]
                
                if len(new_set) + len(new_remaining) > best_size:
                    backtrack(new_set, new_vecs, new_remaining)
        
        backtrack(best_set[:], current_vecs, remaining)
    
    # Local search: try additions and swaps
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            
            # Try adding any candidate
            current_vecs = {zero_vec}
            for elem in best_set:
                add_to_vecs(current_vecs, exp_vecs[elem])
            
            for x in candidates:
                if x in best_set:
                    continue
                if is_valid_with_vecs(current_vecs, exp_vecs[x]):
                    best_set.append(x)
                    best_size += 1
                    improved = True
                    break
            
            # Try swaps
            if not improved:
                for i, old_x in enumerate(best_set):
                    trial_set = [x for j, x in enumerate(best_set) if j != i]
                    
                    trial_vecs = {zero_vec}
                    for elem in trial_set:
                        add_to_vecs(trial_vecs, exp_vecs[elem])
                    
                    for new_x in candidates:
                        if new_x in trial_set:
                            continue
                        if is_valid_with_vecs(trial_vecs, exp_vecs[new_x]):
                            best_set = trial_set + [new_x]
                            improved = True
                            break
                    if improved:
                        break
    
    # More randomized search with time remaining
    start_time = time.time()
    max_time = max(0, deadline - time.time() - 1.0)
    while time.time() - start_time < max_time:
        shuffled = candidates[:]
        random.shuffle(shuffled)
        result = greedy_search(shuffled)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]
    
    # Final verification with actual products
    products = {1}
    for x in best_set:
        new_prods = {p * x for p in products}
        products.update(new_prods)
    
    return [int(x) for x in best_set]
