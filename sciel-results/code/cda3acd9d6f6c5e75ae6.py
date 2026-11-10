import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    timeout = 115  # Safety margin for 150s wall-clock limit
    
    def build_counts(chosen):
        """Build sum-count array from a chosen set."""
        counts = [0] * (max_sum + 1)
        for i, x in enumerate(chosen):
            for y in chosen[i:]:
                counts[x + y] += 1
        return counts
    
    def can_add(chosen, counts, x, chosen_set):
        """Check if x can be added without violating B2[2] constraint."""
        if x in chosen_set:
            return False
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if y != x and counts[x + y] + 1 > 2:
                return False
        return True
    
    def add_element(chosen, counts, x, chosen_set):
        """Add x to chosen set and update counts."""
        for y in chosen:
            counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
        chosen_set.add(x)
    
    def remove_element(chosen, counts, x, chosen_set):
        """Remove x from chosen set and update counts."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
        chosen_set.discard(x)
    
    def greedy_random(rng, order='random'):
        """Run greedy construction with specified ordering."""
        if order == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order == 'ascending':
            candidates = list(range(1, N + 1))
        elif order == 'descending':
            candidates = list(range(N, 0, -1))
        elif order == 'sparse_first':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        elif order == 'prime_first':
            # Primes tend to have fewer additive relations
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [i for i in range(1, N + 1) if is_prime(i)]
            non_primes = [i for i in range(1, N + 1) if not is_prime(i)]
            rng.shuffle(primes)
            rng.shuffle(non_primes)
            candidates = primes + non_primes
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        
        chosen = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x, chosen_set):
                add_element(chosen, counts, x, chosen_set)
                
        return chosen, counts, chosen_set
    
    def local_search(chosen, counts, chosen_set, max_rounds=50, start_time=None, timeout=None):
        """Improved local search with add-then-swap strategy."""
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            
            if timeout and start_time:
                if time.time() - start_time > timeout:
                    break
            
            # Phase 1: Try to add any element (scan all)
            for x in range(1, N + 1):
                if x not in chosen_set and can_add(chosen, counts, x, chosen_set):
                    add_element(chosen, counts, x, chosen_set)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try 1-for-1 swaps
            if len(chosen) > 0:
                # Sample elements to try removing
                sample_size = min(len(chosen), 30)
                sampled = random.sample(chosen, sample_size)
                
                for x in sampled:
                    # Temporarily remove x
                    idx = chosen.index(x)
                    for y in chosen:
                        if y != x:
                            counts[x + y] -= 1
                    counts[2 * x] -= 1
                    chosen.pop(idx)
                    chosen_set.discard(x)
                    
                    # Try to find an element that can now be added
                    found = False
                    for y in range(1, N + 1):
                        if y not in chosen_set and can_add(chosen, counts, y, chosen_set):
                            add_element(chosen, counts, y, chosen_set)
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Put x back
                        for z in chosen:
                            counts[x + z] += 1
                        counts[2 * x] += 1
                        chosen.append(x)
                        chosen_set.add(x)
                    
                    if improved:
                        break
        
        return chosen, counts, chosen_set
    
    def local_search_sa(chosen, counts, chosen_set, start_time=None, timeout=None):
        """Simulated annealing-style local search."""
        if timeout and start_time:
            if time.time() - start_time > timeout:
                return chosen, counts, chosen_set
        
        current_len = len(chosen)
        best_chosen = chosen[:]
        best_counts = counts[:]
        best_chosen_set = chosen_set.copy()
        
        temperature = 50.0
        cooling = 0.95
        
        while time.time() - start_time < timeout:
            if temperature < 1.0:
                break
            
            # Try a random swap: remove one random element, add one random candidate
            if len(chosen) == 0:
                break
            
            x = random.choice(chosen)
            y = random.randint(1, N)
            
            if y in chosen_set or y == x:
                continue
            
            # Remove x
            idx = chosen.index(x)
            for z in chosen:
                if z != x:
                    counts[x + z] -= 1
            counts[2 * x] -= 1
            chosen.pop(idx)
            chosen_set.discard(x)
            
            # Try to add y
            if can_add(chosen, counts, y, chosen_set):
                add_element(chosen, counts, y, chosen_set)
                new_len = len(chosen)
                
                if new_len > current_len:
                    current_len = new_len
                    if new_len > len(best_chosen):
                        best_chosen = chosen[:]
                        best_counts = counts[:]
                        best_chosen_set = chosen_set.copy()
                else:
                    # Accept with probability exp(-delta/T) where delta = current_len - new_len
                    delta = current_len - new_len
                    if delta > 0 and random.random() < np.exp(-delta / temperature):
                        current_len = new_len
                    else:
                        # Revert
                        idx2 = chosen.index(y)
                        for z in chosen:
                            if z != y:
                                counts[y + z] -= 1
                        counts[2 * y] -= 1
                        chosen.pop(idx2)
                        chosen_set.discard(y)
                        
                        # Put x back
                        for z in chosen:
                            counts[x + z] += 1
                        counts[2 * x] += 1
                        chosen.append(x)
                        chosen_set.add(x)
            else:
                # Put x back
                for z in chosen:
                    counts[x + z] += 1
                counts[2 * x] += 1
                chosen.append(x)
                chosen_set.add(x)
            
            temperature *= cooling
        
        # Restore best
        chosen[:] = best_chosen
        chosen_set.clear()
        chosen_set.update(best_chosen)
        counts[:] = best_counts
        
        return chosen, counts, chosen_set
    
    best_set = []
    best_len = 0
    
    # Phase 1: Quick greedy with diverse orderings to find a good starting point
    rng = random.Random(42)
    orders = ['random', 'ascending', 'descending', 'sparse_first', 'prime_first']
    
    for order in orders:
        if time.time() - start_time > timeout - 30:
            break
        chosen, counts, chosen_set = greedy_random(rng, order)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    # Phase 2: Local search on the best greedy solution
    if best_set and time.time() - start_time < timeout - 20:
        chosen = best_set[:]
        chosen_set = set(chosen)
        counts = build_counts(chosen)
        chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_rounds=100, start_time=start_time, timeout=timeout - 15)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    # Phase 3: Simulated annealing polish on the best solution
    if best_set and time.time() - start_time < timeout - 10:
        chosen = best_set[:]
        chosen_set = set(chosen)
        counts = build_counts(chosen)
        chosen, counts, chosen_set = local_search_sa(chosen, counts, chosen_set, start_time=start_time, timeout=timeout - 5)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    # Phase 4: More restarts with local search if time allows
    while time.time() - start_time < timeout - 5:
        order = random.choice(['random', 'sparse_first', 'prime_first'])
        chosen, counts, chosen_set = greedy_random(random.Random(), order)
        
        # Quick local search
        remaining = timeout - (time.time() - start_time) - 3
        if remaining > 2:
            chosen, counts, chosen_set = local_search(chosen, counts, chosen_set, max_rounds=30, start_time=start_time, timeout=timeout - 3)
        
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    return best_set
