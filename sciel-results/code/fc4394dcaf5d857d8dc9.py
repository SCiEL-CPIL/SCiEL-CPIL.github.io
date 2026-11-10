import time
import random
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    best_set = []
    best_len = 0
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    def can_add(chosen, counts, x):
        """Check if x can be added without violating B2[2] constraint."""
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen:
            if y != x and counts[x + y] + 1 > 2:
                return False
        return True
    
    def add_element(chosen, counts, x):
        """Add x to chosen set and update counts."""
        for y in chosen:
            if y != x:
                counts[x + y] += 1
        counts[2 * x] += 1
        chosen.append(x)
    
    def remove_element(chosen, counts, x):
        """Remove x from chosen set and update counts."""
        for y in chosen:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen.remove(x)
    
    def greedy_with_order(order_type, rng):
        """Run greedy construction with specified ordering."""
        if order_type == 'random':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        elif order_type == 'ascending':
            candidates = list(range(1, N + 1))
        elif order_type == 'descending':
            candidates = list(range(N, 0, -1))
        elif order_type == 'sparse_first':
            odds = list(range(1, N + 1, 2))
            evens = list(range(2, N + 1, 2))
            rng.shuffle(odds)
            rng.shuffle(evens)
            candidates = odds + evens
        elif order_type == 'primes_first':
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            composites = [x for x in range(1, N + 1) if not is_prime(x)]
            rng.shuffle(primes)
            rng.shuffle(composites)
            candidates = primes + composites
        elif order_type == 'powers_first':
            powers = [2**k for k in range(10) if 2**k <= N]
            rest = [x for x in range(1, N + 1) if x not in set(powers)]
            rng.shuffle(powers)
            rng.shuffle(rest)
            candidates = powers + rest
        elif order_type == 'weighted':
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
        
        chosen = []
        counts = [0] * (max_sum + 1)
        
        for x in candidates:
            if can_add(chosen, counts, x):
                add_element(chosen, counts, x)
                
        return chosen, counts
    
    def local_search(chosen, counts, max_rounds=50, time_limit=None, start_time=None):
        """Improved local search with insertion and swap strategies."""
        chosen_set = set(chosen)
        improved = True
        rounds = 0
        
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            
            if time_limit and start_time:
                if time.time() - start_time > time_limit:
                    break
            
            # Try to add any element
            for x in range(1, N + 1):
                if x not in chosen_set and can_add(chosen, counts, x):
                    add_element(chosen, counts, x)
                    chosen_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one element, try to add another
            if len(chosen) > 0:
                # Sample elements to try removing
                sample_size = min(len(chosen), 100)
                sampled = random.sample(chosen, sample_size)
                
                for x in sampled:
                    # Temporarily remove x
                    remove_element(chosen, counts, x)
                    chosen_set.discard(x)
                    
                    # Try to find an element that can now be added
                    found = False
                    for y in range(1, N + 1):
                        if y not in chosen_set and can_add(chosen, counts, y):
                            add_element(chosen, counts, y)
                            chosen_set.add(y)
                            improved = True
                            found = True
                            break
                    
                    if not found:
                        # Put x back
                        add_element(chosen, counts, x)
                        chosen_set.add(x)
                    
                    if improved:
                        break
        
        return chosen, counts
    
    # Phase 1: Quick greedy with diverse orderings
    rng = random.Random(42)
    orderings = [
        'ascending',
        'descending',
        'random',
        'sparse_first',
        'primes_first',
        'powers_first',
        'weighted',
        'random',
        'random',
        'sparse_first',
    ]
    
    for order_type in orderings:
        if time.time() > deadline:
            break
        current_set, current_counts = greedy_with_order(order_type, rng)
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    # Phase 2: Local search on best found
    if best_set and time.time() < deadline - 10:
        chosen = best_set[:]
        counts = [0] * (max_sum + 1)
        for x in chosen:
            for y in chosen:
                if y >= x:
                    counts[x + y] += 1
        chosen, counts = local_search(chosen, counts, max_rounds=100, time_limit=deadline - time.time(), start_time=start_time)
        if len(chosen) > best_len:
            best_len = len(chosen)
            best_set = chosen[:]
    
    # Phase 3: More restarts with local search if time allows
    while time.time() < deadline - 5:
        order_type = random.choice(['random', 'sparse_first', 'primes_first', 'powers_first', 'weighted'])
        current_set, current_counts = greedy_with_order(order_type, rng)
        
        # Quick local search
        remaining_time = deadline - time.time()
        if remaining_time > 2:
            current_set, current_counts = local_search(
                current_set, current_counts, 
                max_rounds=min(30, int(remaining_time * 5)),
                time_limit=remaining_time - 1,
                start_time=start_time
            )
        
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_set = current_set[:]
    
    return best_set
