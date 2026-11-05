import time
import random
from math import gcd
from collections import defaultdict

def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are equal.
    """
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    
    # Precompute GCD table for speed
    gcd_table = [[0] * 129 for _ in range(129)]
    for i in range(1, 129):
        for j in range(i, 129):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute prime factors count to prioritize "dense" numbers
    # Numbers with many small factors are less likely to be coprime to others
    def count_prime_factors(n):
        count = 0
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
            p += 1
        if temp > 1:
            count += 1
        return count

    prime_factor_counts = [0] * 129
    for i in range(1, 129):
        prime_factor_counts[i] = count_prime_factors(i)

    def is_valid_addition(x, selected):
        """
        Check if x can be added to selected without creating a bad triple.
        A bad triple (a, b, c) requires gcd(a,b) == gcd(a,c) == gcd(b,c).
        If we add x, we need to ensure no pair (a, b) in selected
        satisfies gcd(x,a) == gcd(x,b) == gcd(a,b).
        """
        sel_list = list(selected)
        n = len(sel_list)
        for i in range(n):
            a = sel_list[i]
            g_xa = gcd_table[x][a]
            for j in range(i + 1, n):
                b = sel_list[j]
                if g_xa == gcd_table[x][b] == gcd_table[a][b]:
                    return False
        return True

    def greedy_build(order):
        """Build a valid set by adding elements in the given order."""
        selected = set()
        for x in order:
            if is_valid_addition(x, selected):
                selected.add(x)
        return selected

    def local_search(selected, max_iters=200):
        """
        Improve the set using add, remove, and swap moves.
        """
        selected = set(selected)
        improved = True
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            
            # 1. Try to add any excluded element
            excluded = [x for x in range(1, 129) if x not in selected]
            for x in excluded:
                if is_valid_addition(x, selected):
                    selected.add(x)
                    improved = True
                    break
            
            # 2. If no add, try 1-for-1 swap (remove one, add one)
            #    or 1-for-2 swap (remove one, add two)
            if not improved:
                # Try removing a "bad" element to allow others in
                # Heuristic: remove element that is part of many "would-be" bad triples
                # If we remove x, can we add y?
                # To save time, we try random swaps first, then systematic if needed.
                
                # Systematic 1-for-1 swap
                found_swap = False
                for x in list(selected):
                    temp = selected - {x}
                    for y in excluded:
                        if is_valid_addition(y, temp):
                            # Check if this swap is beneficial? 
                            # Size stays same. We might want to prefer adding 'denser' numbers
                            # or just take it to escape local minima.
                            # Let's take it if it allows further additions later.
                            # For now, just swap to explore.
                            selected = temp
                            selected.add(y)
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
                
                # If no 1-for-1 swap found, try 1-for-2 (remove 1, add 2)
                if not found_swap:
                    for x in list(selected):
                        temp = selected - {x}
                        added = []
                        for y in excluded:
                            if y == x:
                                continue
                            if is_valid_addition(y, temp | set(added)):
                                added.append(y)
                        if len(added) >= 2:
                            selected = temp | set(added)
                            improved = True
                            break

        return selected

    # Strategy: 
    # 1. Greedy initialization based on "density" (more prime factors = better)
    # 2. Local search to polish
    # 3. Randomized restarts to find global optimum
    
    best_set = set()
    best_size = 0
    
    # Initial greedy: Sort by number of prime factors descending
    # Numbers like 60 (2,3,5) are good. Primes are bad.
    order_dense = sorted(range(1, 129), key=lambda x: prime_factor_counts[x], reverse=True)
    s_init = greedy_build(order_dense)
    s_init = local_search(s_init, max_iters=100)
    best_size = len(s_init)
    best_set = set(s_init)
    
    # Randomized restarts
    random.seed(42)
    start_time = time.time()
    
    # We can also try starting from the dense set and removing random elements to "shake" it
    for trial in range(1000):
        if time.time() > deadline:
            break
            
        if random.random() < 0.3:
            # Perturb best set
            current = set(best_set)
            num_remove = random.randint(1, 3)
            to_remove = random.sample(list(current), min(num_remove, len(current)))
            for x in to_remove:
                current.remove(x)
        else:
            # Random order greedy
            order = list(range(1, 129))
            random.shuffle(order)
            # Bias towards dense numbers slightly?
            # Let's just shuffle
            current = greedy_build(order)
            
        # Local search
        s_new = local_search(current, max_iters=50)
        
        if len(s_new) > best_size:
            best_size = len(s_new)
            best_set = set(s_new)
            # If we found a new best, we can try to polish it more
            s_polished = local_search(best_set, max_iters=100)
            if len(s_polished) > best_size:
                best_size = len(s_polished)
                best_set = set(s_polished)

    # Final polish on the best set found
    s_final = local_search(best_set, max_iters=200)
    if len(s_final) > best_size:
        best_set = s_final
        best_size = len(s_final)

    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
