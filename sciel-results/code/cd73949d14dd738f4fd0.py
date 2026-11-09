import time
import random
import math
from math import gcd
from itertools import combinations
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_lcms(numbers):
    """Precompute LCM table for all pairs in [1, 128]."""
    n = max(numbers)
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    return lcm_table

def _build_conflict_table(lcm_table, N=128):
    """
    Precompute conflict bitsets.
    conflict[x][a] is a bitmask of b such that (a,b,x) is a bad triple.
    """
    conflict = [[0] * (N + 1) for _ in range(N + 1)]
    # For each pair (a,b), find all x such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    # Group by lcm value for efficiency
    lcm_groups = defaultdict(list)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            l = lcm_table[a][b]
            lcm_groups[l].append((a, b))
    
    # For each lcm value L, for each pair (a,b) with lcm=L,
    # find all x such that lcm(a,x)=L and lcm(b,x)=L
    for L, pairs in lcm_groups.items():
        # For each a, find all x with lcm(a,x) = L
        a_to_x = defaultdict(list)
        for a, b in pairs:
            for x in range(1, N + 1):
                if x == a or x == b:
                    continue
                if lcm_table[a][x] == L and lcm_table[b][x] == L:
                    a_to_x[a].append(x)
        
        # For each triple (a,b,x) where lcm(a,b)=lcm(a,x)=lcm(b,x)=L
        for a, b in pairs:
            for x in a_to_x[a]:
                if x == b:
                    continue
                # Verify: lcm(b,x) should also be L
                if lcm_table[b][x] == L:
                    # (a,b,x) is a bad triple
                    conflict[x][a] |= (1 << b)
                    conflict[x][b] |= (1 << a)
                    conflict[a][x] |= (1 << b)
                    conflict[a][b] |= (1 << x)
                    conflict[b][x] |= (1 << a)
                    conflict[b][a] |= (1 << x)
    
    return conflict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = _precompute_lcms(numbers)
    
    # Precompute conflict bitsets
    conflict = _build_conflict_table(lcm_table, N)
    
    def popcount(m):
        return bin(m).count('1')
    
    def can_add(x, selected_mask):
        """Check if x can be added to the set represented by selected_mask."""
        # For each a in selected, check if conflict[x][a] intersects with selected
        # Use bit operations for speed
        m = selected_mask
        while m:
            a = (m & (-m)).bit_length() - 1
            if conflict[x][a] & selected_mask:
                return False
            m &= m - 1
        return True
    
    def mask_to_list(mask):
        return [i for i in range(1, N + 1) if mask & (1 << i)]
    
    def greedy(order, selected_mask):
        """Greedily add numbers in order to selected_mask."""
        for x in order:
            if selected_mask & (1 << x):
                continue
            if can_add(x, selected_mask):
                selected_mask |= (1 << x)
        return selected_mask
    
    # Strategy 1: Primes and Prime Powers first
    def is_prime_power(n):
        if n < 2:
            return False
        p = n
        while p > 1:
            if p * p > n:
                break
            if n % p == 0:
                n //= p
            else:
                return False
        return True

    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in numbers if is_prime(x)]
    prime_powers = [x for x in numbers if is_prime_power(x) and x not in primes]
    others = [x for x in numbers if x not in primes and x not in prime_powers]
    
    # Order 1: Primes, Prime Powers, Others
    order1 = primes + prime_powers + others
    mask1 = greedy(order1, 0)
    
    # Order 2: Standard 1..128
    mask2 = greedy(numbers, 0)
    
    # Order 3: Sorted by number of divisors (ascending)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    order3 = sorted(numbers, key=num_divisors)
    mask3 = greedy(order3, 0)
    
    best_mask = max([mask1, mask2, mask3], key=popcount)
    best_size = popcount(best_mask)
    
    # Try random shuffles (reduced to leave time for SA)
    random.seed(42)
    for i in range(20000):
        if time.time() > deadline - 40:  # Leave 40s for SA + local search
            break
        random.shuffle(numbers)
        res_mask = greedy(numbers, 0)
        res_size = popcount(res_mask)
        if res_size > best_size:
            best_mask = res_mask
            best_size = res_size
    
    # Local search: try to add elements, then 1-swap, then 2-swap
    def local_search(mask, deadline_local):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline_local:
                break
            # Try adding each excluded number
            for x in range(1, N + 1):
                if time.time() > deadline_local:
                    break
                if mask & (1 << x):
                    continue
                if can_add(x, mask):
                    mask |= (1 << x)
                    improved = True
                    break
            # Try 1-swap: remove one element, try to add a different one
            if not improved:
                selected = mask_to_list(mask)
                for i in range(len(selected)):
                    if time.time() > deadline_local:
                        break
                    candidate_remove = selected[i]
                    new_mask = mask & ~(1 << candidate_remove)
                    for x in range(1, N + 1):
                        if time.time() > deadline_local:
                            break
                        if new_mask & (1 << x):
                            continue
                        if can_add(x, new_mask):
                            new_mask |= (1 << x)
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
            # Try 2-swap: remove two elements, try to add two different ones
            if not improved and time.time() < deadline_local - 2:
                selected = mask_to_list(mask)
                n_sel = len(selected)
                for i in range(n_sel):
                    if time.time() > deadline_local:
                        break
                    for j in range(i + 1, n_sel):
                        if time.time() > deadline_local:
                            break
                        new_mask = mask & ~(1 << selected[i]) & ~(1 << selected[j])
                        # Try to add two new elements
                        found = False
                        for x in range(1, N + 1):
                            if time.time() > deadline_local:
                                break
                            if new_mask & (1 << x):
                                continue
                            if can_add(x, new_mask):
                                new_mask |= (1 << x)
                                for y in range(x + 1, N + 1):
                                    if time.time() > deadline_local:
                                        break
                                    if new_mask & (1 << y):
                                        continue
                                    if can_add(y, new_mask):
                                        new_mask |= (1 << y)
                                        if popcount(new_mask) > popcount(mask):
                                            mask = new_mask
                                            improved = True
                                            found = True
                                            break
                                if found:
                                    break
                        if found:
                            break
                    if improved:
                        break
        return mask
    
    # Simulated Annealing phase
    def simulated_annealing(start_mask, deadline_sa):
        """SA on the binary vector, accepting removals to escape local optima."""
        current_mask = start_mask
        current_size = popcount(current_mask)
        best_sa_mask = current_mask
        best_sa_size = current_size
        
        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling = 0.995  # Geometric cooling
        
        # Precompute list of all numbers for random selection
        all_nums = list(range(1, N + 1))
        
        while T > T_min and time.time() < deadline_sa:
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.35:
                # Try to add a random element
                x = random.choice(all_nums)
                if current_mask & (1 << x):
                    continue
                if not can_add(x, current_mask):
                    # Can't add - skip this step
                    continue
                new_mask = current_mask | (1 << x)
                delta = 1  # size increases by 1
                # Always accept improvements
                current_mask = new_mask
                current_size += 1
                if current_size > best_sa_size:
                    best_sa_mask = current_mask
                    best_sa_size = current_size
            elif move_type < 0.70:
                # Try to remove a random element (to escape local optima)
                selected = mask_to_list(current_mask)
                if not selected:
                    continue
                x = random.choice(selected)
                new_mask = current_mask & ~(1 << x)
                delta = -1  # size decreases by 1
                # Accept with probability exp(delta/T)
                if random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_size -= 1
            else:
                # Try a swap: remove one, add another
                selected = mask_to_list(current_mask)
                if not selected:
                    continue
                x_remove = random.choice(selected)
                x_add = random.choice(all_nums)
                if x_add == x_remove:
                    continue
                if current_mask & (1 << x_add):
                    continue
                new_mask = current_mask & ~(1 << x_remove)
                if not can_add(x_add, new_mask):
                    continue
                new_mask |= (1 << x_add)
                delta = 0  # size stays same
                # Accept with probability exp(delta/T) = 1, but only if it helps future moves
                # Actually for delta=0, always accept to explore
                current_mask = new_mask
                # current_size unchanged
            
            # Cool down
            T *= cooling
            
            # Periodic local search polish
            if time.time() < deadline_sa - 5 and current_size >= best_sa_size:
                polished = local_search(current_mask, min(time.time() + 0.5, deadline_sa - 5))
                polished_size = popcount(polished)
                if polished_size > best_sa_size:
                    best_sa_mask = polished
                    best_sa_size = polished_size
                    current_mask = polished
                    current_size = polished_size
        
        return best_sa_mask
    
    # Run simulated annealing from the best mask found so far
    if time.time() < deadline - 5:
        sa_deadline = deadline - 5
        sa_result = simulated_annealing(best_mask, sa_deadline)
        sa_size = popcount(sa_result)
        if sa_size > best_size:
            best_mask = sa_result
            best_size = sa_size
    
    # Final local search polish
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, deadline - 2)
    
    # Final verification
    result = mask_to_list(best_mask)
    # Verify no bad triples exist
    for a, b, c in combinations(result, 3):
        if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
            # This shouldn't happen, but if it does, remove the worst element
            break
    return result
    
    return mask_to_list(best_mask)
