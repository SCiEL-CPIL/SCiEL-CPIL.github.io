import time
import random
import math
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcd matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Precompute conflict pairs: for each x, store set of (a,b) pairs that form bad triple with x
    conflict_pairs = {i: set() for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == gab and gcd_mat[b][c] == gab:
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
    
    # Convert to lists for faster iteration
    conflict_lists = {i: list(conflict_pairs[i]) for i in range(1, N + 1)}
    
    # Precompute conflict masks using Python integers as bitsets
    # For each x, conflict_mask[x] is a set of integers where each integer encodes a pair (a,b)
    # Actually, let's use a different approach: for each x, precompute a dict mapping a -> set of b's
    conflict_dict = {i: {} for i in range(1, N + 1)}
    for i in range(1, N + 1):
        for (a, b) in conflict_pairs[i]:
            if a not in conflict_dict[i]:
                conflict_dict[i][a] = set()
            conflict_dict[i][a].add(b)
            if b not in conflict_dict[i]:
                conflict_dict[i][b] = set()
            conflict_dict[i][b].add(a)
    
    # Fast validity check: can we add x to set S (represented as boolean array)?
    def can_add(x, S_bool):
        for a, b_set in conflict_dict[x].items():
            if S_bool[a]:
                for b in b_set:
                    if S_bool[b]:
                        return False
        return True
    
    # Simulated Annealing
    def simulated_annealing(S_init, t_start=1.0, t_end=0.01, cooling=0.9995, max_iters=50000):
        S = set(S_init)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        
        T = t_start
        rng = random.Random()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            if T < t_end:
                break
            
            # Choose random move: add, remove, or swap
            move_type = rng.random()
            
            if move_type < 0.3:
                # Try to add a random element not in S
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                x = rng.choice(not_in_S)
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
            
            elif move_type < 0.6:
                # Try to remove a random element from S
                if not S:
                    continue
                u = rng.choice(list(S))
                S.remove(u)
                S_bool[u] = False
                current_size -= 1
            
            else:
                # Swap: remove u from S, add v not in S
                if not S:
                    continue
                not_in_S = [x for x in numbers if not S_bool[x]]
                if not not_in_S:
                    continue
                u = rng.choice(list(S))
                v = rng.choice(not_in_S)
                
                # Check if v can be added after removing u
                S_bool[u] = False
                S.remove(u)
                
                if can_add(v, S_bool):
                    S.add(v)
                    S_bool[v] = True
                else:
                    S.add(u)
                    S_bool[u] = True
                
                if current_size > best_size:
                    best_size = current_size
                    best_S = S.copy()
            
            # Cooling
            T *= cooling
        
        return best_S, best_size
    
    # Greedy with different orderings
    def greedy_construct(ordering='random'):
        S = set()
        S_bool = [False] * (N + 1)
        
        if ordering == 'random':
            candidates = numbers[:]
            random.shuffle(candidates)
        elif ordering == 'ascending':
            candidates = numbers[:]
        elif ordering == 'descending':
            candidates = list(reversed(numbers))
        elif ordering == 'prime_first':
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in numbers if is_prime(x)]
            composites = [x for x in numbers if not is_prime(x)]
            random.shuffle(primes)
            random.shuffle(composites)
            candidates = primes + composites
        elif ordering == 'large_first':
            candidates = list(reversed(numbers))
            random.shuffle(candidates)
        elif ordering == 'few_divisors_first':
            # Numbers with fewer divisors are less constrained
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            candidates = sorted(numbers, key=num_divisors)
            random.shuffle(candidates)
        elif ordering == 'many_divisors_first':
            def num_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 1
                        if i != n // i:
                            count += 1
                return count
            candidates = sorted(numbers, key=num_divisors, reverse=True)
            random.shuffle(candidates)
        else:
            candidates = numbers[:]
            random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_bool):
                S.add(x)
                S_bool[x] = True
        
        return S
    
    # Local search with swap-then-verify
    def local_search(S, max_iters=20000):
        S = set(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add any element
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    added = True
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    break
            
            if not added:
                no_improve += 1
                
                # Try swaps
                if S and not_in_S:
                    improved = False
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        u = random.choice(list(S))
                        v = random.choice(not_in_S)
                        
                        S_bool[u] = False
                        S.remove(u)
                        
                        if can_add(v, S_bool):
                            S.add(v)
                            S_bool[v] = True
                            improved = True
                        else:
                            S.add(u)
                            S_bool[u] = True
                    
                    if not improved:
                        # Try double swap: remove 2, add 2
                        if len(S) >= 2 and len(not_in_S) >= 2:
                            for _ in range(20):
                                if time.time() > deadline:
                                    break
                                u1, u2 = random.sample(list(S), 2)
                                v1, v2 = random.sample(not_in_S, 2)
                                
                                S_bool[u1] = False
                                S_bool[u2] = False
                                S.remove(u1)
                                S.remove(u2)
                                
                                if can_add(v1, S_bool):
                                    S.add(v1)
                                    S_bool[v1] = True
                                    if can_add(v2, S_bool):
                                        S.add(v2)
                                        S_bool[v2] = True
                                        improved = True
                                    else:
                                        S.remove(v1)
                                        S_bool[v1] = False
                                S.add(u1)
                                S.add(u2)
                                S_bool[u1] = True
                                S_bool[u2] = True
                                
                                if improved:
                                    break
                        
                        if not improved:
                            if no_improve > 100:
                                break
        
        return best_S, best_size
    
    # Blocking analysis: find elements that block the most additions
    def find_blocking_elements(S, S_bool):
        not_in_S = [x for x in numbers if not S_bool[x]]
        if not not_in_S or not S:
            return {}
        
        block_count = {u: 0 for u in S}
        
        for x in not_in_S:
            # Find which elements in S are blocking x
            for a, b_set in conflict_dict[x].items():
                if S_bool[a]:
                    for b in b_set:
                        if S_bool[b]:
                            block_count[a] = block_count.get(a, 0) + 1
                            block_count[b] = block_count.get(b, 0) + 1
        
        return block_count
    
    # Targeted local search using blocking analysis
    def targeted_local_search(S, max_iters=15000):
        S = set(S)
        S_bool = [False] * (N + 1)
        for x in S:
            S_bool[x] = True
        
        current_size = len(S)
        best_S = S.copy()
        best_size = current_size
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add any element
            added = False
            not_in_S = [x for x in numbers if not S_bool[x]]
            random.shuffle(not_in_S)
            
            for x in not_in_S:
                if can_add(x, S_bool):
                    S.add(x)
                    S_bool[x] = True
                    current_size += 1
                    added = True
                    no_improve = 0
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                    break
            
            if not added:
                no_improve += 1
                
                if no_improve > 20:
                    # Use blocking analysis to guide removals
                    block_counts = find_blocking_elements(S, S_bool)
                    if block_counts:
                        # Remove the most blocking element
                        worst = max(block_counts, key=lambda k: block_counts[k])
                        S.remove(worst)
                        S_bool[worst] = False
                        current_size -= 1
                        no_improve = 0  # Reset since we made a change
                        continue
                
                # Try targeted swaps: remove a blocking element, try to add multiple
                if S and not_in_S:
                    block_counts = find_blocking_elements(S, S_bool)
                    if block_counts:
                        sorted_blocking = sorted(block_counts.items(), key=lambda x: -x[1])[:5]
                        for u, _ in sorted_blocking:
                            if time.time() > deadline:
                                break
                            S_bool[u] = False
                            S.remove(u)
                            
                            # Try to add multiple elements
                            added_any = False
                            for x in not_in_S[:]:
                                if can_add(x, S_bool):
                                    S.add(x)
                                    S_bool[x] = True
                                    current_size += 1
                                    added_any = True
                            
                            if not added_any:
                                S.add(u)
                                S_bool[u] = True
                            
                            if current_size > best_size:
                                best_size = current_size
                                best_S = S.copy()
                                no_improve = 0
                            break
        
        return best_S, best_size
    
    best_S = set()
    best_size = 0
    
    # Phase 1: Multiple greedy starts with different orderings
    orderings = ['random', 'ascending', 'descending', 'prime_first', 'large_first', 'few_divisors_first', 'many_divisors_first']
    for ordering in orderings:
        if time.time() > deadline:
            break
        S = greedy_construct(ordering)
        S, size = local_search(S, max_iters=8000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Phase 2: Simulated annealing from best solution
    if best_S:
        S, size = simulated_annealing(best_S, max_iters=25000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Phase 3: Targeted local search
    if best_S and time.time() < deadline - 5:
        S, size = targeted_local_search(best_S, max_iters=15000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    # Phase 4: More SA restarts with targeted polish
    while time.time() < deadline - 10:
        S = greedy_construct('random')
        S, size = simulated_annealing(S, t_start=0.5, t_end=0.01, cooling=0.999, max_iters=15000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
        
        # Quick targeted polish
        if best_S and time.time() < deadline - 5:
            S, size = targeted_local_search(best_S, max_iters=5000)
            if size > best_size:
                best_size = size
                best_S = S.copy()
    
    # Final polish
    if best_S and time.time() < deadline - 2:
        S, size = local_search(best_S, max_iters=10000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
        
        S, size = targeted_local_search(best_S, max_iters=5000)
        if size > best_size:
            best_size = size
            best_S = S.copy()
    
    return list(best_S)

# EVOLVE-BLOCK-END
