"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 128
    numbers = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l
            
    # Precompute bad triples
    # conflict_pairs[x] = set of (a,b) such that {a,b,x} is bad
    # Then can_add(x, S) is: check if any pair in S is in conflict_pairs[x]
    
    conflict_pairs = [set() for _ in range(n + 1)]
    
    # For each pair (a,b), find all z such that {a,b,z} is a bad triple
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for z in range(1, n + 1):
                if z == a or z == b:
                    continue
                if lcm_table[a][z] == L_ab and lcm_table[b][z] == L_ab:
                    p1 = (min(a,b), max(a,b))
                    p2 = (min(a,z), max(a,z))
                    p3 = (min(b,z), max(b,z))
                    conflict_pairs[z].add(p1)
                    conflict_pairs[b].add(p2)
                    conflict_pairs[a].add(p3)
    
    # Precompute for each element x, the list of other elements that conflict with it
    # conflict_with[x] = set of y such that there exists z where {x,y,z} is bad
    # Actually, we need: for a given set S, can we add x? 
    # x conflicts with S if there exist a,b in S such that (a,b) in conflict_pairs[x]
    
    # For faster can_add, precompute for each x a list of "witness pairs"
    # But the current approach is fine. Let's optimize can_add.
    
    def can_add_fast(x, current_set):
        """Check if x can be added to current_set without forming a bad triple."""
        # For each pair (a,b) in conflict_pairs[x], check if both a and b are in current_set
        for (a, b) in conflict_pairs[x]:
            if a in current_set and b in current_set:
                return False
        return True

    def greedy_add(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add_fast(x, selected):
                selected.add(x)
        return selected

    def local_search(initial_set, max_iters=500):
        """Local search with swaps and kicks."""
        current_set = set(initial_set)
        best_set = set(current_set)
        improved = True
        iter_count = 0
        
        while improved and time.time() < deadline - 5 and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # 1. Try to add an element
            added = False
            for in_elem in range(1, n + 1):
                if in_elem in current_set:
                    continue
                if can_add_fast(in_elem, current_set):
                    current_set.add(in_elem)
                    improved = True
                    added = True
                    break
            
            if added:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                continue
            
            # 2. Try swap: remove one, add another
            swapped = False
            for out_elem in list(current_set):
                if time.time() > deadline - 5:
                    break
                trial_base = current_set - {out_elem}
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add_fast(in_elem, trial_base):
                        current_set = trial_base | {in_elem}
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                continue

            # 3. Kick: remove 2 random elements, try to add 2 new ones
            if len(current_set) >= 2:
                removed = random.sample(list(current_set), 2)
                trial_base = current_set - set(removed)
                added_any = False
                for in_elem in range(1, n + 1):
                    if in_elem in current_set:
                        continue
                    if can_add_fast(in_elem, trial_base):
                        trial_base.add(in_elem)
                        added_any = True
                        break
                if added_any:
                    for in_elem in range(1, n + 1):
                        if in_elem in trial_base:
                            continue
                        if can_add_fast(in_elem, trial_base):
                            trial_base.add(in_elem)
                            break
                    if len(trial_base) >= len(current_set):
                        current_set = trial_base
                        improved = True
                        if len(current_set) > len(best_set):
                            best_set = set(current_set)

        return best_set

    def simulated_annealing(initial_set, max_iters=1000):
        """Simulated annealing to escape local optima."""
        current_set = set(initial_set)
        best_set = set(current_set)
        best_size = len(current_set)
        
        # Start temperature high, decay
        temp_start = 10.0
        temp_end = 0.1
        decay = (temp_end / temp_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            temp = temp_start * (decay ** it)
            
            # Choose a random move
            move_type = random.choice(['add', 'remove', 'swap', 'kick2'])
            
            if move_type == 'add':
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if not candidates:
                    continue
                in_elem = random.choice(candidates)
                trial_set = set(current_set) | {in_elem}
                delta = len(trial_set) - len(current_set)
                # Check validity
                if not can_add_fast(in_elem, current_set):
                    # Invalid move, accept with probability exp(-1/temp) if temp > 0
                    if random.random() < 0.01:  # Small probability of accepting invalid? No, must be valid
                        continue
                    continue
                current_set = trial_set
                
            elif move_type == 'remove':
                if not current_set:
                    continue
                out_elem = random.choice(list(current_set))
                trial_set = current_set - {out_elem}
                delta = len(trial_set) - len(current_set)
                # Always accept removals in SA for diversity? Or use energy
                # Energy = -size + penalty for conflicts (but we maintain validity)
                # Let's just accept if it doesn't reduce size too much
                if random.random() < 0.3:  # 30% chance to accept a size-decreasing move
                    current_set = trial_set
                    
            elif move_type == 'swap':
                if not current_set:
                    continue
                out_elem = random.choice(list(current_set))
                candidates = [x for x in range(1, n + 1) if x not in current_set]
                if not candidates:
                    continue
                in_elem = random.choice(candidates)
                trial_base = current_set - {out_elem}
                if can_add_fast(in_elem, trial_base):
                    current_set = trial_base | {in_elem}
                    
            elif move_type == 'kick2':
                if len(current_set) < 2:
                    continue
                removed = random.sample(list(current_set), 2)
                trial_base = current_set - set(removed)
                # Try to add back 2 elements
                candidates = [x for x in range(1, n + 1) if x not in trial_base]
                random.shuffle(candidates)
                added_count = 0
                for in_elem in candidates:
                    if added_count >= 2:
                        break
                    if can_add_fast(in_elem, trial_base):
                        trial_base.add(in_elem)
                        added_count += 1
                if len(trial_base) >= len(current_set):
                    current_set = trial_base
            
            # Track best
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        
        return best_set

    random.seed(42)
    best_set = set()
    
    # Candidate 1: Primes + 1
    primes = [x for x in range(2, n + 1) if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    candidate1 = set(primes + [1])
    if len(candidate1) > len(best_set):
        best_set = candidate1

    # Candidate 2: Range [65, 128]
    candidate2 = set(range(65, n + 1))
    valid = True
    lst2 = sorted(candidate2)
    for i in range(len(lst2)):
        if not valid: break
        a = lst2[i]
        for j in range(i + 1, len(lst2)):
            if not valid: break
            b = lst2[j]
            for k in range(j + 1, len(lst2)):
                c = lst2[k]
                if (a, b) in conflict_pairs[c]:
                    valid = False
                    break
    if valid and len(candidate2) > len(best_set):
        best_set = candidate2

    # Candidate 3: Multiple greedy strategies
    num_restarts = 40
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if i % 4 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif i % 4 == 1:
            conflict_counts = [len(conflict_pairs[x]) for x in range(n + 1)]
            order = sorted(range(1, n + 1), key=lambda x: conflict_counts[x])
        elif i % 4 == 2:
            # Descending order
            order = list(range(n, 0, -1))
        else:
            # Ascending order
            order = list(range(1, n + 1))
        
        s = greedy_add(order)
        s_improved = local_search(s, max_iters=80)
        if len(s_improved) > len(best_set):
            best_set = s_improved

    # Candidate 4: Simulated annealing on best set
    if time.time() < deadline - 15 and best_set:
        sa_result = simulated_annealing(best_set, max_iters=200)
        if len(sa_result) > len(best_set):
            best_set = sa_result
            # Local search after SA
            if time.time() < deadline - 5:
                best_set = local_search(best_set, max_iters=100)

    # Final local search on the best set found
    if time.time() < deadline - 5 and best_set:
        best_set = local_search(best_set, max_iters=150)
        
    # Verify and return
    if not best_set:
        best_set = {1}
        
    return sorted(best_set)
# EVOLVE-BLOCK-END
