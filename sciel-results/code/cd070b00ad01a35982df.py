import time
import random
from math import gcd
from functools import reduce

def lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 90s limit

    # Precompute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    weights = {s: L // s for s in range(1, 23)}

    # Use a single large prime for fast modular collision detection
    P = 2**31 - 1  # Mersenne prime
    
    def count_collisions_mod(S):
        """Count number of collision pairs using modular arithmetic (fast)."""
        if not S:
            return 0
        w = [weights[s] % P for s in S]
        sums = {0: 1}
        collisions = 0
        for val in w:
            new_sums = {}
            for s, cnt in sums.items():
                ns = (s + val) % P
                if ns in sums:
                    collisions += cnt * sums[ns]
                if ns in new_sums:
                    collisions += cnt * new_sums[ns]
                else:
                    new_sums[ns] = cnt
            for ns, cnt in new_sums.items():
                if ns in sums:
                    sums[ns] += cnt
                else:
                    sums[ns] = cnt
        return collisions

    def count_collisions_exact(S):
        """Count number of collision pairs using exact arithmetic."""
        if not S:
            return 0
        w = [weights[s] for s in S]
        sums = {0: 1}
        collisions = 0
        for val in w:
            new_sums = {}
            for s, cnt in sums.items():
                ns = s + val
                if ns in sums:
                    collisions += cnt * sums[ns]
                if ns in new_sums:
                    collisions += cnt * new_sums[ns]
                else:
                    new_sums[ns] = cnt
            for ns, cnt in new_sums.items():
                if ns in sums:
                    sums[ns] += cnt
                else:
                    sums[ns] = cnt
        return collisions

    def is_dissociated_fast(S):
        """Quick check using modular arithmetic as a necessary condition."""
        if not S:
            return True
        return count_collisions_mod(S) == 0

    def is_dissociated(S):
        if not S:
            return True
        return count_collisions_exact(S) == 0

    best_set = []
    best_size = 0
    candidate_sets = []

    # Strategy 1: Greedy with random restarts
    for _ in range(150):
        if time.time() > deadline - 25:
            break
        
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        
        current_set = []
        for x in cand_order:
            new_set = current_set + [x]
            if is_dissociated_fast(new_set) and is_dissociated(new_set):
                current_set = new_set
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            candidate_sets = [list(current_set)]
        elif len(current_set) == best_size:
            candidate_sets.append(list(current_set))
            if len(candidate_sets) > 10:
                candidate_sets.pop(0)

    # Strategy 2: Simulated Annealing to minimize collisions
    # Objective: minimize (number_of_collisions, -size)
    # We use a smooth surrogate: collision_count + penalty_for_small_size
    
    def annealing_objective(S):
        """Smooth objective: collisions dominate, then prefer larger sets."""
        if not S:
            return (0, 0)
        c = count_collisions_mod(S)
        size = len(S)
        # Penalize small sets more when collisions are equal
        return (c, -size)

    search_candidates = [list(best_set)] if best_set else []
    for cs in candidate_sets[1:]:
        if cs != best_set:
            search_candidates.append(list(cs))
    
    # Also add some random starting points
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        rand_set = random.sample(range(1, 23), random.randint(6, 10))
        search_candidates.append(rand_set)

    for cand in search_candidates:
        if time.time() > deadline - 3:
            break
        
        current = list(cand)
        if not current:
            continue
        
        current_obj = annealing_objective(current)
        best_in_anneal = list(current)
        best_in_anneal_obj = current_obj
        
        # Simulated annealing parameters
        T_start = 5.0
        T_end = 0.01
        cooling_rate = 0.995
        T = T_start
        max_iter = 5000
        iter_count = 0
        
        while T > T_end and iter_count < max_iter:
            if time.time() > deadline - 3:
                break
            
            # Generate a neighbor: random swap, add, or remove
            missing = [x for x in range(1, 23) if x not in current]
            move_type = random.random()
            
            if move_type < 0.4 and missing:
                # Add an element
                x = random.choice(missing)
                new_set = current + [x]
            elif move_type < 0.7 and current:
                # Remove an element
                x = random.choice(current)
                new_set = [y for y in current if y != x]
            elif move_type < 0.85 and len(current) > 0 and missing:
                # Single swap
                x_remove = random.choice(current)
                x_add = random.choice(missing)
                new_set = [y for y in current if y != x_remove] + [x_add]
            elif move_type < 0.95 and len(current) > 1 and len(missing) >= 1:
                # Double swap: remove 1, add 2
                x_remove = random.choice(current)
                x_adds = random.sample(missing, min(2, len(missing)))
                new_set = [y for y in current if y != x_remove] + x_adds
            else:
                # Triple swap: remove 2, add 3
                if len(current) > 2 and len(missing) >= 3:
                    x_removes = random.sample(current, min(2, len(current)))
                    x_adds = random.sample(missing, min(3, len(missing)))
                    new_set = [y for y in current if y not in x_removes] + x_adds
                else:
                    # Fallback: single add
                    if missing:
                        x = random.choice(missing)
                        new_set = current + [x]
                    else:
                        x = random.choice(current)
                        new_set = [y for y in current if y != x]
            
            if len(new_set) != len(set(new_set)):
                T *= cooling_rate
                iter_count += 1
                continue
            
            new_obj = annealing_objective(new_set)
            
            # Accept if better or with probability exp(-delta/T)
            # Compare tuples: (collisions, -size)
            if new_obj <= current_obj:
                current = new_set
                current_obj = new_obj
                if new_obj < best_in_anneal_obj:
                    best_in_anneal_obj = new_obj
                    best_in_anneal = list(new_set)
            else:
                # Compute delta in a scalar form
                # delta = (new_collisions - cur_collisions) + 0.1*(new_size - cur_size)
                delta = (new_obj[0] - current_obj[0]) + 0.1 * (current_obj[1] - new_obj[1])
                if delta < 0:
                    delta = 0
                # Accept with probability exp(-delta/T)
                if random.random() < min(1.0, 2.718281828 ** (-delta / max(T, 1e-10))):
                    current = new_set
                    current_obj = new_obj
                    if new_obj < best_in_anneal_obj:
                        best_in_anneal_obj = new_obj
                        best_in_anneal = list(new_set)
            
            T *= cooling_rate
            iter_count += 1
        
        # After annealing, try to polish: greedily add elements
        polished = list(best_in_anneal)
        for x in range(1, 23):
            if x not in polished:
                new_set = polished + [x]
                if is_dissociated_fast(new_set) and is_dissociated(new_set):
                    polished = new_set
        
        if len(polished) > best_size:
            best_size = len(polished)
            best_set = list(polished)
            candidate_sets = [list(polished)]
        elif len(polished) == best_size:
            candidate_sets.append(list(polished))
            if len(candidate_sets) > 10:
                candidate_sets.pop(0)

    # Strategy 3: Final local search from best set
    if best_set:
        current = list(best_set)
        stall_count = 0
        max_stall = 100
        
        while time.time() < deadline - 1 and stall_count < max_stall:
            improved = False
            missing = [x for x in range(1, 23) if x not in current]
            
            # Try to add a random missing element
            if missing:
                x = random.choice(missing)
                new_set = current + [x]
                if is_dissociated_fast(new_set) and is_dissociated(new_set):
                    current = new_set
                    improved = True
            
            # Try single swap
            if not improved and len(current) > 1 and missing:
                x_remove = random.choice(current)
                x_add = random.choice(missing)
                new_set = [x for x in current if x != x_remove] + [x_add]
                if is_dissociated_fast(new_set) and is_dissociated(new_set):
                    current = new_set
                    improved = True
            
            # Try double swap
            if not improved and len(current) > 2 and len(missing) >= 2:
                remove_candidates = random.sample(current, min(2, len(current)))
                add_candidates = random.sample(missing, min(2, len(missing)))
                new_set = [x for x in current if x not in remove_candidates] + add_candidates
                if len(new_set) == len(set(new_set)):
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            # Try triple swap
            if not improved and len(current) > 3 and len(missing) >= 3:
                remove_candidates = random.sample(current, min(3, len(current)))
                add_candidates = random.sample(missing, min(3, len(missing)))
                new_set = [x for x in current if x not in remove_candidates] + add_candidates
                if len(new_set) == len(set(new_set)):
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            # Try quadruple swap
            if not improved and len(current) > 4 and len(missing) >= 4:
                remove_candidates = random.sample(current, min(4, len(current)))
                add_candidates = random.sample(missing, min(4, len(missing)))
                new_set = [x for x in current if x not in remove_candidates] + add_candidates
                if len(new_set) == len(set(new_set)):
                    if is_dissociated_fast(new_set) and is_dissociated(new_set):
                        current = new_set
                        improved = True
            
            if improved:
                stall_count = 0
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = list(current)
            else:
                stall_count += 1

    if not best_set:
        return [1]
        
    return best_set
