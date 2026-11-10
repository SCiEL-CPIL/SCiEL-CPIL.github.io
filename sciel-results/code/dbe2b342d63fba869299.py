"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # 3. Build conflict structure for fast validity checks
    # For each element x, store the set of pairs (u,v) such that {x,u,v} is a bad triple
    conflict = {i: set() for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))

    # Helper to check if adding x to S creates any new violation
    def creates_violation(S, x):
        for u, v in conflict[x]:
            if u in S and v in S:
                return True
        return False

    # Helper to calculate the change in violation count if we add x to S
    def delta_violations_add(S, x):
        count = 0
        for u, v in conflict[x]:
            if u in S and v in S:
                count += 1
        return count

    # Helper to calculate the change in violation count if we remove x from S
    def delta_violations_remove(S, x):
        count = 0
        for u, v in conflict[x]:
            if u in S and v in S:
                count += 1
        return count

    def get_violations_count(S):
        """Count bad triples fully contained in S."""
        count = 0
        for a, b, c in bad_triples:
            if a in S and b in S and c in S:
                count += 1
        return count

    def energy(S):
        return -len(S) + 10.0 * get_violations_count(S)

    # 4. Initial Heuristic: Greedy Randomized
    # Try a few different random greedy starts
    best_set = set()
    best_size = 0
    best_energy = float('inf')

    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(numbers)
        S = set()
        for x in numbers:
            if not creates_violation(S, x):
                S.add(x)
        
        e = energy(S)
        if e < best_energy:
            best_energy = e
            best_set = set(S)
            best_size = len(S)

    # 5. Simulated Annealing
    # We will run SA from the best greedy solution found.
    # Move types:
    # 1. Add a random element x not in S.
    # 2. Remove a random element x in S.
    # 3. Swap: Remove x in S, Add y not in S.
    
    # To make SA effective, we need to be able to evaluate the energy change quickly.
    # We maintain current_violations and update it via deltas.
    
    def sa_run(initial_S, steps=50000):
        S = set(initial_S)
        current_violations = get_violations_count(S)
        current_energy = -len(S) + 10.0 * current_violations
        
        local_best_S = set(S)
        local_best_energy = current_energy
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.9995
        
        for step in range(steps):
            if time.time() > deadline - 2:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [x for x in numbers if x not in S]
                if not candidates:
                    continue
                x = random.choice(candidates)
                
                # Calculate delta
                dv = delta_violations_add(S, x)
                # Energy change: -1 (size +1) + 10 * dv
                delta_E = -1.0 + 10.0 * dv
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.add(x)
                    current_violations += dv
                    current_energy += delta_E
            elif move_type < 0.8:
                # Remove a random element
                if not S:
                    continue
                x = random.choice(list(S))
                
                # Calculate delta
                dv = delta_violations_remove(S, x)
                # Energy change: +1 (size -1) - 10 * dv
                delta_E = 1.0 - 10.0 * dv
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.remove(x)
                    current_violations -= dv
                    current_energy += delta_E
            else:
                # Swap
                if not S:
                    continue
                x = random.choice(list(S))
                candidates = [y for y in numbers if y not in S]
                if not candidates:
                    continue
                y = random.choice(candidates)
                
                # Remove x, Add y
                # Delta for removing x
                dv_rem = delta_violations_remove(S, x)
                # Delta for adding y to (S - {x})
                S_temp = S - {x}
                dv_add = delta_violations_add(S_temp, y)
                
                # Total delta in violations: dv_add - dv_rem
                # Total delta in size: 0
                delta_E = 10.0 * (dv_add - dv_rem)
                
                if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                    S.remove(x)
                    S.add(y)
                    current_violations += dv_add - dv_rem
            
            # Check if we found a better valid solution
            if current_violations == 0:
                if len(S) > len(local_best_S):
                    local_best_S = set(S)
                    local_best_energy = current_energy
            
            # Cool down
            T = max(T * cooling_rate, T_min)
            
            # If we've reached T_min, restart with a higher T? Or just continue?
            # For now, just continue.
            
        return local_best_S

    # Run SA
    if time.time() < deadline - 5:
        # Run SA from the best greedy solution
        best_set = sa_run(best_set)
        best_size = len(best_set)
        
        # If we have time, run SA a few more times from slightly perturbed versions of the best set
        if time.time() < deadline - 10:
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Perturb best_set
                S_perturbed = set(best_set)
                k = random.randint(2, 5)
                to_remove = random.sample(list(S_perturbed), min(k, len(S_perturbed)))
                for x in to_remove:
                    S_perturbed.remove(x)
                
                # Fill greedily
                order_rand = list(range(1, n + 1))
                random.shuffle(order_rand)
                for y in order_rand:
                    if y in S_perturbed:
                        continue
                    if not creates_violation(S_perturbed, y):
                        S_perturbed.add(y)
                
                # Run SA from this perturbed state
                S_sa = sa_run(S_perturbed, steps=10000)
                if len(S_sa) > len(best_set):
                    best_set = S_sa
                    best_size = len(best_set)

    # Final validation and cleanup
    # Ensure best_set is valid
    if get_violations_count(best_set) > 0:
        # If invalid, fall back to a greedy construction
        S_greedy = set()
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if not creates_violation(S_greedy, x):
                S_greedy.add(x)
        best_set = S_greedy

    return sorted(list(best_set))

# EVOLVE-BLOCK-END
