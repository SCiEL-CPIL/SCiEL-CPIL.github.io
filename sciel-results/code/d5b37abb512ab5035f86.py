import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def count_violations(subset):
        s = set(subset)
        count = 0
        for a in subset:
            for b in subset:
                if a + b in s:
                    count += 1
        return count

    def solve_one(values, time_budget):
        n = len(values)
        if n == 0:
            return []
        
        best_subset = []
        best_size = 0
        rng_local = random.Random(12345)
        
        # Strategy 1: Modular arithmetic based initial solutions
        # For each modulus m, try to find a residue class that is sum-free
        moduli_to_try = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        for m in moduli_to_try:
            if time.time() > start_time + time_budget - 2.0:
                break
            
            # Try each residue class
            for r in range(m):
                # A residue class r mod m is sum-free if 2r != r' for any r' in the class
                # More precisely: if we take all x ≡ r (mod m), then a+b ≡ 2r (mod m)
                # We need 2r not ≡ r (mod m) for the set to be sum-free... 
                # Actually, the condition is: no a,b,c in S with a+b=c
                # If all elements ≡ r (mod m), then a+b ≡ 2r (mod m) and c ≡ r (mod m)
                # So we need 2r ≠ r (mod m), i.e., r ≠ 0 (mod m)
                # But even if 2r ≡ r (mod m), it's not necessarily a violation unless 2r mod m = r mod m AND there exist actual a,b,c
                # Let's just try it and check
                candidate = [x for x in values if x % m == r]
                if len(candidate) > best_size:
                    # Verify it's actually sum-free
                    if is_sum_free(candidate):
                        best_size = len(candidate)
                        best_subset = candidate
                    else:
                        # Try a greedy filter within this residue class
                        candidate_set = set(candidate)
                        filtered = []
                        F = set()
                        for x in sorted(candidate, key=abs):
                            if x not in F and 2*x not in candidate_set:
                                filtered.append(x)
                                for s in filtered[:-1]:
                                    F.add(x - s)
                                    F.add(s - x)
                        if len(filtered) > best_size and is_sum_free(filtered):
                            best_size = len(filtered)
                            best_subset = filtered
        
        # Strategy 2: Greedy with multiple orderings
        def greedy_solve(order):
            S = []
            S_set = set()
            F = set()
            for x in order:
                if x in F or 2*x in S_set:
                    continue
                S.append(x)
                S_set.add(x)
                for s in S[:-1]:
                    F.add(x - s)
                    F.add(s - x)
            return S

        order1 = sorted(values, key=lambda x: abs(x), reverse=True)
        res1 = greedy_solve(order1)
        if len(res1) > best_size:
            best_size = len(res1)
            best_subset = res1
            
        order2 = sorted(values, key=lambda x: abs(x))
        res2 = greedy_solve(order2)
        if len(res2) > best_size:
            best_size = len(res2)
            best_subset = res2

        # Strategy 3: Simulated annealing on the current best
        if time.time() < start_time + time_budget - 5.0:
            current = list(best_subset)
            current_set = set(current)
            current_energy = count_violations(current)
            
            # Temperature schedule
            t_start = 1.0
            t_end = 0.01
            cooling = 0.995
            temperature = t_start
            
            # Precompute for efficiency
            values_set = set(values)
            
            while temperature > t_end and time.time() < start_time + time_budget - 3.0:
                # Random move: add, remove, or swap
                move_type = rng_local.choice(['add', 'remove', 'swap'])
                
                if move_type == 'add':
                    # Try to add a random element not in current
                    not_in_current = [x for x in values if x not in current_set]
                    if not not_in_current:
                        temperature *= cooling
                        continue
                    x = rng_local.choice(not_in_current)
                    new_set = current_set | {x}
                    new_energy = count_violations(list(new_set))
                    delta = new_energy - current_energy
                    if delta <= 0 or rng_local.random() < math.exp(-delta / temperature):
                        current_set = new_set
                        current = list(new_set)
                        current_energy = new_energy
                elif move_type == 'remove':
                    if not current:
                        temperature *= cooling
                        continue
                    x = rng_local.choice(current)
                    new_set = current_set - {x}
                    new_energy = count_violations(list(new_set))
                    delta = new_energy - current_energy
                    if delta <= 0 or rng_local.random() < math.exp(-delta / temperature):
                        current_set = new_set
                        current = list(new_set)
                        current_energy = new_energy
                elif move_type == 'swap':
                    not_in_current = [x for x in values if x not in current_set]
                    if not not_in_current or not current:
                        temperature *= cooling
                        continue
                    x_out = rng_local.choice(current)
                    x_in = rng_local.choice(not_in_current)
                    new_set = (current_set - {x_out}) | {x_in}
                    new_energy = count_violations(list(new_set))
                    delta = new_energy - current_energy
                    if delta <= 0 or rng_local.random() < math.exp(-delta / temperature):
                        current_set = new_set
                        current = list(new_set)
                        current_energy = new_energy
                
                # Track best
                if current_energy == 0 and len(current) > best_size:
                    best_size = len(current)
                    best_subset = current[:]
                
                temperature *= cooling
        
        # Final verification
        if best_subset and not is_sum_free(best_subset):
            # Fallback to a verified sum-free set
            odd_subset = [x for x in values if x % 2 != 0]
            if is_sum_free(odd_subset):
                return odd_subset
            # Greedy fallback
            return greedy_solve(sorted(values, key=lambda x: abs(x), reverse=True))
        
        return best_subset

    results = []
    total_budget = deadline - start_time
    per_instance_budget = total_budget / 3.0
    
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2 != 0])
            continue
            
        vals = instance(size, seed)
        remaining = min(per_instance_budget, deadline - time.time())
        res = solve_one(vals, remaining)
        results.append(res)
        
    return results
