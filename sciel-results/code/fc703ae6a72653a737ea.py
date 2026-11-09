import time
import random
import math
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that 2*d <= 511 (since a>=1, a+2d<=512 => 2d<=511)

    def full_score(mask):
        """Calculate the number of distinct d > 0 such that a, a+d, a+2d are in the set."""
        count = 0
        for d in range(1, MAX_D + 1):
            if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        """Convert bitmask to list of integers."""
        res = []
        for i in range(1, MAX_VAL + 1):
            if (mask >> i) & 1:
                res.append(i)
        return res

    def list_to_mask(lst):
        """Convert list of integers to bitmask."""
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Precompute for each value x, the list of d's that involve x in a 3-AP
    # A 3-AP with difference d is (a, a+d, a+2d).
    # x can be:
    #  1. The first term: a = x. Then the AP is (x, x+d, x+2d). Valid if x+2d <= MAX_VAL.
    #     d ranges from 1 to (MAX_VAL - x) // 2.
    #  2. The middle term: a+d = x => a = x-d. AP is (x-d, x, x+d). Valid if x-d >= 1 and x+d <= MAX_VAL.
    #     d ranges from 1 to min(x-1, MAX_VAL-x).
    #  3. The last term: a+2d = x => a = x-2d. AP is (x-2d, x-d, x). Valid if x-2d >= 1.
    #     d ranges from 1 to (x-1) // 2.
    #
    # For delta scoring, when we remove x and add y, the d's whose status might change are
    # those d's that involve x OR those d's that involve y.
    # We need to know, for each d, how many 3-APs with difference d exist in the current set.
    # Let count[d] = number of 3-APs with difference d.
    # A d is "active" if count[d] > 0.
    # Score = number of active d's.
    #
    # When we swap x -> y:
    #   1. For each d in d_involving_x:
    #      - For each AP with difference d that includes x, decrement count[d].
    #   2. For each d in d_involving_y:
    #      - For each AP with difference d that includes y, increment count[d].
    #   3. Score change = (new active d's) - (old active d's).
    #
    # To make this efficient, we precompute for each x the list of (d, role) pairs,
    # but actually we just need the set of d's that involve x.
    # However, to update count[d] correctly, we need to know specifically which APs are affected.
    #
    # Alternative approach for delta scoring:
    # Instead of maintaining count[d], we can just check the affected d's.
    # For a swap x -> y:
    #   affected_d = set of d's that involve x OR y.
    #   For each d in affected_d:
    #     old_active = (mask has a 3-AP with diff d)
    #     new_active = (new_mask has a 3-AP with diff d)
    #     delta += (1 if new_active else 0) - (1 if old_active else 0)
    #
    # This is much simpler. The size of affected_d is at most:
    #   For x: d1 = (MAX_VAL - x) // 2, d2 = min(x-1, MAX_VAL-x), d3 = (x-1) // 2
    #   Total unique d's involving x is at most d1 + d2 + d3, but many overlap.
    #   In the worst case (x ~ 256), d1 ~ 128, d2 ~ 128, d3 ~ 128, so ~384 d's.
    #   But actually, d_involving_x is the union of:
    #     {1, ..., (MAX_VAL-x)//2}  (x as first term)
    #     {1, ..., min(x-1, MAX_VAL-x)}  (x as middle term)
    #     {1, ..., (x-1)//2}  (x as last term)
    #   This is just {1, ..., max((MAX_VAL-x)//2, min(x-1, MAX_VAL-x), (x-1)//2)}
    #   Which is at most ~255.
    #
    # So for each swap, we check at most ~255 d's. Each check is a bitmask AND operation.
    # This is still similar to full_score, but we only check the affected d's.
    # However, in practice, the affected d's for a swap are the union of d's involving x and d's involving y.
    # This union can be up to ~510 d's in the worst case, but typically less.
    #
    # Actually, let's think more carefully. The d's that involve x are:
    #   D_x = {d : there exists a 3-AP (a, a+d, a+2d) with x in it}
    #   x in AP means x=a, x=a+d, or x=a+2d.
    #   If x=a: d can be 1 to (MAX_VAL-x)//2
    #   If x=a+d: d can be 1 to min(x-1, MAX_VAL-x)
    #   If x=a+2d: d can be 1 to (x-1)//2
    #   So D_x = {1, 2, ..., max((MAX_VAL-x)//2, min(x-1, MAX_VAL-x), (x-1)//2)}
    #
    # Similarly for y.
    # D_affected = D_x ∪ D_y
    #
    # For each d in D_affected, we check if the old mask has a 3-AP with diff d,
    # and if the new mask has a 3-AP with diff d.
    #
    # This is at most ~510 checks per swap, each O(1) with bitmask.
    # Full score is 255 checks. So this is about 2x slower per evaluation than full score?
    # Wait, no. Full score checks all 255 d's. Delta score checks D_x ∪ D_y, which can be up to ~510.
    # That's actually slower!
    #
    # Let me reconsider. The key insight from the cell findings is "Delta scoring avoids full 255-iteration scan".
    # How? Maybe by maintaining count[d] and only updating the d's that change?
    #
    # Let's use the count[d] approach:
    # Maintain count[d] for d in 1..MAX_D.
    # Score = number of d with count[d] > 0.
    #
    # When we remove x:
    #   For each d in D_x:
    #     For each AP with diff d that includes x, decrement count[d].
    #     If count[d] becomes 0, score decreases by 1.
    #
    # When we add y:
    #   For each d in D_y:
    #     For each AP with diff d that includes y, increment count[d].
    #     If count[d] becomes 1, score increases by 1.
    #
    # To do this, we need to know, for each x and d in D_x, how many APs with diff d include x.
    # This is at most 3 (x as first, middle, or last term).
    #
    # Let's precompute for each x, a list of (d, role) where role indicates which position x is in.
    # Actually, for a given x and d, x can be in at most one AP with difference d? No.
    # For a fixed d and x, the APs containing x are:
    #   - (x, x+d, x+2d) if x+2d <= MAX_VAL
    #   - (x-d, x, x+d) if x-d >= 1 and x+d <= MAX_VAL
    #   - (x-2d, x-d, x) if x-2d >= 1
    # So at most 3 APs per (x, d) pair.
    #
    # This is getting complex. Let's use a simpler delta approach:
    # For a swap x -> y, the affected d's are D_x ∪ D_y.
    # For each d in D_affected:
    #   old_has = (mask & (mask >> d) & (mask >> (2*d))) != 0
    #   new_mask = mask ^ (1 << x) ^ (1 << y)
    #   new_has = (new_mask & (new_mask >> d) & (new_mask >> (2*d))) != 0
    #   delta += (1 if new_has else 0) - (1 if old_has else 0)
    #
    # The size of D_affected is |D_x ∪ D_y|.
    # D_x = {1, ..., M_x} where M_x = max((MAX_VAL-x)//2, min(x-1, MAX_VAL-x), (x-1)//2)
    # D_y = {1, ..., M_y} where M_y = max((MAX_VAL-y)//2, min(y-1, MAX_VAL-y), (y-1)//2)
    # D_x ∪ D_y = {1, ..., max(M_x, M_y)}
    #
    # So we check max(M_x, M_y) d's. This is at most 255.
    # In the best case, if x and y are both near the middle, M_x and M_y are ~128, so we check 128 d's.
    # If x is near 1 and y is near 512, M_x ~ 255, M_y ~ 255, so we check 255 d's.
    #
    # On average, this is about half of the full 255 checks. So ~2x speedup.
    #
    # But the cell findings say ~25x. Let me think again.
    # Maybe the delta scoring is even more efficient by only checking the d's where the count actually changes?
    #
    # For now, let's implement the union approach and see.
    #
    # Actually, let's use a different approach: maintain count[d] and update it incrementally.
    # This requires precomputing for each x, the list of (d, ap_index) pairs.
    # But that's complex. Let's stick with the union approach for simplicity.

    # Precompute D_x for each x
    D_x = [0] * (MAX_VAL + 1)
    for x in range(1, MAX_VAL + 1):
        d1 = (MAX_VAL - x) // 2
        d2 = min(x - 1, MAX_VAL - x)
        d3 = (x - 1) // 2
        D_x[x] = max(d1, d2, d3)

    def delta_score(mask, x, y):
        """Calculate the change in score when swapping x for y in the set represented by mask."""
        new_mask = mask ^ (1 << x) ^ (1 << y)
        max_d = max(D_x[x], D_x[y])
        delta = 0
        for d in range(1, max_d + 1):
            old_has = (mask & (mask >> d) & (mask >> (2 * d))) != 0
            new_has = (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0
            if new_has and not old_has:
                delta += 1
            elif old_has and not new_has:
                delta -= 1
        return delta

    # Genetic Algorithm
    POP_SIZE = 20
    ELITE_SIZE = 4
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.3

    # Initialize population
    population = []
    for _ in range(POP_SIZE):
        lst = random.sample(range(1, MAX_VAL + 1), N)
        mask = list_to_mask(lst)
        score = full_score(mask)
        population.append((lst, mask, score))

    # Sort by score
    population.sort(key=lambda x: x[2], reverse=True)

    best_list = population[0][0][:]
    best_mask = population[0][1]
    best_score = population[0][2]

    # Precompute some structured starts
    structured_starts = [
        list(range(1, 1 + 31 * 16, 16)),  # AP with d=16
        list(range(1, 33)),  # Dense at start
        list(range(481, 513)),  # Dense at end
    ]
    for lst in structured_starts:
        mask = list_to_mask(lst)
        score = full_score(mask)
        if score > best_score:
            best_score = score
            best_list = lst[:]
            best_mask = mask
        population.append((lst, mask, score))

    generation = 0
    while time.time() < deadline:
        generation += 1
        # Selection: tournament selection
        def tournament_select(pop):
            candidates = random.sample(pop, min(TOURNAMENT_SIZE, len(pop)))
            return max(candidates, key=lambda x: x[2])

        # Create new population
        new_population = []
        
        # Elitism: keep the best ELITE_SIZE individuals
        for i in range(min(ELITE_SIZE, len(population))):
            new_population.append(population[i])

        # Fill the rest with offspring
        while len(new_population) < POP_SIZE:
            if time.time() > deadline - 5:
                break
                
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover: uniform crossover
            child_list = []
            for i in range(N):
                if random.random() < 0.5:
                    child_list.append(parent1[0][i])
                else:
                    child_list.append(parent2[0][i])
            
            # Remove duplicates and ensure we have N distinct elements
            child_set = set(child_list)
            if len(child_set) < N:
                # Add random elements to fill
                while len(child_set) < N:
                    child_set.add(random.randint(1, MAX_VAL))
            child_list = list(child_set)
            
            # Mutation
            if random.random() < MUTATION_RATE:
                num_mutations = random.randint(1, 3)
                for _ in range(num_mutations):
                    idx = random.randint(0, N - 1)
                    old_val = child_list[idx]
                    new_val = random.randint(1, MAX_VAL)
                    if new_val not in child_set or new_val == old_val:
                        child_set.remove(old_val)
                        child_set.add(new_val)
                        child_list[idx] = new_val
            
            child_list = list(child_set)
            child_mask = list_to_mask(child_list)
            child_score = full_score(child_mask)
            new_population.append((child_list, child_mask, child_score))
        
        population = new_population
        population.sort(key=lambda x: x[2], reverse=True)
        
        if population[0][2] > best_score:
            best_score = population[0][2]
            best_list = population[0][0][:]
            best_mask = population[0][1]

        # Local search on the best individual
        if time.time() < deadline - 10:
            current_list = population[0][0][:]
            current_mask = population[0][1]
            current_score = population[0][2]
            current_set = set(current_list)
            
            T = 5.0
            cooling_rate = 0.9995
            min_T = 0.1
            stagnation_count = 0
            stagnation_limit = 500
            
            while time.time() < deadline - 5:
                idx = random.randint(0, N - 1)
                old_val = current_list[idx]
                
                # Pick a new value
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in current_set:
                    continue
                
                delta = delta_score(current_mask, old_val, new_val)
                new_score = current_score + delta
                
                if delta > 0:
                    accept = True
                    stagnation_count = 0
                else:
                    if T < 1e-10:
                        accept = False
                    else:
                        accept = random.random() < math.exp(delta / T)
                
                if accept:
                    current_list[idx] = new_val
                    current_set.remove(old_val)
                    current_set.add(new_val)
                    current_score = new_score
                    current_mask = current_mask ^ (1 << old_val) ^ (1 << new_val)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_list = current_list[:]
                        best_mask = current_mask
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
                else:
                    stagnation_count += 1
                
                T *= cooling_rate
                if T < min_T:
                    T = min_T
                
                if stagnation_count > stagnation_limit:
                    break

    return best_list
# EVOLVE-BLOCK-END
