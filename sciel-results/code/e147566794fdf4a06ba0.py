import time
import random
import copy

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit

    def count_nonzero_square(exps, coeffs):
        """
        Returns the number of nonzero terms in the exact integer square.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients (corresponding to exps)
        """
        if len(exps) != 12 or len(coeffs) != 12:
            return float('inf')
        
        # Check constraints
        if len(set(exps)) != 12:
            return float('inf')
        if max(exps) > 40:
            return float('inf')
        for c in coeffs:
            if c == 0 or not (-64 <= c <= 64):
                return float('inf')
        
        # Calculate square coefficients using a dictionary for sparse convolution
        square_dict = {}
        for i in range(12):
            ei = exps[i]
            ci = coeffs[i]
            for j in range(12):
                ej = exps[j]
                cj = coeffs[j]
                deg = ei + ej
                square_dict[deg] = square_dict.get(deg, 0) + ci * cj
        
        # Count nonzero
        return sum(1 for v in square_dict.values() if v != 0)

    def get_list_representation(exps, coeffs):
        """
        Converts sparse (exps, coeffs) to the required list format.
        """
        max_exp = max(exps)
        if max_exp > 40:
            return None
        res = [0] * (max_exp + 1)
        for e, c in zip(exps, coeffs):
            res[e] = c
        # Ensure length <= 41 and nonzero final coefficient
        if len(res) > 41:
            return None
        if res[-1] == 0:
            return None
        # Ensure exactly 12 nonzero terms
        if sum(1 for x in res if x != 0) != 12:
            return None
        return res

    # Initial population
    pop_size = 100
    population = []
    
    # Strategy for initialization:
    # 1. Arithmetic Progressions (dense sums, high collision chance)
    # 2. Random sets within [0, 40]
    
    for _ in range(pop_size):
        # 50% chance: Arithmetic Progression
        if random.random() < 0.5:
            start = random.randint(0, 10)
            step = random.randint(1, 3) # Small steps to keep degree low
            exps = [start + i * step for i in range(12)]
            if max(exps) > 40:
                exps = list(range(12)) # Fallback to 0..11
        else:
            # Random exponents, sorted
            exps = sorted(random.sample(range(0, 41), 12))
        
        # Random coefficients, mostly small to allow cancellation
        # Values from {-3, -2, -1, 1, 2, 3} are good for exact cancellation
        coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
        
        population.append((exps, coeffs))

    best_score = float('inf')
    best_candidate = None
    
    # Genetic Algorithm Loop
    generations = 0
    while time.time() < deadline:
        generations += 1
        
        # Evaluate population
        scored_pop = []
        for exps, coeffs in population:
            score = count_nonzero_square(exps, coeffs)
            scored_pop.append((score, exps, coeffs))
        
        # Find best in this generation
        scored_pop.sort(key=lambda x: x[0])
        current_best_score, current_best_exps, current_best_coeffs = scored_pop[0]
        
        if current_best_score < best_score:
            best_score = current_best_score
            best_candidate = (current_best_exps, current_best_coeffs)
            
        # If we found a very good score, we might be done?
        # Theoretical minimum for 12 terms is likely around 12-15.
        # If we hit 12, it's likely optimal or close to it.
        if best_score <= 12:
            break

        # Create next generation
        next_pop = []
        
        # Elitism: Keep top 20%
        elite_count = max(1, pop_size // 5)
        for i in range(elite_count):
            _, e, c = scored_pop[i]
            next_pop.append((e[:], c[:]))
        
        # Fill rest with offspring
        while len(next_pop) < pop_size and time.time() < deadline:
            # Select parents (Tournament selection)
            p1_idx = random.randint(0, len(scored_pop)-1)
            p2_idx = random.randint(0, len(scored_pop)-1)
            
            # Tournament
            if random.random() < 0.5:
                p1_idx = min(p1_idx, random.randint(0, len(scored_pop)-1), key=lambda i: scored_pop[i][0])
            if random.random() < 0.5:
                p2_idx = min(p2_idx, random.randint(0, len(scored_pop)-1), key=lambda i: scored_pop[i][0])
            
            _, e1, c1 = scored_pop[p1_idx]
            _, e2, c2 = scored_pop[p2_idx]
            
            # Crossover: Mix exponents and coefficients
            # Simple approach: Take exponents from parent 1, coefficients from parent 2?
            # No, that breaks the structure.
            # Better: Uniform crossover on the sorted lists.
            new_exps = []
            new_coeffs = []
            for i in range(12):
                if random.random() < 0.5:
                    new_exps.append(e1[i])
                    new_coeffs.append(c1[i])
                else:
                    new_exps.append(e2[i])
                    new_coeffs.append(c2[i])
            
            # Ensure uniqueness of exponents
            # If duplicates exist, fix them by adding small offsets
            # This is a rough fix, but valid for search
            unique_exps = []
            for e in new_exps:
                if e in unique_exps:
                    # Try to find a nearby unique exponent
                    for delta in [1, -1, 2, -2]:
                        ne = e + delta
                        if 0 <= ne <= 40 and ne not in unique_exps:
                            unique_exps.append(ne)
                            break
                    else:
                        unique_exps.append(e) # Fallback, will be invalid in eval
                else:
                    unique_exps.append(e)
            
            # Sort and align coefficients
            # Note: Coefficients are tied to the *original* index in the parent,
            # but since we are mixing, we just treat them as a set of (exp, coeff) pairs.
            # The order in the list doesn't matter for the polynomial sum.
            # So we just take the unique_exps and the corresponding coeffs?
            # Wait, the coeffs in new_coeffs correspond to the *chosen* parent's coefficient for that slot.
            # If we picked e1[i], we picked c1[i]. So the pair is (new_exps[i], new_coeffs[i]).
            # We just need to ensure exponents are unique.
            
            # Re-verify uniqueness properly
            final_exps = []
            final_coeffs = []
            used = set()
            for e, c in zip(new_exps, new_coeffs):
                if e in used:
                    # Mutate exponent to make it unique
                    while e in used or e < 0 or e > 40:
                        e += random.choice([-1, 1])
                used.add(e)
                final_exps.append(e)
                final_coeffs.append(c)
            
            next_pop.append((final_exps, final_coeffs))
            
        # Mutation
        for i in range(len(next_pop)):
            if random.random() < 0.3: # Mutate exponents
                idx = random.randint(0, 11)
                delta = random.choice([-2, -1, 1, 2])
                new_e = next_pop[i][0][idx] + delta
                if 0 <= new_e <= 40 and new_e not in next_pop[i][0]:
                    next_pop[i][0][idx] = new_e
                    next_pop[i][0].sort()
                    # Note: Coefficients stay with the term? 
                    # If we change the exponent, the coefficient stays with that "slot" in our internal representation.
                    # But the polynomial is defined by the set of (exp, coeff).
                    # So we just updated the exponent for that term.
            
            if random.random() < 0.5: # Mutate coefficients
                idx = random.randint(0, 11)
                # Change to a random small integer
                new_c = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_c != next_pop[i][1][idx]:
                    next_pop[i][1][idx] = new_c
                    
        population = next_pop

    # Final validation and formatting
    if best_candidate is not None:
        exps, coeffs = best_candidate
        res_list = get_list_representation(exps, coeffs)
        if res_list is not None:
            return res_list

    # Fallback: Return a valid, though not necessarily optimal, candidate
    # Use the seed-like structure: 12 ones
    return [1] * 12
