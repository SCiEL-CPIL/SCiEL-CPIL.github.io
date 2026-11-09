import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCD matrix
    G = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            G[i][j] = gcd(i, j)
            G[j][i] = G[i][j]
            
    # Identify all bad triples (a,b,c) with a<b<c and gcd(a,b)==gcd(a,c)==gcd(b,c)
    # Represent triples as bitmasks of the 3 elements
    bad_triples = []
    triple_masks = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = G[a][b]
            for c in range(b + 1, N + 1):
                if G[a][c] == g_ab and G[b][c] == g_ab:
                    mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
                    bad_triples.append((a, b, c))
                    triple_masks.append(mask)
    
    T = len(bad_triples)
    
    # Precompute for each element (bit position), the list of triple indices containing it
    # This helps in fast validity checks if needed, but bitmask intersection is faster for global checks
    # Actually, for GA fitness, we just need to check if S is independent.
    # S is independent iff for all t in bad_triples: S & t_mask != t_mask (i.e., not all 3 are in S)
    # Equivalently: S is valid if (S & t_mask).bit_count() < 3 for all t.
    # Or: No t_mask is a subset of S.
    
    def is_valid(S_mask):
        # Check if S_mask contains any bad triple
        # A triple is contained if (S_mask & t_mask) == t_mask
        for t_mask in triple_masks:
            if (S_mask & t_mask) == t_mask:
                return False
        return True
    
    def count_violations(S_mask):
        # Count how many bad triples are fully contained in S_mask
        # Used for penalty in fitness
        violations = 0
        for t_mask in triple_masks:
            if (S_mask & t_mask) == t_mask:
                violations += 1
        return violations

    # Initial Population
    pop_size = 50
    population = []
    
    # 1. Greedy construction (randomized order)
    def greedy_construct():
        S = 0
        remaining = list(range(N))
        random.shuffle(remaining)
        for i in remaining:
            # Check if adding i creates a violation
            # We can check if any bad triple containing i is now fully in S | {i}
            # Faster: just check all triples? No, T is small enough?
            # T is roughly sum over g of C(n_g, 3). 
            # Let's just use the global check for simplicity in the first pass, 
            # but optimize if needed.
            # Actually, checking all T triples for each of N additions is N*T ~ 128*1000 = 128k ops.
            # 50 restarts * 128k = 6.4M ops. Very fast.
            new_S = S | (1 << i)
            if is_valid(new_S):
                S = new_S
        return S

    # Generate initial population with diverse greedy starts
    for _ in range(pop_size // 2):
        population.append(greedy_construct())
        
    # 2. Random subsets (biased towards larger sets)
    for _ in range(pop_size // 2):
        S = 0
        for i in range(N):
            if random.random() < 0.6: # Bias towards inclusion
                S |= (1 << i)
        population.append(S)

    # Evaluate initial fitness
    # Fitness = size if valid, else size - penalty * violations
    # We want to maximize size.
    # Let's track (size, violations, mask)
    
    def evaluate(S_mask):
        size = S_mask.bit_count()
        # Quick check: if size is small, likely valid? No.
        # We need exact violations for penalty.
        # Optimization: If we know a set is valid, violations=0.
        # We'll cache validity? No, masks are unique.
        viol = count_violations(S_mask)
        # Penalize heavily to ensure valid sets are preferred
        # If viol > 0, fitness should be lower than any valid set with smaller size?
        # Max size is 128. Min valid size is 1.
        # If we use penalty = 1000 * viol, then a set of size 100 with 1 violation
        # has fitness 100 - 1000 = -900.
        # A valid set of size 50 has fitness 50.
        # So valid sets always beat invalid ones with significant violations.
        # But we want to explore invalid states to jump out of local minima.
        # Maybe softer penalty?
        # Let's use: fitness = size - 10 * viol
        # This allows accepting invalid states if they have many elements, 
        # but valid states are still strongly preferred.
        fitness = size - 10 * viol
        return fitness, size, viol

    # Initialize population stats
    pop_stats = []
    for S in population:
        f, s, v = evaluate(S)
        pop_stats.append((f, s, v, S))
        
    # Sort by fitness
    pop_stats.sort(key=lambda x: x[0], reverse=True)
    
    best_fitness = pop_stats[0][0]
    best_size = pop_stats[0][1]
    best_mask = pop_stats[0][3]
    
    # Genetic Algorithm Loop
    # Crossover: Bitwise AND, OR, or Uniform?
    # Uniform crossover: For each bit, pick from parent1 or parent2 randomly.
    # Mutation: Flip bits with probability p_mut.
    
    # Adaptive parameters
    p_crossover = 0.8
    p_mutation = 0.05
    
    generation = 0
    max_generations = 5000 # Limit, but time is the real constraint
    
    while time.time() < deadline:
        generation += 1
        new_population = []
        
        # Elitism: Keep top 5
        for i in range(min(5, len(pop_stats))):
            new_population.append(pop_stats[i])
            
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 2.0:
                break
                
            # Selection: Tournament
            def tournament_select(pop):
                k = 3
                candidates = random.sample(pop, k)
                return max(candidates, key=lambda x: x[0])
            
            p1 = tournament_select(pop_stats)
            p2 = tournament_select(pop_stats)
            
            if random.random() < p_crossover:
                # Uniform Crossover
                S_child = 0
                for i in range(N):
                    if random.random() < 0.5:
                        if p1[3] & (1 << i):
                            S_child |= (1 << i)
                    else:
                        if p2[3] & (1 << i):
                            S_child |= (1 << i)
            else:
                # No crossover, just copy parent
                S_child = p1[3]
                
            # Mutation
            # Flip bits
            # We can iterate through set bits and clear them, or unset bits and set them.
            # Or just iterate all N bits.
            for i in range(N):
                if random.random() < p_mutation:
                    S_child ^= (1 << i)
            
            # Evaluate
            f, s, v = evaluate(S_child)
            new_population.append((f, s, v, S_child))
            
        # Update population
        pop_stats = new_population
        pop_stats.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if pop_stats[0][0] > best_fitness:
            best_fitness = pop_stats[0][0]
            best_size = pop_stats[0][1]
            best_mask = pop_stats[0][3]
            
        # Local Search on the best individual found so far (or current best in pop)
        # This helps polish the solution.
        # We will run a simple 1-swap local search on the top individual of the current gen.
        # But GA already does this. Maybe run a dedicated local search on the global best every 50 gens?
        
        if generation % 50 == 0:
            # Run local search on best_mask
            current_S = best_mask
            improved = True
            while improved:
                improved = False
                # Try removing an element
                # If valid after removal, do it.
                # We need to check validity quickly.
                # Removing element i:
                #   If current_S is valid, new_S is valid iff no triple becomes "fully contained" by removal?
                #   No, removal only makes it MORE valid (fewer elements, so less likely to contain a triple).
                #   Wait, independent set property: If S is independent, any subset is independent.
                #   So if we start from a VALID set, removing elements keeps it valid.
                #   So we can greedily remove elements as long as they are not "essential"?
                #   Actually, we want to MAXIMIZE size. So we should ADD elements.
                #   If S is valid, adding an element might make it invalid.
                #   So local search should try to ADD elements.
                
                # Let's try adding elements to best_mask
                # For each i not in S:
                #   if S | (1<<i) is valid, add it.
                # Repeat until no more can be added.
                
                # This is a greedy repair/extension.
                added = True
                while added:
                    added = False
                    for i in range(N):
                        if not (best_mask & (1 << i)):
                            new_S = best_mask | (1 << i)
                            if is_valid(new_S):
                                best_mask = new_S
                                added = True
                                # Update best stats if valid
                                new_size = best_mask.bit_count()
                                if new_size > best_size:
                                    best_size = new_size
                                    best_fitness = new_size # Violations = 0
                                    
                # Also try removing elements that allow adding others? (2-swap)
                # This is more complex. Let's stick to extension for now.
                # The GA handles the "move to a different basin" part.
                # The local search handles the "climb the peak" part.

    # Final check: Ensure best_mask is valid
    # If best_mask is invalid (violations > 0), we need to repair it.
    # The fitness function penalizes violations, so the "best" by fitness might still be invalid
    # if the penalty is too low relative to size gains.
    # Let's verify and repair if necessary.
    
    if count_violations(best_mask) > 0:
        # Repair: Remove elements involved in violations until valid
        # Greedy removal
        S = best_mask
        viol = count_violations(S)
        while viol > 0:
            # Find a violating triple
            # We need to find which triples are violated.
            # Modify count_violations to return the list? Or just re-scan.
            removed = False
            for t_mask in triple_masks:
                if (S & t_mask) == t_mask:
                    # Remove one element from this triple
                    # Pick one at random or the one that appears in most other violations?
                    # Just pick the first one.
                    # Find an element in t_mask that is in S
                    for i in range(N):
                        if t_mask & (1 << i):
                            S ^= (1 << i)
                            removed = True
                            break
                    break
            if not removed:
                break
            viol = count_violations(S)
        best_mask = S
        best_size = best_mask.bit_count()
        
    # Convert mask to list
    S_list = [i + 1 for i in range(N) if best_mask & (1 << i)]
    
    if not S_list:
        return [1]
        
    return S_list

# EVOLVE-BLOCK-START
# (Note: The code above is the implementation. The EVOLVE-BLOCK markers are for the evolution system.)
# EVOLVE-BLOCK-END
