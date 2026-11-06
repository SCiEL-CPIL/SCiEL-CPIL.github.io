"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
K = 4  # length of progression

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) within 1..256
# We store them as 0-indexed tuples (i, i+d, i+2d, i+3d)
APs = []
ap_indices_by_pos = [[] for _ in range(N)]

for d in range(1, N):
    max_a = N - 3 * d
    if max_a < 1:
        break
    for a in range(1, max_a + 1):
        i1 = a - 1
        i2 = i1 + d
        i3 = i2 + d
        i4 = i3 + d
        ap_id = len(APs)
        APs.append((i1, i2, i3, i4))
        ap_indices_by_pos[i1].append(ap_id)
        ap_indices_by_pos[i2].append(ap_id)
        ap_indices_by_pos[i3].append(ap_id)
        ap_indices_by_pos[i4].append(ap_id)

TOTAL_APS = len(APs)

def count_mono_4ap_fast(coloring):
    """Count monochromatic 4-term APs in a 0/1 coloring of 1..256.
    Used for final verification or initial scoring if needed, but GA uses incremental."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3 * d
        if max_a < 1:
            break
        seq1 = c[:max_a]
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return count

class Individual:
    __slots__ = ['coloring', 'score']
    
    def __init__(self, coloring):
        self.coloring = np.array(coloring, dtype=np.int8)
        self.score = self._compute_score()
        
    def _compute_score(self):
        c = self.coloring
        count = 0
        # Vectorized check over all APs is faster for initial population
        # But let's use the precomputed list for consistency if we were doing it manually.
        # For speed in init, numpy is better.
        for ap in APs:
            i1, i2, i3, i4 = ap
            if c[i1] == c[i2] == c[i3] == c[i4]:
                count += 1
        return count
        
    def flip_delta(self, pos):
        """Calculate the change in score if we flip the bit at 'pos'."""
        c = self.coloring
        old_bit = c[pos]
        
        # Count how many APs containing 'pos' are currently monochromatic
        # and how many would be monochromatic if we flipped 'pos'.
        
        # An AP is monochromatic if all 4 bits are equal.
        # If we flip bit at 'pos':
        # 1. APs that were mono and contained 'pos' with the same color as the rest:
        #    - If the other 3 were 'old_bit', the AP was mono (all old_bit).
        #      After flip, the AP has 3 old_bit and 1 (1-old_bit). Not mono.
        #      So we lose 1 mono AP for each such AP.
        # 2. APs that were NOT mono and contained 'pos':
        #    - If the other 3 were (1-old_bit), the AP was (old_bit, 1-old_bit, 1-old_bit, 1-old_bit).
        #      After flip, the AP has 4 (1-old_bit). It becomes mono.
        #      So we gain 1 mono AP for each such AP.
        
        delta = 0
        target_bit = 1 - old_bit
        
        for ap_id in ap_indices_by_pos[pos]:
            i1, i2, i3, i4 = APs[ap_id]
            
            # Get the other 3 indices
            # Optimization: just check the 3 other bits
            # We need to check if the other 3 are all old_bit (lose) or all target_bit (gain)
            
            # Unroll for speed? There are at most ~100 APs per position.
            # Let's just iterate.
            
            # Indices other than pos
            # Since APs are small tuples, we can just pick the 3 that aren't pos.
            # But pos is one of i1..i4.
            
            # Helper to get the 3 other values
            # This is a bit slow in Python loop.
            # Alternative: precompute for each AP, the mask of which position is which?
            # No, let's just do it simply.
            
            # Since we are in a tight loop, let's optimize the inner logic.
            # We can precompute for each AP a list of the 3 other indices relative to each position?
            # That would be memory heavy but fast.
            # Let's try a simpler approach first: just check the 4 values.
            
            # Actually, we can precompute for each AP, the XOR of the 4 bits? No.
            
            # Let's just check the other 3.
            # We know which index is 'pos'.
            # If i1 == pos: others are i2, i3, i4
            # If i2 == pos: others are i1, i3, i4
            # etc.
            
            if i1 == pos:
                b1, b2, b3 = c[i2], c[i3], c[i4]
            elif i2 == pos:
                b1, b2, b3 = c[i1], c[i3], c[i4]
            elif i3 == pos:
                b1, b2, b3 = c[i1], c[i2], c[i4]
            else: # i4 == pos
                b1, b2, b3 = c[i1], c[i2], c[i3]
                
            # Check if b1, b2, b3 are all old_bit
            if b1 == old_bit and b2 == old_bit and b3 == old_bit:
                delta -= 1 # Was mono, now not
            # Check if b1, b2, b3 are all target_bit
            elif b1 == target_bit and b2 == target_bit and b3 == target_bit:
                delta += 1 # Was not mono, now is
        
        return delta

def local_search(individual, max_steps=1000, deadline=None):
    """Hill climbing using incremental delta updates."""
    current = individual
    improved = True
    steps = 0
    
    while improved and steps < max_steps:
        if deadline and time.time() > deadline:
            break
        improved = False
        best_pos = -1
        best_delta = 0
        current_score = current.score
        
        # Try flipping each position
        for pos in range(N):
            d = current.flip_delta(pos)
            if d < best_delta:
                best_delta = d
                best_pos = pos
                
        if best_pos != -1:
            # Apply flip
            current.coloring[best_pos] = 1 - current.coloring[best_pos]
            current.score += best_delta
            improved = True
            steps += 1
            
    return current

def genetic_algorithm(deadline):
    pop_size = 100
    mutation_rate = 0.05
    elite_size = 10
    
    # Initialize population with diverse random and structured starts
    population = []
    
    # Random
    for _ in range(pop_size // 2):
        c = np.random.randint(0, 2, N).astype(np.int8)
        population.append(Individual(c))
        
    # Structured patterns
    patterns = []
    # Alternating
    patterns.append(np.array([i % 2 for i in range(N)], dtype=np.int8))
    # Blocks
    for bs in [2, 4, 8, 16, 32, 64]:
        patterns.append(np.array([(i // bs) % 2 for i in range(N)], dtype=np.int8))
    # Van der Waerden like mod
    for mod in [3, 5, 7, 11]:
        patterns.append(np.array([1 if (i % mod) < mod//2 else 0 for i in range(N)], dtype=np.int8))
        
    for p in patterns:
        # Add noise
        c = p.copy()
        mask = np.random.rand(N) < 0.1
        c[mask] = 1 - c[mask]
        population.append(Individual(c))
        
    # Fill rest with random
    while len(population) < pop_size:
        c = np.random.randint(0, 2, N).astype(np.int8)
        population.append(Individual(c))
        
    # Sort initial population
    population.sort(key=lambda x: x.score)
    
    best_individual = population[0]
    
    generation = 0
    start_time = time.time()
    
    while time.time() < deadline - 5:
        generation += 1
        
        # Elitism
        new_population = population[:elite_size]
        
        # Selection and Crossover
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = random.choice(population[:pop_size//2])
            parent2 = random.choice(population[:pop_size//2])
            
            # Uniform Crossover
            mask = np.random.randint(0, 2, N).astype(np.int8)
            child_coloring = np.where(mask == 1, parent1.coloring, parent2.coloring)
            
            # Mutation
            mut_mask = np.random.rand(N) < mutation_rate
            child_coloring[mut_mask] = 1 - child_coloring[mut_mask]
            
            child = Individual(child_coloring)
            
            # Local Search on child (limited steps to save time)
            # We use a small budget for local search per child
            if time.time() < deadline - 10:
                child = local_search(child, max_steps=50, deadline=deadline - 5)
            
            new_population.append(child)
            
        # Update population
        population = new_population
        population.sort(key=lambda x: x.score)
        
        if population[0].score < best_individual.score:
            best_individual = population[0]
            
        # Adaptive mutation rate?
        # If no improvement in last 10 generations, increase mutation
        if generation % 10 == 0 and population[0].score == best_individual.score:
            mutation_rate = min(0.2, mutation_rate * 1.2)
            
    return best_individual.coloring.tolist()

def construct():
    deadline = time.time() + 110 # Safety margin
    
    # Run GA
    result = genetic_algorithm(deadline)
    
    # Final polish with more aggressive local search on the best result
    best_ind = Individual(result)
    best_ind = local_search(best_ind, max_steps=5000, deadline=deadline)
    
    return best_ind.coloring.tolist()
# EVOLVE-BLOCK-END
