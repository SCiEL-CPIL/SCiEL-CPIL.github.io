import time
import random
import numpy as np

N = 256

# Precompute 4-APs and 3-APs
four_aps = []
three_aps = []

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_aps.append((a, b, c, e))

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_aps.append((a, b, c))

# Precompute conflict masks for 4-APs
# For each element x, store a list of masks representing the other elements in 4-APs containing x
four_ap_conflict_masks = {i: [] for i in range(1, N + 1)}
for ap in four_aps:
    a, b, c, e = ap
    # For a
    mask = 0
    mask |= (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    four_ap_conflict_masks[a].append(mask)
    # For b
    mask = 0
    mask |= (1 << (a - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    four_ap_conflict_masks[b].append(mask)
    # For c
    mask = 0
    mask |= (1 << (a - 1)) | (1 << (b - 1)) | (1 << (e - 1))
    four_ap_conflict_masks[c].append(mask)
    # For e
    mask = 0
    mask |= (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
    four_ap_conflict_masks[e].append(mask)

# Precompute 3-AP partner masks for fast delta calculation? 
# Actually, for fitness, we just need to count total 3-APs.
# Let's precompute a list of (mask_a, mask_b, mask_c) for each 3-AP?
# Or just iterate through three_aps and check bits. 
# Given N=256, there are ~ (256/2)*(256/2) ~ 16k 3-APs? 
# Sum_{d=1}^{128} (256-2d) = 128*256 - 2*(128*129/2) = 32768 - 16512 = 16256 3-APs.
# Checking 16k 3-APs per fitness eval is fast enough if done in C/NumPy, but in Python loop it might be slow.
# Let's use a vectorized approach or precomputed bitmasks.

# Create an array of bitmasks for each 3-AP
three_ap_masks = []
for a, b, c in three_aps:
    m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
    three_ap_masks.append(m)

# Convert to numpy array for fast checking? 
# np.bitwise_and(A_mask, mask) == mask
# This is still a loop in Python if we do it element-wise.
# Alternative: Precompute for each element x, the list of 3-APs it belongs to?
# No, we need the count of COMPLETE 3-APs.

# Let's stick to a Python loop for fitness but optimize it.
# Or use a precomputed table? 
# Since we are using GA, fitness is called many times.
# Let's try to make fitness fast.

def is_4ap_free_mask(A_mask):
    """Check if A_mask is 4-AP free."""
    # Efficient check: iterate through elements in A_mask? No, iterate through 4-APs?
    # Better: for each element x in A_mask, check if any of its 4-APs are fully present.
    # But we need to check the whole set.
    # Fastest: Iterate through all 4-APs and check if all 4 bits are set in A_mask.
    # There are ~ (256/3)*(256/3) ~ 7k 4-APs.
    # Sum_{d=1}^{85} (256-3d) = 85*256 - 3*(85*86/2) = 21760 - 10965 = 10795 4-APs.
    # Checking 10k 4-APs per candidate is okay.
    
    # Even faster: Precompute masks for 4-APs
    # We already have four_aps. Let's create masks.
    # But we can also check during construction/mutation if we maintain validity.
    # However, GA crossover/mutation can break validity.
    # So we need a fast validator.
    
    # Let's precompute 4-AP masks
    four_ap_masks = []
    for a, b, c, e in four_aps:
        m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
        four_ap_masks.append(m)
        
    # Check if any 4-AP mask is a subset of A_mask
    for m in four_ap_masks:
        if (A_mask & m) == m:
            return False
    return True

# Precompute four_ap_masks globally
four_ap_masks = []
for a, b, c, e in four_aps:
    m = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
    four_ap_masks.append(m)

def fitness_mask(A_mask):
    """
    Return the number of 3-APs if A_mask is 4-AP free, else -1.
    """
    # Check 4-AP freedom
    for m in four_ap_masks:
        if (A_mask & m) == m:
            return -1
    
    # Count 3-APs
    count = 0
    for m in three_ap_masks:
        if (A_mask & m) == m:
            count += 1
    return count

def greedy_initial_mask(seed_val=0):
    """Generate a 4-AP free set using a greedy approach."""
    random.seed(seed_val)
    A_mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        bit = 1 << (x - 1)
        # Check if adding x creates a 4-AP
        valid = True
        # Check all 4-APs containing x
        # We can use four_ap_conflict_masks[x]
        for cmask in four_ap_conflict_masks[x]:
            if (A_mask & cmask) == cmask:
                valid = False
                break
        if valid:
            A_mask |= bit
    return A_mask

def crossover(parent1, parent2):
    """Uniform crossover."""
    mask1 = 0
    mask2 = 0
    for i in range(N):
        if random.random() < 0.5:
            if parent1 & (1 << i):
                mask1 |= (1 << i)
            if parent2 & (1 << i):
                mask2 |= (1 << i)
        else:
            if parent2 & (1 << i):
                mask1 |= (1 << i)
            if parent1 & (1 << i):
                mask2 |= (1 << i)
    return mask1, mask2

def mutate(A_mask, rate=0.05):
    """Bit-flip mutation."""
    for i in range(N):
        if random.random() < rate:
            A_mask ^= (1 << i)
    return A_mask

def repair(A_mask):
    """
    Try to repair A_mask to be 4-AP free by removing elements that cause 4-APs.
    This is a simple heuristic: if a 4-AP is found, remove a random element from it.
    """
    changed = True
    while changed:
        changed = False
        for m in four_ap_masks:
            if (A_mask & m) == m:
                # Found a 4-AP. Remove a random element from it.
                # Extract elements
                elems = []
                for i in range(N):
                    if m & (1 << i):
                        elems.append(i)
                # Pick one to remove
                victim = random.choice(elems)
                A_mask &= ~(1 << victim)
                changed = True
                break
    return A_mask

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initialize population
    pop_size = 50
    population = []
    for i in range(pop_size):
        population.append(greedy_initial_mask(i))
    
    # Evaluate initial population
    scores = [fitness_mask(m) for m in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_mask = population[best_idx]
    best_score = scores[best_idx]
    
    if best_score < 0:
        # Fallback if greedy failed (unlikely)
        best_mask = greedy_initial_mask(999)
        best_score = fitness_mask(best_mask)
        if best_score < 0:
             # Emergency repair
             best_mask = repair(best_mask)
             best_score = fitness_mask(best_mask)
    
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Sort population by fitness
        indexed_pop = [(scores[i], population[i]) for i in range(pop_size)]
        indexed_pop.sort(key=lambda x: x[0], reverse=True)
        population = [p[1] for p in indexed_pop]
        scores = [p[0] for p in indexed_pop]
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population = population[:elite_count]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                idxs = random.sample(range(pop_size), 3)
                best = max(idxs, key=lambda i: scores[i])
                return population[best]
            
            p1 = select()
            p2 = select()
            
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, rate=0.05)
            c2 = mutate(c2, rate=0.05)
            
            # Repair if necessary
            if fitness_mask(c1) < 0:
                c1 = repair(c1)
            if fitness_mask(c2) < 0:
                c2 = repair(c2)
                
            new_population.append(c1)
            new_population.append(c2)
            
            # Trim if over
            if len(new_population) >= pop_size:
                break
        
        # Evaluate new population
        new_scores = [fitness_mask(m) for m in new_population]
        
        # Update best
        new_best_idx = max(range(pop_size), key=lambda i: new_scores[i])
        if new_scores[new_best_idx] > best_score:
            best_score = new_scores[new_best_idx]
            best_mask = new_population[new_best_idx]
            
        population = new_population
        scores = new_scores
        
        # Local search on best found so far every few generations?
        # Maybe just rely on GA for now.
        
    # Final polish: simple local search on the best mask
    # Try adding/removing single elements to improve score while maintaining 4-AP freedom
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # Try additions
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if best_mask & bit:
                continue
            # Check 4-AP freedom
            valid = True
            for cmask in four_ap_conflict_masks[x]:
                if (best_mask & cmask) == cmask:
                    valid = False
                    break
            if valid:
                new_mask = best_mask | bit
                new_score = fitness_mask(new_mask)
                if new_score > current_score:
                    best_mask = new_mask
                    current_score = new_score
                    improved = True
                    break
        
        # Try removals (if adding didn't work, maybe removing a "bad" element allows more additions later? 
        # But we maximize 3-APs. Removing an element usually reduces 3-APs unless it was "blocking" many others?
        # Actually, removing an element never increases the number of 3-APs in the set.
        # So removal is only useful if it allows future additions that yield net gain.
        # Given time constraints, we'll skip removals in final polish to keep it simple, 
        # or just do a few swap attempts.
        
    # Convert to list
    result = []
    for i in range(1, N + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
    
    return sorted(result)

# EVOLVE-BLOCK-END
