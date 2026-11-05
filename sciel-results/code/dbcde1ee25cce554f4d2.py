import time
import random
import math

N = 256
DEADLINE_MARGIN = 5.0

# Precomputation
# Bit i-1 represents element i (1-indexed in problem, 0-indexed in bit)
def bit(e):
    return 1 << (e - 1)

# Precompute 4-AP conflicts
# For each element x, store list of masks of the OTHER 3 elements in any 4-AP containing x.
# If A contains x and (A & mask) == mask, then adding x (or having x) creates a 4-AP.
# Actually, for a set A to be 4-AP free, for any 4-AP {a,b,c,d}, it must not contain all 4.
# Equivalently, if we try to add x to A, we must ensure that for every 4-AP containing x, 
# the other 3 elements are NOT all in A.

four_conflicts = [[] for _ in range(N + 1)]

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        
        # For each element, the conflict is the set of the other 3
        four_conflicts[a].append(bit(b) | bit(c) | bit(e))
        four_conflicts[b].append(bit(a) | bit(c) | bit(e))
        four_conflicts[c].append(bit(a) | bit(b) | bit(e))
        four_conflicts[e].append(bit(a) | bit(b) | bit(c))

# Precompute 3-AP gains
# For each element x, store list of masks of the OTHER 2 elements in any 3-AP containing x.
# If A contains x and (A & mask) == mask, then x completes a 3-AP.
three_gains = [[] for _ in range(N + 1)]

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        
        three_gains[a].append(bit(b) | bit(c))
        three_gains[b].append(bit(a) | bit(c))
        three_gains[c].append(bit(a) | bit(b))

def is_4ap_free(mask):
    """Check if the set represented by mask is 4-AP free."""
    # We can check this more efficiently by iterating through elements in the set
    # and checking their conflicts.
    # However, for a full check, we just need to ensure no 4-AP is fully contained.
    # A faster way: For each 4-AP, check if all 4 bits are set.
    # But we precomputed conflicts per element.
    # If an element x is in the set, and any of its conflict masks is fully contained in the set (excluding x itself, which is handled by the mask definition), then it's bad.
    # Wait, the conflict mask for x in a 4-AP {x, y, z, w} is bit(y)|bit(z)|bit(w).
    # If (mask & conflict_mask) == conflict_mask, then y,z,w are in the set. Since x is in the set, the 4-AP is complete.
    
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        # Index is log2(lsb) + 1
        # Instead of log2, we can precompute index for each power of 2? No, N=256 is small.
        # Or just iterate 1..N.
        pass
    
    # Iterating 1..N is O(N). N=256. This is fast enough.
    for i in range(1, N + 1):
        if (mask >> (i - 1)) & 1:
            for conflict in four_conflicts[i]:
                if (mask & conflict) == conflict:
                    return False
    return True

def count_3aps(mask):
    """Count distinct 3-APs in the set."""
    count = 0
    for i in range(1, N + 1):
        if (mask >> (i - 1)) & 1:
            for gain in three_gains[i]:
                if (mask & gain) == gain:
                    count += 1
    return count // 3

def random_valid_set(rng, density=0.5):
    """Generate a random set and repair it to be 4-AP free."""
    # Start with random bits
    mask = 0
    for i in range(1, N + 1):
        if rng.random() < density:
            mask |= bit(i)
    
    # Repair: Remove elements that cause 4-APs
    # This is a simple heuristic repair.
    # To make it better, we can be smarter about which element to remove.
    # But for initialization, random removal is okay.
    
    changed = True
    while changed:
        changed = False
        for i in range(1, N + 1):
            if (mask >> (i - 1)) & 1:
                for conflict in four_conflicts[i]:
                    if (mask & conflict) == conflict:
                        # Conflict found. Remove i (or one of the others).
                        # Removing i is simplest.
                        mask &= ~bit(i)
                        changed = True
                        break
                if changed:
                    break
    return mask

def mutate(mask, rng, rate=0.1):
    """Mutate a mask by flipping bits, ensuring validity."""
    new_mask = mask
    for i in range(1, N + 1):
        if rng.random() < rate:
            if (new_mask >> (i - 1)) & 1:
                # Try to remove
                new_mask &= ~bit(i)
            else:
                # Try to add
                # Check if adding i keeps it 4-AP free
                temp_mask = new_mask | bit(i)
                if is_4ap_free(temp_mask):
                    new_mask = temp_mask
                # If not free, we could try to remove conflicting elements, but that's complex.
                # Just skip if invalid.
    return new_mask

def crossover(m1, m2, rng):
    """Uniform crossover."""
    child = 0
    for i in range(1, N + 1):
        if rng.random() < 0.5:
            if (m1 >> (i - 1)) & 1:
                child |= bit(i)
        else:
            if (m2 >> (i - 1)) & 1:
                child |= bit(i)
    
    # Repair child to be 4-AP free
    changed = True
    while changed:
        changed = False
        for i in range(1, N + 1):
            if (child >> (i - 1)) & 1:
                for conflict in four_conflicts[i]:
                    if (child & conflict) == conflict:
                        child &= ~bit(i)
                        changed = True
                        break
                if changed:
                    break
    return child

def local_search(mask, deadline, rng):
    """
    Perform local search (Hill Climbing with random restarts/neighbors)
    """
    current_mask = mask
    current_score = count_3aps(current_mask)
    
    while time.time() < deadline - 1.0:
        improved = False
        
        # Try Additions
        best_add_x = -1
        best_add_delta = 0
        for x in range(1, N + 1):
            if (current_mask >> (x - 1)) & 1:
                continue
            # Check if adding x is valid
            if not is_4ap_free(current_mask | bit(x)):
                continue
            # Calculate delta
            delta = 0
            for gain in three_gains[x]:
                if (current_mask & gain) == gain:
                    delta += 1
            if delta > best_add_delta:
                best_add_delta = delta
                best_add_x = x
        
        if best_add_x != -1:
            current_mask |= bit(best_add_x)
            current_score += best_add_delta
            improved = True
            continue
            
        # Try Swaps
        best_swap = None
        best_swap_score = current_score
        
        # Sample a subset of swaps to save time? No, N=256 is small.
        # But O(N^2) swaps * O(N) check = O(N^3) = 16M ops. Might be slow in Python.
        # Let's optimize the swap check.
        
        # Precompute loss for each x in A
        losses = [0] * (N + 1)
        for x in range(1, N + 1):
            if (current_mask >> (x - 1)) & 1:
                for gain in three_gains[x]:
                    if (current_mask & gain) == gain:
                        losses[x] += 1
        
        for x in range(1, N + 1):
            if time.time() > deadline - 2.0:
                break
            if not ((current_mask >> (x - 1)) & 1):
                continue
            
            loss = losses[x]
            mask_without_x = current_mask & ~bit(x)
            
            for y in range(1, N + 1):
                if y == x:
                    continue
                if (mask_without_x >> (y - 1)) & 1:
                    continue
                
                # Check validity of adding y to mask_without_x
                # This is the expensive part.
                valid = True
                for conflict in four_conflicts[y]:
                    if (mask_without_x & conflict) == conflict:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate gain for y
                gain = 0
                for g in three_gains[y]:
                    if (mask_without_x & g) == g:
                        gain += 1
                
                new_score = current_score - loss + gain
                if new_score > best_swap_score:
                    best_swap_score = new_score
                    best_swap = (x, y)
                    
        if best_swap:
            x, y = best_swap
            current_mask = (current_mask & ~bit(x)) | bit(y)
            current_score = best_swap_score
            improved = True
            continue
            
        if not improved:
            # Stuck in local optimum. Perturb.
            # Remove a random element and add a random valid one?
            # Or just break and let GA handle it.
            break
            
    return current_mask, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 90.0 - DEADLINE_MARGIN
    
    rng = random.Random(42)
    
    # Initialize population
    pop_size = 20
    population = []
    scores = []
    
    for _ in range(pop_size):
        m = random_valid_set(rng, density=0.6)
        s = count_3aps(m)
        population.append(m)
        scores.append(s)
    
    best_mask = max(population, key=lambda m: count_3aps(m))
    best_score = count_3aps(best_mask)
    
    # GA Loop
    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        
        # Sort by score
        combined = sorted(zip(scores, population), reverse=True)
        scores = [c[0] for c in combined]
        population = [c[1] for c in combined]
        
        # Elitism: Keep top 2
        new_population = [population[0], population[1]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection: Tournament
            c1 = rng.choice(population[:10])
            c2 = rng.choice(population[:10])
            
            child = crossover(c1, c2, rng)
            child = mutate(child, rng, rate=0.05)
            
            # Local Search on child (with limited time)
            # We don't want to spend too much time on every child.
            # Maybe only do local search on the best few?
            # For now, just add to pool.
            new_population.append(child)
            
        population = new_population
        scores = [count_3aps(m) for m in population]
        
        # Update best
        current_best = max(scores)
        if current_best > best_score:
            best_score = current_best
            best_mask = population[scores.index(current_best)]
            
        # Periodic Local Search on the best individual
        if gen % 5 == 0:
            ls_mask, ls_score = local_search(best_mask, deadline, rng)
            if ls_score > best_score:
                best_score = ls_score
                best_mask = ls_mask
                # Inject improved best into population
                population[0] = best_mask
                scores[0] = best_score

    # Final local search on the best found
    if time.time() < deadline - 1.0:
        final_mask, final_score = local_search(best_mask, deadline, rng)
        if final_score > best_score:
            best_score = final_score
            best_mask = final_mask

    # Convert mask to list
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}, Score: {count_3aps(sum([bit(x) for x in res]))}")
