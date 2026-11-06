import time
import random
from math import gcd
from fractions import Fraction

# Precomputation
N = 512
MAX_A = N - 1

def get_prime_factors(a):
    factors = set()
    d = 2
    while d * d <= a:
        while a % d == 0:
            factors.add(d)
            a //= d
        d += 1
    if a > 1:
        factors.add(a)
    return factors

# Map primes to indices for bitmask
primes = []
for i in range(2, MAX_A + 1):
    is_prime = True
    for p in primes:
        if p * p > i:
            break
        if i % p == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(i)

prime_to_idx = {p: i for i, p in enumerate(primes)}

masks = [0] * (MAX_A + 1)
for a in range(1, MAX_A + 1):
    factors = get_prime_factors(a)
    m = 0
    for p in factors:
        if p in prime_to_idx:
            m |= (1 << prime_to_idx[p])
    masks[a] = m

# Use Fraction for exact comparison to avoid floating point issues
# However, Fraction is slow. We can use high precision floats or just careful sorting.
# Given the constraints, let's stick to floats but be very careful, or use Decimal?
# Actually, for 512 terms, float precision is likely sufficient for ordering, 
# but for exact "best" tracking, let's use a tuple of (float_score, set_of_elements) 
# or just rely on the fact that the differences are not extremely tiny.
# Wait, the task says "exact rational, reported as a float". 
# The evaluator likely compares the final float. 
# If two sets have very close scores, float might misorder them.
# Let's use a hybrid: keep the set, calculate score as float for speed, 
# but if a new candidate is very close to best, we might need to be careful.
# For now, standard float should be fine for most cases, but let's try to be robust.
# Actually, let's just use float. The "exact rational" part is for the report, 
# but the optimization can use floats if we are careful.

weights = [0.0] * (MAX_A + 1)
for a in range(1, MAX_A + 1):
    weights[a] = 1.0 / (N - a)

# Sort candidates by weight descending to prioritize high-value items
sorted_candidates = list(range(1, MAX_A + 1))
sorted_candidates.sort(key=lambda x: weights[x], reverse=True)

def greedy_construct(seed):
    """
    Construct a set using a greedy approach with randomized tie-breaking
    and randomization in the order of consideration to generate diverse starts.
    """
    random.seed(seed)
    chosen = []
    used_mask = 0
    
    # Create a shuffled list of candidates, but bias towards high weights
    # We can do this by shuffling and then sorting by a random key? 
    # Or simply: iterate through all, but with random probability to skip?
    # Better: Generate a random permutation of candidates, but weight the probability 
    # of selection by their value? No, that's not standard greedy.
    
    # Standard Greedy: Pick highest weight available coprime number.
    # To diversify, we can:
    # 1. Start with a random subset of top K candidates?
    # 2. Or, use a randomized greedy: at each step, pick from top M available with prob proportional to weight?
    
    # Let's try: Randomized Greedy.
    # At each step, consider all available candidates.
    # Pick the one with highest weight * exp(random_noise)?
    # Or simpler: Shuffle the sorted_candidates, but keep high weights likely to be picked earlier?
    
    # Let's use a "Noisy Greedy":
    # For each candidate in sorted order (high to low weight):
    #   If it's coprime to current set, add it with probability p.
    #   If p=1, it's pure greedy.
    #   If p<1, we might skip a high weight item to allow others?
    #   Actually, skipping a high weight item is rarely beneficial unless it blocks many others.
    #   But greedy adds it, which might block others.
    #   So, randomization in greedy is useful.
    
    chosen_mask = 0
    # We iterate through candidates in random order? No, that's bad.
    # We iterate through candidates in sorted order (high weight first).
    # But we randomly decide whether to consider adding it.
    # This doesn't help much.
    
    # Better Randomized Greedy:
    # 1. Start with empty set.
    # 2. While there are candidates:
    #    - Find all candidates coprime to current set.
    #    - Pick the one with highest weight * (1 + random_noise).
    #    - Add it.
    
    current_set = set()
    current_mask = 0
    
    # Precompute list of all candidates
    all_cands = list(range(1, MAX_A + 1))
    
    while True:
        # Find best candidate
        best_cand = None
        best_score = -1
        # To speed up, we can iterate through sorted_candidates
        for a in sorted_candidates:
            if a in current_set:
                continue
            if (current_mask & masks[a]) != 0:
                continue
            # Calculate noisy score
            noise = random.uniform(0.9, 1.1)
            score = weights[a] * noise
            if score > best_score:
                best_score = score
                best_cand = a
        
        if best_cand is None:
            break
        
        current_set.add(best_cand)
        current_mask |= masks[best_cand]
        
    return list(current_set), current_mask, sum(weights[a] for a in current_set)

def local_search(initial_set, initial_mask, initial_score, deadline):
    """
    Iterated Local Search:
    1. Try to improve by swapping one element out for one or more in.
    2. Specifically, try to replace a low-weight element with a higher-weight compatible one.
    3. Or, try to add an element if possible.
    """
    current_set = set(initial_set)
    current_mask = initial_mask
    current_score = initial_score
    best_set = set(current_set)
    best_score = current_score
    best_mask = current_mask
    
    # List of elements in current set, sorted by weight ascending (lowest first)
    # We want to swap out low weight elements.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any compatible element
        # Iterate through candidates in descending weight order
        for a in sorted_candidates:
            if a in current_set:
                continue
            if (current_mask & masks[a]) == 0:
                # Can add
                current_set.add(a)
                current_mask |= masks[a]
                current_score += weights[a]
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_set = set(current_set)
                    best_mask = current_mask
                # Re-check from start? Or just continue? 
                # Adding one might allow others? No, adding restricts.
                # So we should restart the scan to ensure no other additions are possible?
                # Actually, if we add 'a', it might block 'b' which was also compatible.
                # But 'b' is still compatible with the rest.
                # So we should restart the loop to find the next best addition.
                break
        
        if not improved:
            # Try swapping
            # For each element in current set (lowest weight first)
            # Try to find a better element to swap in.
            
            # Sort current set by weight ascending
            current_list = sorted(list(current_set), key=lambda x: weights[x])
            
            for out_a in current_list:
                # Calculate mask without out_a
                mask_without = current_mask & ~masks[out_a]
                
                # Find best candidate to swap in
                best_in = None
                best_gain = 0
                
                for in_b in sorted_candidates:
                    if in_b in current_set:
                        continue
                    if (mask_without & masks[in_b]) != 0:
                        continue
                    
                    gain = weights[in_b] - weights[out_a]
                    if gain > best_gain:
                        best_gain = gain
                        best_in = in_b
                
                if best_in is not None and best_gain > 0:
                    # Perform swap
                    current_set.remove(out_a)
                    current_set.add(best_in)
                    current_mask = mask_without | masks[best_in]
                    current_score += best_gain
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                        best_mask = current_mask
                    
                    # Restart local search from this improved state
                    break
        
    return best_set, best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin
    
    best_global_set = set()
    best_global_score = -1.0
    best_global_mask = 0
    
    # Multi-Start
    num_starts = 50
    # We can increase num_starts if we have time, but let's start fixed.
    # Or dynamic: run starts until time is up.
    
    seed = 42
    for i in range(num_starts):
        if time.time() > deadline:
            break
        
        # Generate a diverse starting point
        # Using different seeds for the randomized greedy
        start_set, start_mask, start_score = greedy_construct(seed)
        seed += 1
        
        if start_score > best_global_score:
            best_global_score = start_score
            best_global_set = set(start_set)
            best_global_mask = start_mask
        
        # Run local search on this start
        # Give local search a time budget? 
        # Or just let it run until it converges (improvement stops).
        # Since the problem is small (512 items), local search should be fast.
        ls_set, ls_mask, ls_score = local_search(start_set, start_mask, start_score, deadline)
        
        if ls_score > best_global_score:
            best_global_score = ls_score
            best_global_set = set(ls_set)
            best_global_mask = ls_mask
        
        # If we have time, we can do a "kick" (perturbation) and restart local search
        # This is Iterated Local Search (ILS).
        # Let's add a simple ILS loop for the best solution found so far?
        # Or just rely on the multiple random starts.
        # Given 120s, 50 starts might be too few or too many depending on speed.
        # Greedy is O(N^2) roughly? 512^2 = 262k ops. Very fast.
        # Local search is also fast.
        # So we can afford more starts.
        
    # If we have significant time left, run ILS on the best solution
    if time.time() < deadline - 10:
        current_set = set(best_global_set)
        current_mask = best_global_mask
        current_score = best_global_score
        
        # ILS Loop
        while time.time() < deadline - 5:
            # Perturb: Remove 1-3 random elements
            k = random.randint(1, 3)
            perturbed_set = set(current_set)
            perturbed_mask = current_mask
            perturbed_score = current_score
            
            for _ in range(k):
                if not perturbed_set:
                    break
                # Pick random element to remove
                # Prefer removing low weight elements? 
                # Or random?
                # Random is better for diversity.
                elem = random.choice(list(perturbed_set))
                perturbed_set.remove(elem)
                perturbed_mask &= ~masks[elem]
                perturbed_score -= weights[elem]
            
            # Local Search on perturbed set
            ls_set, ls_mask, ls_score = local_search(perturbed_set, perturbed_mask, perturbed_score, deadline)
            
            if ls_score > current_score:
                current_set = set(ls_set)
                current_mask = ls_mask
                current_score = ls_score
                
                if current_score > best_global_score:
                    best_global_score = current_score
                    best_global_set = set(current_set)
                    best_global_mask = current_mask
            else:
                # Accept perturbation with some probability? 
                # Or just restart from best?
                # Standard ILS accepts if improved, or with prob exp(-delta/T).
                # Let's just keep the best for simplicity and speed.
                pass
                
    return list(best_global_set)

# Test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Score: {sum(1.0/(512-a) for a in res)}")
    # Verify pairwise coprime
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            if gcd(res[i], res[j]) != 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
