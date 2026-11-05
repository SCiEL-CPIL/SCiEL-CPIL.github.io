import time
import random
from itertools import combinations

def is_sidon_4(subset):
    """Check if a subset of 4 distinct integers is Sidon."""
    # Sums: a+a, a+b, a+c, a+d, b+b, b+c, b+d, c+c, c+d, d+d
    # 10 sums. Must be distinct.
    sums = []
    for i in range(4):
        for j in range(i, 4):
            sums.append(subset[i] + subset[j])
    return len(set(sums)) == 10

def objective(S):
    """
    Returns a score to be minimized.
    0: No Sidon 4-subset, and No Sidon 3-subset (Max Sidon Size 2)
    1: No Sidon 4-subset, but Has Sidon 3-subset (Max Sidon Size 3)
    >1: Number of Sidon 4-subsets (Max Sidon Size >= 4)
    """
    # Count Sidon 4-subsets
    count_4 = 0
    for subset in combinations(S, 4):
        if is_sidon_4(subset):
            count_4 += 1
            # Early exit if we just want to know if it's > 0?
            # But for optimization, knowing the count helps.
            # However, 495 checks is fast.
            
    if count_4 > 0:
        return count_4
    
    # Check for Sidon 3
    # A 3-subset is Sidon iff it is NOT a 3-term AP.
    for subset in combinations(S, 3):
        a, b, c = subset
        # Check if any is the average of the other two
        if a + c == 2 * b or a + b == 2 * c or b + c == 2 * a:
            continue # This 3-subset is NOT Sidon
        else:
            # Found a Sidon 3-subset
            return 1
            
    # No Sidon 3-subset found
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin for 60s limit
    
    best_set = None
    best_score = 13 # Max possible Sidon size is 12, so 13 is infinity
    
    random.seed(42)
    
    # Initial population
    population = []
    for _ in range(50):
        s = random.sample(range(129), 12)
        population.append(s)
        
    # Evaluate initial population
    for s in population:
        score = objective(s)
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score == 0:
                break # Found size 2, best possible
                
    if best_score == 0:
        return best_set
        
    # Local Search
    # We iterate, picking a random individual, and trying to improve it.
    iterations = 0
    max_iterations = 5000
    
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        
        # Pick a candidate: 80% from population, 20% new random
        if random.random() < 0.8 and population:
            s = list(random.choice(population))
        else:
            s = random.sample(range(129), 12)
            
        current_score = objective(s)
        
        # Try to improve by swapping one element
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            old_val = s[i]
            # Try a few random replacements
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val in s:
                    continue
                s[i] = new_val
                new_score = objective(s)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    break
                else:
                    s[i] = old_val
            
            if improved:
                break # Restart improvement for this individual with new state
                
        if improved:
            # Add improved solution to population
            population.append(list(s))
            if len(population) > 200:
                # Keep population size manageable, remove worst
                # Simple: just keep last 200
                population = population[-200:]
                
            if current_score < best_score:
                best_score = current_score
                best_set = list(s)
                if best_score == 0:
                    break
                elif best_score == 1:
                    # Found size 3. This is likely the best we can get.
                    # We can continue searching for 0 (size 2) for a while,
                    # but if it's hard, 1 is a great result.
                    pass
        else:
            # No improvement found for this individual
            pass

    return best_set
