import time
import random
from itertools import combinations

def is_sidon_with_diagonals(subset):
    """
    Check if a subset of integers has all unordered pair sums (including diagonals) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        ai = subset[i]
        for j in range(i, k):
            s = ai + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """
    Find the size of the largest subset of candidate_set that is Sidon with diagonals.
    candidate_set is a list of 12 distinct integers.
    """
    n = len(candidate_set)
    # We check from largest k down to 1
    # Optimization: If we find a Sidon subset of size k, we return k.
    # We can prune if the number of possible sums is less than k*(k+1)/2?
    # No, sums depend on values.
    
    for k in range(n, 0, -1):
        found = False
        for comb in combinations(candidate_set, k):
            if is_sidon_with_diagonals(comb):
                found = True
                break
        if found:
            return k
    return 0

def collision_density(candidate_set):
    """
    Count the number of additive quadruple conflicts.
    Used as a heuristic for tie-breaking or intermediate fitness.
    """
    sums = {}
    for i in range(len(candidate_set)):
        for j in range(i, len(candidate_set)):
            s = candidate_set[i] + candidate_set[j]
            if s not in sums:
                sums[s] = 0
            sums[s] += 1
    
    conflicts = 0
    for count in sums.values():
        if count > 1:
            conflicts += count * (count - 1) // 2
    return conflicts

def construct():
    start_time = time.time()
    # Evaluator kills after 90s. We aim for 85s.
    deadline = 85.0
    
    # The key insight from the prompt's knowledge:
    # "Restricting the search space to a small integer range [0, 25] forces sum collisions..."
    # We will focus heavily on small ranges.
    
    best_set = None
    best_score = 13 # Max possible is 12
    
    # 1. Try specific dense candidates
    # Consecutive integers are very dense.
    for start in range(0, 20):
        cand = [start + i for i in range(12)]
        if max(cand) > 128:
            break
        if len(set(cand)) != 12:
            continue
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            if best_score <= 3:
                return best_set

    # 2. Random search in small ranges
    # Ranges: [0, 12], [0, 15], [0, 20], [0, 30]
    ranges = [12, 13, 14, 15, 16, 18, 20, 25, 30]
    
    # Prioritize smaller ranges
    for R in ranges:
        if time.time() > deadline:
            break
            
        # Number of iterations per range, decreasing for larger ranges
        if R <= 15:
            iterations = 5000
        elif R <= 25:
            iterations = 2000
        else:
            iterations = 500
            
        for _ in range(iterations):
            if time.time() > deadline:
                break
                
            # Generate a random set of 12 distinct integers in [0, R]
            # Using sample is fast
            try:
                cand = random.sample(range(R + 1), 12)
            except ValueError:
                continue
                
            cand.sort()
            
            # Quick check: if the range is very small, collisions are high.
            # Calculate score
            score = max_sidon_size(cand)
            
            if score < best_score:
                best_score = score
                best_set = cand[:]
                if best_score <= 3:
                    return best_set
            elif score == best_score:
                # Tie-breaking: prefer higher collision density (more collisions = harder to find large Sidon subsets? 
                # Actually, if score is same, density doesn't change the max Sidon size, but might indicate robustness)
                # We can just keep the first one found or the one with higher density.
                pass

    # 3. Local Search / Annealing around the best found set
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
        
    current_set = best_set[:]
    current_score = best_score
    
    # If we already found a very small score, we can stop early
    if current_score <= 3:
        return current_set
        
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        new_set = current_set[:]
        
        # Perturb 1 or 2 elements
        num_replaces = random.choice([1, 1, 2])
        indices = random.sample(range(12), num_replaces)
        
        for idx in indices:
            # Try to move the element to a nearby value or a random value in a small range
            if random.random() < 0.5:
                # Small step
                step = random.randint(-3, 3)
                new_val = current_set[idx] + step
            else:
                # Jump to small range
                new_val = random.randint(0, 30)
            
            # Ensure validity
            attempts = 0
            while (new_val in new_set or new_val < 0 or new_val > 128) and attempts < 10:
                new_val = random.randint(0, 30)
                attempts += 1
            
            if new_val in new_set or new_val < 0 or new_val > 128:
                continue # Skip this perturbation if we can't fix it easily
            
            new_set[idx] = new_val
        
        if len(set(new_set)) != 12:
            continue
            
        new_set.sort()
        new_score = max_sidon_size(new_set)
        
        if new_score < current_score:
            current_set = new_set[:]
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = new_set[:]
                if best_score <= 3:
                    break
        elif new_score == current_score:
            # Accept with some probability to escape local optima
            if random.random() < 0.5:
                current_set = new_set[:]
        else:
            # Reject worse moves (or accept with very low prob)
            pass
            
        temperature *= cooling_rate
        if temperature < 0.05:
            temperature = 0.05

    return best_set
