import random
import time
from itertools import combinations

def is_sidon_fast(subset):
    """Check if a list of ints is a Sidon set (all pair sums i<=j unique)."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    """Find the size of the largest Sidon subset of S."""
    S = sorted(S)
    n = len(S)
    # Try from largest k downwards
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon_fast(list(subset)):
                return k
    return 2

def construct():
    # We want to minimize the largest Sidon subset size.
    # Start with a random set
    random.seed(42)
    current_set = random.sample(range(129), 12)
    current_score = largest_sidon_subset_size(current_set)
    
    best_set = current_set[:]
    best_score = current_score
    
    # Local search: try to swap elements to reduce the score
    # If we find a set with score < best_score, update.
    # We run this for a fixed number of steps or time.
    
    start_time = time.time()
    timeout = 50 # Leave 10s margin
    
    # To make search more effective, we can try to "break" Sidon subsets.
    # A simple strategy: Pick a random pair (i, j) and swap S[i] with a random value 
    # or swap S[i] and S[j].
    # Swapping S[i] and S[j] doesn't change the set, so that's useless.
    # We need to change the *values*.
    # Strategy: Pick index i, pick a new random value v in [0,128] not in S.
    # Replace S[i] with v.
    
    iterations = 0
    while time.time() - start_time < timeout:
        # Pick a random index to mutate
        i = random.randint(0, 11)
        old_val = current_set[i]
        
        # Pick a new value
        # To keep it distinct, we can pick from a pre-computed list or just retry
        new_val = random.randint(0, 128)
        if new_val in current_set:
            continue
            
        # Apply mutation
        current_set[i] = new_val
        
        # Evaluate
        score = largest_sidon_subset_size(current_set)
        
        # Accept if better, or with some probability (SA)
        # For simplicity, let's do strict hill climbing first, 
        # but if stuck, we might need to accept worse moves.
        # Given the small space, strict hill climbing might get stuck in local minima.
        # Let's add a simple random restart if no improvement for 1000 steps.
        
        if score < current_score:
            current_score = score
            if score < best_score:
                best_score = score
                best_set = current_set[:]
            if best_score == 2:
                break # Can't get better than 2
        else:
            # Revert if worse or equal (Strict Hill Climbing)
            # Or accept with prob exp(-delta/T)
            # Let's try a simple SA approach
            if score > current_score:
                # Accept with probability
                # Delta is score difference
                # Lower score is better.
                # If score > current_score, it's worse.
                # Let's just revert for now to keep it simple and fast.
                # Actually, random restarts are better.
                current_set[i] = old_val
                iterations += 1
                if iterations % 1000 == 0:
                    # Random restart
                    current_set = random.sample(range(129), 12)
                    current_score = largest_sidon_subset_size(current_set)
                    if current_score < best_score:
                        best_score = current_score
                        best_set = current_set[:]
                    iterations = 0
            else:
                # Equal score, keep it? Or revert?
                # Keeping it allows drifting.
                pass
                
    return best_set
