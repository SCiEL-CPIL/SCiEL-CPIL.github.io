import random
import itertools

def is_sidon(subset):
    sums = set()
    for i, a in enumerate(subset):
        for j in range(i, len(subset)):
            b = subset[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    # Check size 5
    for subset in itertools.combinations(S, 5):
        if is_sidon(subset):
            return 5
    # Check size 4
    for subset in itertools.combinations(S, 4):
        if is_sidon(subset):
            return 4
    return 3

def construct():
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Try a few random perturbations to see if we can find a better set
    # (i.e., lower max_sidon_size)
    # Since best_score is likely 4, we look for 3.
    # If we find 3, we return it.
    
    for _ in range(100):
        # Start with AP and perturb
        S = list(range(12))
        # Randomly move 1-2 elements
        for _ in range(random.randint(1, 3)):
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val not in S:
                S[idx] = new_val
                break # only one move per iteration for simplicity
        
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S
            if best_score == 3:
                break
                
    return best_set
