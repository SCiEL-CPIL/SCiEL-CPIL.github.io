"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_mask(mask):
    """
    Count distinct d > 0 such that there exists a with a, a+d, a+2d all in set.
    mask is a 512-bit integer representing the set (bit i set means i+1 in set, i.e., value = i+1).
    Actually let's use bit i for value i (1-indexed), so bit 1..512. But Python ints are fine.
    Let's use bit i for value i, where i ranges 1..512.
    """
    # For each d from 1 to 255 (since 2d + a <= 512, and a >= 1, so 2d <= 511 => d <= 255)
    # Check if there exists a such that bits a, a+d, a+2d are all set.
    # This is equivalent to: (mask & (mask >> d) & (mask >> (2*d))) != 0
    # But we need to be careful with the bit positions.
    # Let's define mask such that bit k is set if value k is in the set (k from 1 to 512).
    # Then for a given d, we need a, a+d, a+2d all in [1, 512].
    # The condition (mask & (mask >> d) & (mask >> (2*d))) != 0 checks if there is some bit position
    # where all three are set. But shifting right by d moves bit k to k-d.
    # So mask >> d has bit k set if mask had bit k+d set.
    # So mask & (mask >> d) has bit k set if mask had bits k and k+d set.
    # Then (mask & (mask >> d) & (mask >> (2*d))) has bit k set if mask had bits k, k+d, k+2d set.
    # So if this is non-zero, there exists a = k such that a, a+d, a+2d are all in the set.
    # We need k >= 1 and k+2d <= 512. But the bit operations naturally handle this because
    # bits beyond 512 are 0, and bits below 1 don't exist (or are 0 in our representation).
    # Actually, in Python, shifting right discards low bits. So mask >> d will have bits 1..512-d.
    # The AND will only have bits where all three are set, which implies k >= 1 and k+2d <= 512.
    # So this is correct.
    
    count = 0
    # d can range from 1 to 255
    for d in range(1, 256):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count


def _score_set(s):
    """Score a set of integers (1..512)."""
    mask = 0
    for x in s:
        mask |= (1 << x)
    return _score_mask(mask)


def _greedy_init():
    """
    Greedy construction: start with empty set, add elements that maximize score increment.
    """
    s = set()
    mask = 0
    best_score = 0
    candidates = list(range(1, 513))
    random.shuffle(candidates)
    
    for _ in range(32):
        best_elem = None
        best_new_score = best_score
        for c in candidates:
            if c in s:
                continue
            # Try adding c
            new_mask = mask | (1 << c)
            new_score = _score_mask(new_mask)
            if new_score > best_new_score:
                best_new_score = new_score
                best_elem = c
        if best_elem is not None:
            s.add(best_elem)
            mask |= (1 << best_elem)
            best_score = best_new_score
        else:
            # Fallback: add a random candidate
            if candidates:
                c = candidates.pop()
                s.add(c)
                mask |= (1 << c)
                best_score = _score_mask(mask)
    return s, best_score


def _simulated_annealing(s, initial_score, deadline):
    """
    Simulated annealing to improve the set.
    """
    current = set(s)
    current_score = initial_score
    best = set(current)
    best_score = current_score
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    # Number of steps: we'll run until deadline
    alpha = 0.9995  # cooling rate per step
    
    T = T_start
    while T > T_end:
        if time.time() > deadline:
            break
        
        # Generate a neighbor: remove a random element, add a random new one
        if len(current) != 32:
            break
        
        # Pick a random element to remove
        to_remove = random.choice(list(current))
        # Pick a random element to add (not in current)
        # To speed up, we can sample from 1..512
        to_add = random.randint(1, 512)
        if to_add in current:
            continue
        
        # Compute new score
        new_set = set(current)
        new_set.remove(to_remove)
        new_set.add(to_add)
        new_score = _score_set(new_set)
        
        # Accept if better, or with probability exp((new_score - current_score) / T)
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best = set(current)
                best_score = current_score
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if random.random() < prob:
                    current = new_set
                    current_score = new_score
        
        T *= alpha
    
    return best, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe within 150s limit
    
    # Seed random for reproducibility? No, let's use a fixed seed for determinism in evaluation?
    # Actually, the task doesn't require determinism, but let's use a fixed seed for consistent results.
    random.seed(42)
    np.random.seed(42)
    
    # Phase 1: Greedy initialization
    s, score = _greedy_init()
    best_s = set(s)
    best_score = score
    
    # Phase 2: Simulated annealing
    best_s, best_score = _simulated_annealing(best_s, best_score, deadline)
    
    # Phase 3: Local search (hill climbing) on the best solution
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current = set(best_s)
        current_score = best_score
        
        # Try all single swaps
        for a in list(current):
            if time.time() > deadline:
                break
            for b in range(1, 513):
                if b in current:
                    continue
                new_set = set(current)
                new_set.remove(a)
                new_set.add(b)
                new_score = _score_set(new_set)
                if new_score > current_score:
                    current = new_set
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            best_s = current
            best_score = current_score
    
    return sorted(list(best_s))

# EVOLVE-BLOCK-END
