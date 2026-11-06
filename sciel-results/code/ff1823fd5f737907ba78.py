"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _score_set_bits(bits):
    """Score a set represented as a bitset (int) of 512 bits.
    Returns number of distinct d>0 such that exists a with a, a+d, a+2d in set.
    """
    # We need to count distinct d where (bits & (bits >> d) & (bits >> (2*d))) != 0
    # But careful: shifting a bitset by d means position i in result corresponds to
    # original positions i, i+d, i+2d all set.
    # Actually: if we have bits where bit i means element i is in set,
    # then (bits >> d) has bit i set iff bits has bit i+d set.
    # (bits >> (2*d)) has bit i set iff bits has bit i+2*d set.
    # So bits & (bits >> d) & (bits >> (2*d)) has bit i set iff i, i+d, i+2d all in set.
    # We just need to check if this AND is nonzero for each d.
    
    count = 0
    # d can range from 1 to 255 (since a+2d <= 512, and a>=1, so 2d <= 511, d <= 255)
    # Actually a >= 1, a + 2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
            count += 1
    return count


def _score_set_list(lst):
    """Score a set given as a list of integers."""
    bits = 0
    for x in lst:
        bits |= (1 << x)
    return _score_set_bits(bits)


def _greedy_polish(lst, max_iter=5000):
    """Local search: try adding one element and removing another to improve score."""
    current = set(lst)
    current_score = _score_set_list(list(current))
    
    for _ in range(max_iter):
        improved = False
        # Try all pairs (remove r, add a) where a not in current, r in current
        # To make this faster, we can be smarter, but let's try a limited version
        # Try removing each element and see if we can add something better
        for r in list(current):
            new_set = current - {r}
            # Try adding each possible a
            for a in range(1, 513):
                if a in new_set:
                    continue
                candidate = new_set | {a}
                cand_score = _score_set_list(list(candidate))
                if cand_score > current_score:
                    current = candidate
                    current_score = cand_score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return list(current), current_score


def _simulated_annealing(lst, time_limit):
    """Simulated annealing with bitset scoring."""
    start_time = time.time()
    current = set(lst)
    current_bits = 0
    for x in current:
        current_bits |= (1 << x)
    current_score = _score_set_bits(current_bits)
    
    best = set(current)
    best_bits = current_bits
    best_score = current_score
    
    temperature = 10.0
    cooling = 0.9995
    min_temp = 0.01
    
    # Precompute: for each element, its bit
    all_elements = set(range(1, 513))
    
    while time.time() - start_time < time_limit:
        if temperature < min_temp:
            break
        
        # Move: pick a random element to remove and a random element to add
        r = random.choice(list(current))
        a = random.choice(list(all_elements - current))
        
        new_set = (current - {r}) | {a}
        new_bits = current_bits & ~(1 << r) | (1 << a)
        new_score = _score_set_bits(new_bits)
        
        delta = new_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / temperature):
            current = new_set
            current_bits = new_bits
            current_score = new_score
            if current_score > best_score:
                best = set(current)
                best_bits = current_bits
                best_score = current_score
        
        temperature *= cooling
    
    return list(best), best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Start with a good initial set
    # Use a spread-out set to have many progression differences
    # Let's try a few random initializations and pick the best
    random.seed(42)
    
    best_set = list(range(1, 33))
    best_score = _score_set_list(best_set)
    
    # Try several random initializations
    for _ in range(5):
        if time.time() > deadline - 10:
            break
        rand_set = random.sample(range(1, 513), 32)
        s = _score_set_list(rand_set)
        if s > best_score:
            best_score = s
            best_set = rand_set
    
    # Greedy polish
    if time.time() < deadline - 5:
        polished, score = _greedy_polish(best_set, max_iter=2000)
        if score > best_score:
            best_set = polished
            best_score = score
    
    # Simulated annealing with remaining time
    remaining = deadline - time.time()
    if remaining > 1:
        sa_set, sa_score = _simulated_annealing(best_set, remaining - 1)
        if sa_score > best_score:
            best_set = sa_set
            best_score = sa_score
    
    # Final greedy polish
    remaining = deadline - time.time()
    if remaining > 1:
        polished, score = _greedy_polish(best_set, max_iter=1000)
        if score > best_score:
            best_set = polished
            best_score = score
    
    return best_set

# EVOLVE-BLOCK-END
