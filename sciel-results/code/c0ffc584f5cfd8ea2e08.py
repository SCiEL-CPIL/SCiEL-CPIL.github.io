"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import product

_start_time = time.time()
_DEADLINE = _start_time + 135.0  # safety margin before 150s kill

def _score_of(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    if not U:
        return 0.0
    U_set = set(U)
    n = len(U_set)
    max_u = max(U_set)
    if max_u == 0:
        return 1.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U_set:
        for b in U_set:
            sumset.add(a + b)
    sumset_size = len(sumset)
    
    # Compute difference set U-U
    diffset = set()
    for a in U_set:
        for b in U_set:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / log_denom

def _score_of_fast(U_list):
    """Faster scoring using numpy for larger sets."""
    import numpy as np
    U_arr = np.array(U_list, dtype=np.int64)
    n = len(U_arr)
    max_u = int(U_arr.max())
    if max_u == 0:
        return 1.0
    
    # Sumset
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Difference set
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return 0.0
    
    return 1.0 + math.log(ratio) / log_denom

def _generate_base_B_set(B, k):
    """Generate set {sum of k powers of B} - actually just powers of B plus 0."""
    # Simple: {0, B, B^2, ..., B^(k-1)}
    U = [0]
    val = 1
    for i in range(k):
        U.append(val)
        val *= B
    return U

def _generate_structured_set(B, digits_per_pos, num_positions):
    """
    Generate a set where each position in base-B representation has a small set of allowed digits.
    This gives structured sparsity.
    """
    # For each position, pick a subset of digits
    # Start with single digit per position (powers of B), then allow more
    U = [0]
    for pos in range(num_positions):
        base_val = B ** pos
        # Allow digit 0 and digit 1 at this position (binary-like)
        new_U = []
        for u in U:
            new_U.append(u)
            new_U.append(u + base_val)
        U = new_U
    return U

def _genetic_search(initial_U, B, num_positions, max_time=120.0):
    """
    Genetic algorithm that mutates the base-B digit representation.
    Each individual is a set of nonnegative integers.
    """
    deadline = _start_time + max_time
    
    # Start with the initial set
    best_U = list(initial_U)
    best_score = _score_of_fast(best_U)
    
    # Population: list of sets
    # Represent each individual as a sorted list of integers
    pop_size = 20
    population = [best_U]
    
    # Initialize population with variations
    for _ in range(pop_size - 1):
        var = _mutate_set(best_U, B, num_positions)
        population.append(var)
    
    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        
        # Evaluate population
        scored = []
        for ind in population:
            try:
                s = _score_of_fast(ind)
            except:
                s = 0.0
            scored.append((s, ind))
        
        scored.sort(key=lambda x: -x[0])
        best_score = scored[0][0]
        best_U = scored[0][1]
        
        # Selection: keep top half
        elite_count = max(2, pop_size // 4)
        elites = [ind for _, ind in scored[:elite_count]]
        
        # Create new population
        new_pop = list(elites)
        
        while len(new_pop) < pop_size:
            # Crossover and mutation
            if len(elites) >= 2:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)
                child = _crossover_sets(parent1, parent2, B, num_positions)
            else:
                child = list(best_U)
            
            child = _mutate_set(child, B, num_positions)
            new_pop.append(child)
        
        population = new_pop
    
    return best_U, best_score

def _mutate_set(U, B, num_positions, mutation_rate=0.3):
    """Mutate a set by adding/removing elements or perturbing digits."""
    U = list(U)
    if not U:
        U = [0]
    
    # Ensure 0 is in the set
    if 0 not in U:
        U.append(0)
    
    max_u = max(U) if U else 0
    U_set = set(U)
    
    # Random mutations
    for _ in range(int(mutation_rate * len(U))):
        op = random.random()
        if op < 0.3:
            # Add a new element
            if max_u > 0:
                new_val = random.randint(0, max_u)
            else:
                new_val = random.randint(1, B)
            U_set.add(new_val)
        elif op < 0.5:
            # Remove a random element (not 0)
            removable = [x for x in U_set if x != 0]
            if removable:
                U_set.discard(random.choice(removable))
        elif op < 0.7:
            # Perturb a random element
            if U_set:
                val = random.choice(list(U_set))
                if val != 0:
                    delta = random.randint(-B, B)
                    new_val = max(0, val + delta)
                    U_set.discard(val)
                    U_set.add(new_val)
        else:
            # Add a power-of-B-like element
            if num_positions > 0:
                pos = random.randint(0, num_positions - 1)
                digit = random.randint(1, min(B - 1, 3))
                new_val = digit * (B ** pos)
                U_set.add(new_val)
    
    # Ensure 0 is present
    U_set.add(0)
    return sorted(U_set)

def _crossover_sets(U1, U2, B, num_positions):
    """Crossover two sets."""
    U1_set = set(U1)
    U2_set = set(U2)
    
    # Combine and take random subset
    combined = U1_set | U2_set
    if not combined:
        return [0]
    
    # Keep 0 and random subset of others
    non_zero = [x for x in combined if x != 0]
    keep_count = random.randint(max(1, len(non_zero) // 2), min(len(non_zero), len(non_zero) + 3))
    kept = random.sample(non_zero, min(keep_count, len(non_zero)))
    
    result = {0} | set(kept)
    return sorted(result)

def _try_construction(B, num_positions, max_time=110.0):
    """Try a construction with given parameters."""
    # Start with binary-like set: {sum of subset of {B^0, B^1, ..., B^(k-1)}}
    # That's all numbers from 0 to B^k - 1 with digits 0 or 1
    initial_U = _generate_structured_set(B, 1, num_positions)
    
    if len(initial_U) > 20000:
        # Too large, reduce positions
        num_positions = max(1, num_positions - 2)
        initial_U = _generate_structured_set(B, 1, num_positions)
    
    best_U, best_score = _genetic_search(initial_U, B, num_positions, max_time)
    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()
    
    best_U = [0, 1, 3]
    best_score = _score_of(best_U)
    
    # Try different base and position combinations
    configs = [
        (3, 6),   # B=3, 6 positions -> up to 2^6=64 elements
        (3, 8),   # B=3, 8 positions -> up to 256 elements
        (4, 6),   # B=4, 6 positions
        (5, 5),   # B=5, 5 positions
        (3, 10),  # B=3, 10 positions -> up to 1024 elements
    ]
    
    for B, num_pos in configs:
        if time.time() > _start_time + 100.0:
            break
        
        try:
            U, score = _try_construction(B, num_pos, max_time=100.0 - (time.time() - _start_time))
            if score > best_score:
                best_score = score
                best_U = U
        except Exception:
            continue
    
    # Final check
    if time.time() > _start_time + 130.0:
        return best_U
    
    # Try a few more random constructions if time permits
    random.seed(42)
    for _ in range(5):
        if time.time() > _start_time + 135.0:
            break
        B = random.randint(2, 8)
        num_pos = random.randint(3, 8)
        try:
            U, score = _try_construction(B, num_pos, max_time=135.0 - (time.time() - _start_time))
            if score > best_score:
                best_score = score
                best_U = U
        except Exception:
            continue
    
    return best_U
# EVOLVE-BLOCK-END
