import math
import time
import random

def _score(U):
    U = list(set(U))
    if 0 not in U:
        return 0.0
    mx = max(U)
    if mx == 0:
        return 0.0
    plus = set()
    minus = set()
    for a in U:
        for b in U:
            plus.add(a + b)
            minus.add(a - b)
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def _score_fast(mask, n, max_val):
    """Compute score for a set represented as a bitmask of elements 1..max_val.
    mask is a Python int where bit i is set if (i+1) is in the set.
    Always includes 0.
    """
    if mask == 0:
        return 0.0  # Only {0}, invalid
    
    # Extract elements
    elements = [0]
    for i in range(n):
        if (mask >> i) & 1:
            elements.append(i + 1)
    
    mx = elements[-1]
    if mx == 0:
        return 0.0
    
    # Compute sumset and difference set using sets
    plus = set()
    minus = set()
    for a in elements:
        for b in elements:
            plus.add(a + b)
            minus.add(a - b)
    
    n_plus = len(plus)
    n_minus = len(minus)
    if n_plus == 0:
        return 0.0
    ratio = n_minus / n_plus
    if ratio <= 1:
        return 1.0
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = random.Random(42)
    
    best_score = 0.0
    best_mask = 0
    
    # Phase 1: Exhaustive search for very small sets to find good seeds
    # Search over subsets of {1, 2, ..., 10} of size 1 to 5
    import itertools
    
    for size in range(1, 6):
        for combo in itertools.combinations(range(1, 11), size):
            if time.time() > deadline:
                break
            mask = 0
            for x in combo:
                mask |= (1 << (x - 1))
            s = _score_fast(mask, 10, 10)
            if s > best_score:
                best_score = s
                best_mask = mask
        if time.time() > deadline:
            break
    
    # Phase 2: Simulated annealing over larger search space
    # We search over subsets of {1, 2, ..., 30} with size up to 12
    # Represent as bitmask (30 bits)
    
    N_ELEMENTS = 30  # Search over {1, ..., 30}
    MAX_SIZE = 12    # Maximum number of elements (excluding 0)
    
    # Start from best found in Phase 1, or a random good seed
    current_mask = best_mask
    if current_mask == 0:
        # Random seed
        current_mask = 0
        for x in rng.sample(range(1, N_ELEMENTS + 1), 4):
            current_mask |= (1 << (x - 1))
    
    current_score = _score_fast(current_mask, N_ELEMENTS, N_ELEMENTS)
    
    # Simulated annealing parameters
    T0 = 0.05  # Initial temperature
    T_min = 0.001
    alpha = 0.995  # Cooling rate
    T = T0
    
    iteration = 0
    while time.time() < deadline:
        # Generate neighbor
        # Randomly flip 1-3 bits
        n_flips = rng.randint(1, 3)
        new_mask = current_mask
        for _ in range(n_flips):
            pos = rng.randint(0, N_ELEMENTS - 1)
            new_mask ^= (1 << pos)
        
        # Check size constraint
        size = bin(new_mask).count('1')
        if size > MAX_SIZE or size == 0:
            continue
        
        new_score = _score_fast(new_mask, N_ELEMENTS, N_ELEMENTS)
        
        # Accept or reject
        if new_score > current_score:
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = current_mask
        else:
            delta = new_score - current_score
            if delta < 0:
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    current_mask = new_mask
                    current_score = new_score
        
        # Cool down
        T = max(T * alpha, T_min)
        iteration += 1
        
        # Occasionally restart from a different seed if stuck
        if iteration % 5000 == 0:
            # Try a few random restarts
            for _ in range(5):
                restart_mask = 0
                restart_size = rng.randint(3, 8)
                for x in rng.sample(range(1, N_ELEMENTS + 1), restart_size):
                    restart_mask |= (1 << (x - 1))
                restart_score = _score_fast(restart_mask, N_ELEMENTS, N_ELEMENTS)
                if restart_score > current_score:
                    current_mask = restart_mask
                    current_score = restart_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
    
    # Phase 3: Local search refinement from best found
    if time.time() < deadline - 10:
        current_mask = best_mask
        current_score = best_score
        
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Try adding an element
            if bin(current_mask).count('1') < MAX_SIZE:
                pos = rng.randint(0, N_ELEMENTS - 1)
                if not (current_mask & (1 << pos)):
                    new_mask = current_mask | (1 << pos)
                    new_score = _score_fast(new_mask, N_ELEMENTS, N_ELEMENTS)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
            
            # Try removing an element
            elif bin(current_mask).count('1') > 1:
                set_bits = [i for i in range(N_ELEMENTS) if (current_mask >> i) & 1]
                pos = rng.choice(set_bits)
                new_mask = current_mask & ~(1 << pos)
                new_score = _score_fast(new_mask, N_ELEMENTS, N_ELEMENTS)
                if new_score > current_score:
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
            
            # Try swapping
            if bin(current_mask).count('1') > 1:
                set_bits = [i for i in range(N_ELEMENTS) if (current_mask >> i) & 1]
                unset_bits = [i for i in range(N_ELEMENTS) if not (current_mask >> i) & 1]
                if set_bits and unset_bits:
                    remove_pos = rng.choice(set_bits)
                    add_pos = rng.choice(unset_bits)
                    new_mask = (current_mask & ~(1 << remove_pos)) | (1 << add_pos)
                    new_score = _score_fast(new_mask, N_ELEMENTS, N_ELEMENTS)
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_mask = current_mask
    
    # Convert best_mask to list
    best_set = [0]
    for i in range(N_ELEMENTS):
        if (best_mask >> i) & 1:
            best_set.append(i + 1)
    
    best_set = sorted(best_set)
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set = sorted([0] + best_set)
    
    return best_set
