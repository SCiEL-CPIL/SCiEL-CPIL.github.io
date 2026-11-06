import time
import random
import math

def _score_set_bitset(s):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d all in s.
    Uses Python integers as bitsets for fast checking.
    """
    s_set = set(s)
    if not s_set:
        return 0
    max_val = max(s_set)
    bitset = 0
    for x in s_set:
        bitset |= (1 << x)
    
    diffs_found = set()
    # d can be at most (max_val - 1) // 2
    # Actually, if a=1, a+2d <= max_val => 2d <= max_val - 1
    for d in range(1, (max_val - 1) // 2 + 1):
        if (bitset & (bitset >> d) & (bitset >> (2 * d))) != 0:
            diffs_found.add(d)
    
    return len(diffs_found)

def _simulate_annealing(seed, deadline):
    random.seed(seed)
    
    # Initial states to try
    initial_sets = [
        list(range(1, 33)),  # Dense interval
        list(range(1, 17)) + list(range(257, 273)), # Two blocks
        list(range(1, 9)) + list(range(129, 137)) + list(range(257, 265)) + list(range(385, 393)), # Four blocks
        [random.randint(1, 512) for _ in range(32)] # Random
    ]
    
    best_set = None
    best_score = -1
    
    for init_set in initial_sets:
        # Ensure distinct and valid
        init_set = list(set(init_set))
        while len(init_set) < 32:
            x = random.randint(1, 512)
            if x not in init_set:
                init_set.append(x)
        
        current_set = set(init_set)
        current_score = _score_set_bitset(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        start_time = time.time()
        
        while time.time() < deadline and T > T_min:
            # Generate a move
            # 50% chance 1-swap, 50% chance 2-swap
            if random.random() < 0.5:
                # 1-swap
                if len(current_set) > 0:
                    to_remove = random.choice(list(current_set))
                    current_set.remove(to_remove)
                    new_elem = random.randint(1, 512)
                    while new_elem in current_set:
                        new_elem = random.randint(1, 512)
                    current_set.add(new_elem)
                else:
                    current_set.add(random.randint(1, 512))
            else:
                # 2-swap
                if len(current_set) >= 2:
                    items = list(current_set)
                    i, j = random.sample(range(len(items)), 2)
                    to_remove1 = items[i]
                    to_remove2 = items[j]
                    current_set.remove(to_remove1)
                    current_set.remove(to_remove2)
                    
                    new_elem1 = random.randint(1, 512)
                    while new_elem1 in current_set:
                        new_elem1 = random.randint(1, 512)
                    current_set.add(new_elem1)
                    
                    new_elem2 = random.randint(1, 512)
                    while new_elem2 in current_set:
                        new_elem2 = random.randint(1, 512)
                    current_set.add(new_elem2)
                else:
                    # Add 2
                    for _ in range(2):
                        x = random.randint(1, 512)
                        while x in current_set:
                            x = random.randint(1, 512)
                        current_set.add(x)
            
            new_score = _score_set_bitset(current_set)
            
            # Acceptance criterion
            if new_score >= current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_set = list(current_set)
            else:
                # Metropolis criterion
                delta = current_score - new_score
                if random.random() < math.exp(-delta / T):
                    current_score = new_score
            
            T *= cooling
            
            # Check time
            if time.time() > deadline:
                break
    
    return best_set

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Run SA with a few different seeds to diversify search
    # We have ~145s. Let's split it into 3 runs of ~45s each.
    time_per_run = (deadline - (time.time() - start_time)) / 3.0
    
    best_overall_set = None
    best_overall_score = -1
    
    for i in range(3):
        run_deadline = time.time() + time_per_run
        # Add a small buffer to avoid exceeding the main deadline
        if run_deadline > deadline:
            run_deadline = deadline
            
        result_set = _simulate_annealing(42 + i, run_deadline)
        
        if result_set:
            score = _score_set_bitset(result_set)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_set = result_set
        
        # If we are running out of time, break
        if time.time() > deadline - 5:
            break
    
    if best_overall_set is None:
        # Fallback
        return list(range(1, 33))
        
    return best_overall_set
