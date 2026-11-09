"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _get_sparsity(pairs):
    """Compute the number of nonzero terms in the square of the polynomial defined by pairs.
    pairs: list of (exponent, coefficient) tuples."""
    if not pairs:
        return 0
    max_deg = max(e for e, c in pairs if c != 0)
    arr = np.zeros(2 * max_deg + 1, dtype=int)
    for e, c in pairs:
        if c != 0:
            arr[e] += c
    sq = np.convolve(arr, arr, mode='full')
    return int(np.count_nonzero(sq))

def construct():
    start_time = time.time()
    time_limit = 55.0
    best_sparsity = float('inf')
    best_result = None
    
    # Simulated Annealing parameters
    T_start = 100.0
    T_end = 0.1
    cooling_rate = 0.9995
    max_iterations = 50000
    
    def get_sparsity_and_arr(pairs):
        """Helper to get sparsity and the square array for faster evaluation if needed, 
        but here we just need sparsity count."""
        return _get_sparsity(pairs)

    def generate_initial():
        """
        Generate a promising initial state.
        Consecutive exponents create many sum collisions.
        """
        # Start with a block of consecutive exponents
        start_exp = random.randint(0, 25)
        exponents = [start_exp + i for i in range(12)]
        # Randomize signs to encourage cancellation
        coefficients = [random.choice([-1, 1]) for _ in range(12)]
        pairs = list(zip(exponents, coefficients))
        return pairs

    def neighbor(pairs, T):
        """
        Generate a neighbor.
        Moves:
        1. Shift one exponent by -1, 0, or 1 (if valid and not colliding).
        2. Swap two exponents.
        3. Flip sign of a coefficient.
        4. Change coefficient magnitude (1, 2, -1, -2).
        """
        new_pairs = [p for p in pairs]
        move_type = random.random()
        
        if move_type < 0.4:
            # Shift exponent
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            delta = random.choice([-2, -1, 1, 2])
            new_e = e + delta
            # Check bounds and collision
            if 0 <= new_e < 41:
                used = set(p[0] for p in new_pairs)
                if new_e not in used:
                    new_pairs[i] = (new_e, c)
                    new_pairs.sort(key=lambda x: x[0])
                else:
                    # If collision, try a random unused neighbor
                    available = [x for x in range(max(0, new_e-3), min(41, new_e+4)) if x not in used and x != e]
                    if available:
                        new_e = random.choice(available)
                        new_pairs[i] = (new_e, c)
                        new_pairs.sort(key=lambda x: x[0])
        elif move_type < 0.7:
            # Swap exponents
            i, j = random.sample(range(12), 2)
            e1, c1 = new_pairs[i]
            e2, c2 = new_pairs[j]
            new_pairs[i] = (e2, c1)
            new_pairs[j] = (e1, c2)
            new_pairs.sort(key=lambda x: x[0])
        elif move_type < 0.9:
            # Flip sign
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            new_pairs[i] = (e, -c)
        else:
            # Change coefficient magnitude
            i = random.randint(0, 11)
            e, c = new_pairs[i]
            new_c = random.choice([-2, -1, 1, 2])
            new_pairs[i] = (e, new_c)
            
        return new_pairs

    # Run Simulated Annealing
    T = T_start
    current = generate_initial()
    current_sparsity = _get_sparsity(current)
    
    if current_sparsity < best_sparsity:
        best_sparsity = current_sparsity
        best_result = current[:]

    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        next_p = neighbor(current, T)
        next_sparsity = _get_sparsity(next_p)
        
        delta = next_sparsity - current_sparsity
        
        if delta < 0:
            current = next_p
            current_sparsity = next_sparsity
            if current_sparsity < best_sparsity:
                best_sparsity = current_sparsity
                best_result = current[:]
                if best_sparsity == 12:
                    # Theoretical minimum reached (12 diagonal terms, all cross terms cancel/merge)
                    # We can break early if we are confident, but let's run a bit more to be sure or just break.
                    # Actually, 12 is the absolute minimum for 12 distinct exponents.
                    break
        else:
            # Accept worse solution with probability exp(-delta/T)
            if T > 0:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current = next_p
                    current_sparsity = next_sparsity
                    
        T *= cooling_rate
        if T < T_end:
            T = T_end

    # If SA didn't find a good one, do a few local search restarts on the best found
    if best_sparsity > 15: # Threshold for "good enough"
        for _ in range(20):
            if time.time() - start_time > time_limit:
                break
            # Perturb best_result
            current = best_result[:]
            # Randomly change 2-3 exponents
            for _ in range(3):
                idx = random.randint(0, 11)
                e, c = current[idx]
                new_e = random.randint(0, 40)
                # Ensure no collision
                attempts = 0
                while new_e in [p[0] for p in current if p[0] != e] and attempts < 10:
                    new_e = random.randint(0, 40)
                    attempts += 1
                if new_e not in [p[0] for p in current if p[0] != e]:
                    current[idx] = (new_e, c)
            current.sort(key=lambda x: x[0])
            
            # Local search on this perturbed state
            for _ in range(200):
                if time.time() - start_time > time_limit:
                    break
                next_p = neighbor(current, 1.0) # High temp for local search? No, just greedy
                next_sparsity = _get_sparsity(next_p)
                if next_sparsity <= current_sparsity:
                    current = next_p
                    current_sparsity = next_sparsity
                    if current_sparsity < best_sparsity:
                        best_sparsity = current_sparsity
                        best_result = current[:]
                if current_sparsity == 12:
                    break
            if best_sparsity == 12:
                break

    if best_result is None:
        exponents = list(range(12))
        signs = [1] * 12
        best_result = list(zip(exponents, signs))

    # Convert best_result (list of tuples) to coefficient list
    max_deg = max(e for e, c in best_result)
    result = [0] * (max_deg + 1)
    for e, c in best_result:
        result[e] = c
        
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
        
    # Ensure nonzero final coefficient
    if result[-1] == 0:
        last_nonzero = -1
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                last_nonzero = i
                break
        if last_nonzero != -1:
            result = result[:last_nonzero + 1]
        if result[-1] == 0:
            result[-1] = 1
            
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in result if c != 0)
    if nonzero_count != 12:
        result = [1] * 12
        
    return result

# EVOLVE-BLOCK-END
