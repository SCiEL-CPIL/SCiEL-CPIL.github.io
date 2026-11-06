"""Optimized search for 32 integers in 1..512 maximizing distinct 3-AP differences."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Bitset representation: x in 1..512 maps to bit (x-1)
    # S is an integer where bit i is 1 if (i+1) is in the set
    
    def bitset_from_set(S):
        b = 0
        for x in S:
            b |= (1 << (x - 1))
        return b
    
    def set_from_bitset(b):
        s = []
        i = 0
        while b:
            if b & 1:
                s.append(i + 1)
            b >>= 1
            i += 1
        return s
    
    def score_bitset(b):
        """Count distinct d > 0 such that a, a+d, a+2d all in set for some a."""
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    # Precompute: for each element x in 1..512, count how many 3-APs it participates in
    # This helps with greedy initialization
    def compute_density():
        """Compute density score for each position 1..512."""
        density = [0] * 513
        for d in range(1, 256):
            for a in range(1, 512 - 2 * d + 1):
                density[a] += 1
                density[a + d] += 1
                density[a + 2 * d] += 1
        return density
    
    def greedy_initialize(rng):
        """Greedy initialization: pick elements that maximize 3-AP coverage."""
        density = compute_density()
        # Start with a random seed
        current_set = set()
        current_bit = 0
        current_score = 0
        
        # Add 32 elements greedily
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        # First few: pick high-density positions
        for _ in range(32):
            if not candidates:
                break
            best_elem = None
            best_gain = -1
            # Sample a subset for speed
            sample_size = min(50, len(candidates))
            sample = rng.sample(candidates, sample_size)
            for elem in sample:
                if elem in current_set:
                    continue
                new_bit = current_bit | (1 << (elem - 1))
                new_score = score_bitset(new_bit)
                gain = new_score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_elem = elem
            if best_elem is not None:
                current_set.add(best_elem)
                current_bit |= (1 << (best_elem - 1))
                current_score += best_gain
                candidates.remove(best_elem)
            else:
                # Fallback: pick random
                elem = candidates.pop()
                current_set.add(elem)
                current_bit |= (1 << (elem - 1))
                current_score = score_bitset(current_bit)
        
        return list(current_set), current_bit, current_score
    
    def simulated_annealing(initial_set, initial_bit, initial_score, rng, time_budget):
        """Run simulated annealing starting from given state."""
        current_set = list(initial_set)
        current_bit = initial_bit
        current_score = initial_score
        
        best_set = list(current_set)
        best_bit = current_bit
        best_score = current_score
        
        temperature = 30.0
        cooling = 0.99998
        min_temp = 0.01
        e = 2.718281828459045
        
        while temperature > min_temp:
            if time.time() > time_budget:
                break
            
            move_type = rng.random()
            
            if move_type < 0.6:
                # Swap: replace one element with another
                i = rng.randrange(32)
                old_elem = current_set[i]
                new_elem = rng.randint(1, 512)
                if new_elem == old_elem or new_elem in current_set:
                    continue
                
                # Quick check: would this improve?
                new_bit = (current_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new_elem
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            elif move_type < 0.8:
                # Double swap: replace two elements with two others
                i = rng.randrange(32)
                j = rng.randrange(32)
                if i == j:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                new1 = rng.randint(1, 512)
                new2 = rng.randint(1, 512)
                if new1 in current_set or new2 in current_set or new1 == new2:
                    continue
                
                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            else:
                # Triple swap: replace three elements
                i = rng.randrange(32)
                j = rng.randrange(32)
                k = rng.randrange(32)
                if i == j or i == k or j == k:
                    continue
                old1 = current_set[i]
                old2 = current_set[j]
                old3 = current_set[k]
                new1 = rng.randint(1, 512)
                new2 = rng.randint(1, 512)
                new3 = rng.randint(1, 512)
                if new1 in current_set or new2 in current_set or new3 in current_set:
                    continue
                if len({new1, new2, new3}) < 3:
                    continue
                
                new_bit = current_bit
                new_bit &= ~(1 << (old1 - 1))
                new_bit &= ~(1 << (old2 - 1))
                new_bit &= ~(1 << (old3 - 1))
                new_bit |= (1 << (new1 - 1))
                new_bit |= (1 << (new2 - 1))
                new_bit |= (1 << (new3 - 1))
                new_score = score_bitset(new_bit)
                
                delta = new_score - current_score
                if delta > 0 or rng.random() < (e ** (delta / temperature)):
                    current_set[i] = new1
                    current_set[j] = new2
                    current_set[k] = new3
                    current_bit = new_bit
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                        best_bit = current_bit
            
            temperature *= cooling
        
        return best_set, best_bit, best_score
    
    # Multi-restart strategy with different seeds
    global_best_set = None
    global_best_bit = 0
    global_best_score = -1
    
    # Use multiple restarts with different initializations
    restarts = 5
    time_per_restart = (deadline - time.time()) / restarts
    
    for restart in range(restarts):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(1000 + restart * 7919)
        
        # Greedy initialization
        init_set, init_bit, init_score = greedy_initialize(rng)
        
        # Simulated annealing
        sa_time_budget = min(deadline - 5.0, time.time() + time_per_restart)
        best_set, best_bit, best_score = simulated_annealing(
            init_set, init_bit, init_score, rng, sa_time_budget
        )
        
        if best_score > global_best_score:
            global_best_score = best_score
            global_best_set = best_set
            global_best_bit = best_bit
        
        # Print progress (for debugging)
        # print(f"Restart {restart}: score {best_score}")
    
    # Final polish: local search on the best solution
    if global_best_set is not None and time.time() < deadline - 10.0:
        polish_rng = random.Random(99999)
        polish_set = list(global_best_set)
        polish_bit = global_best_bit
        polish_score = global_best_score
        
        # Hill climbing with small perturbations
        for _ in range(5000):
            if time.time() > deadline - 2.0:
                break
            
            i = polish_rng.randrange(32)
            old_elem = polish_set[i]
            new_elem = polish_rng.randint(1, 512)
            if new_elem == old_elem or new_elem in polish_set:
                continue
            
            new_bit = (polish_bit & ~(1 << (old_elem - 1))) | (1 << (new_elem - 1))
            new_score = score_bitset(new_bit)
            
            if new_score > polish_score:
                polish_set[i] = new_elem
                polish_bit = new_bit
                polish_score = new_score
                if polish_score > global_best_score:
                    global_best_score = polish_score
                    global_best_set = list(polish_set)
                    global_best_bit = polish_bit
        
        # Try double swaps for polish
        for _ in range(3000):
            if time.time() > deadline - 1.0:
                break
            
            i = polish_rng.randrange(32)
            j = polish_rng.randrange(32)
            if i == j:
                continue
            old1 = polish_set[i]
            old2 = polish_set[j]
            new1 = polish_rng.randint(1, 512)
            new2 = polish_rng.randint(1, 512)
            if new1 in polish_set or new2 in polish_set or new1 == new2:
                continue
            
            new_bit = polish_bit
            new_bit &= ~(1 << (old1 - 1))
            new_bit &= ~(1 << (old2 - 1))
            new_bit |= (1 << (new1 - 1))
            new_bit |= (1 << (new2 - 1))
            new_score = score_bitset(new_bit)
            
            if new_score > polish_score:
                polish_set[i] = new1
                polish_set[j] = new2
                polish_bit = new_bit
                polish_score = new_score
                if polish_score > global_best_score:
                    global_best_score = polish_score
                    global_best_set = list(polish_set)
                    global_best_bit = polish_bit
    
    return global_best_set if global_best_set is not None else list(range(1, 33))
