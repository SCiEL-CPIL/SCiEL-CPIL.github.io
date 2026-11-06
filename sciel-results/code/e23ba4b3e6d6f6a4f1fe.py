import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Precompute all valid 3-APs: for each d, list of (a, a+d, a+2d) where all in 1..512
    # For scoring, we just need to check if any a works for each d
    # Using bitset: bit i set means (i+1) is in set
    
    def score_bitset(b):
        count = 0
        for d in range(1, 256):
            if (b & (b >> d) & (b >> (2 * d))) != 0:
                count += 1
        return count

    def greedy_init():
        """Greedy construction: iteratively add element that maximizes new differences covered."""
        S = set()
        # Start with a reasonable seed
        random.seed(42)
        S.add(1)
        
        for _ in range(31):
            best_val = None
            best_new_diffs = -1
            current_score = 0
            # Compute current covered differences
            covered = set()
            bits = 0
            for x in S:
                bits |= (1 << (x - 1))
            for d in range(1, 256):
                if (bits & (bits >> d) & (bits >> (2 * d))) != 0:
                    covered.add(d)
            
            # Try each candidate
            candidates = list(range(1, 513))
            random.shuffle(candidates)
            # Limit candidates for speed
            candidates = candidates[:100]
            
            for val in candidates:
                if val in S:
                    continue
                new_bits = bits | (1 << (val - 1))
                new_covered = set()
                for d in range(1, 256):
                    if (new_bits & (new_bits >> d) & (new_bits >> (2 * d))) != 0:
                        new_covered.add(d)
                new_diffs = len(new_covered - covered)
                if new_diffs > best_new_diffs:
                    best_new_diffs = new_diffs
                    best_val = val
            
            if best_val is not None:
                S.add(best_val)
            else:
                # Fallback: add random
                S.add(random.randint(1, 512))
        
        return list(S)

    # Initialize with greedy, then try a few random starts
    best_set = None
    best_score = -1
    best_bit = 0
    
    # Try greedy init
    greedy_set = greedy_init()
    greedy_bit = 0
    for x in greedy_set:
        greedy_bit |= (1 << (x - 1))
    greedy_score = score_bitset(greedy_bit)
    best_set = greedy_set
    best_score = greedy_score
    best_bit = greedy_bit
    
    # Try a few random inits
    for _ in range(3):
        rand_set = random.sample(range(1, 513), 32)
        rand_bit = 0
        for x in rand_set:
            rand_bit |= (1 << (x - 1))
        rand_score = score_bitset(rand_bit)
        if rand_score > best_score:
            best_score = rand_score
            best_set = rand_set
            best_bit = rand_bit
    
    # Simulated Annealing
    current_set = list(best_set)
    current_bit = best_bit
    current_score = best_score
    
    T = 15.0
    T_min = 0.5
    cooling_rate = 0.99998
    
    while time.time() < deadline and T > T_min:
        # Generate neighbor: swap one element
        idx = random.randint(0, 31)
        old_val = current_set[idx]
        new_val = random.randint(1, 512)
        
        if new_val == old_val or new_val in current_set:
            # Try a few more times
            for _ in range(5):
                new_val = random.randint(1, 512)
                if new_val != old_val and new_val not in current_set:
                    break
            else:
                continue
        
        # Update bitset
        new_bit = current_bit
        new_bit &= ~(1 << (old_val - 1))
        new_bit |= (1 << (new_val - 1))
        
        new_score = score_bitset(new_bit)
        delta = new_score - current_score
        
        if delta > 0 or (delta >= -3 and random.random() < math.exp(delta / T)):
            # Accept
            current_set[idx] = new_val
            current_bit = new_bit
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                best_bit = current_bit
        
        T *= cooling_rate
        
        # Occasional restart from best if stuck
        if random.random() < 0.001:
            current_set = list(best_set)
            current_bit = best_bit
            current_score = best_score
            T = min(T * 1.5, 10.0)
    
    return best_set
