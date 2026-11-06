"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
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
        count = 0
        for d in range(1, 257):
            if (b & (b >> d) & (b >> (2 * d))):
                count += 1
        return count
    
    def popcount(b):
        return bin(b).count('1')
    
    def random_set():
        s = set(random.sample(range(1, 513), 32))
        return s
    
    # Strategy: Simulated annealing with focus on maximizing distinct difference count
    best_set = None
    best_score = -1
    
    # Precompute a "density" score for each number 1..512 based on how many short APs it participates in
    # A number x is "dense" if it participates in many 3-term APs with small d
    def compute_density():
        density = [0] * 513
        for d in range(1, 50):  # Focus on small differences
            for a in range(1, 513 - 2*d + 1):
                if a + 2*d <= 512:
                    density[a] += 1
                    density[a + d] += 1
                    density[a + 2*d] += 1
        return density
    
    density = compute_density()
    
    # Simulated annealing
    current_temp = 100.0
    cooling_rate = 0.995
    min_temp = 0.1
    
    # Start with a set biased towards dense numbers
    dense_numbers = sorted(range(1, 513), key=lambda x: -density[x])
    # Take top 64 dense numbers and sample 32 from them for initial set
    initial_pool = dense_numbers[:64]
    s = set(random.sample(initial_pool, 32))
    
    b = bitset_from_set(s)
    current_score = score_bitset(b)
    
    if current_score > best_score:
        best_score = current_score
        best_set = set(s)
    
    # Run simulated annealing
    iterations = 0
    while time.time() < deadline - 5.0 and current_temp > min_temp:
        iterations += 1
        
        # Generate a neighbor: swap one element with another
        removed = random.choice(list(s))
        # Bias the addition towards dense numbers
        if random.random() < 0.7:
            added = random.choice(dense_numbers[:100])
        else:
            added = random.randint(1, 512)
        
        if added == removed or added in s:
            continue
        
        new_s = set(s)
        new_s.remove(removed)
        new_s.add(added)
        
        if len(new_s) != 32:
            continue
        
        new_b = bitset_from_set(new_s)
        new_score = score_bitset(new_b)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/temp) if worse
        if delta > 0 or random.random() < (2.718281828 ** (-delta / current_temp)):
            s = new_s
            b = new_b
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(s)
        
        current_temp *= cooling_rate
    
    # Final local search to polish
    for _ in range(1000):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for removed in list(s):
            if time.time() > deadline - 5.0:
                break
            
            for added in random.sample(range(1, 513), 20):
                if added == removed or added in s:
                    continue
                
                new_s = set(s)
                new_s.remove(removed)
                new_s.add(added)
                
                if len(new_s) != 32:
                    continue
                
                new_b = bitset_from_set(new_s)
                new_score = score_bitset(new_b)
                
                if new_score > current_score:
                    s = new_s
                    b = new_b
                    current_score = new_score
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(s)
                    break
        
        if not improved:
            break
    
    # Final check
    if best_set is None or len(best_set) != 32:
        best_set = set(range(1, 33))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
