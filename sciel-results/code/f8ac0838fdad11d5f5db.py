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
    dense_numbers = sorted(range(1, 513), key=lambda x: -density[x])
    
    best_set = None
    best_score = -1
    
    def update_best(s, b, score):
        nonlocal best_set, best_score
        if score > best_score:
            best_score = score
            best_set = set(s)
    
    # Helper to run a local search (Hill Climbing) from a given state
    def local_search(s, b, score, max_iters=5000):
        current_s = set(s)
        current_b = b
        current_score = score
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 2.0:
                break
            
            # Try swapping each element with a random candidate
            # To be faster, we don't check all 512 candidates for each of 32 elements (32*512=16k checks)
            # But we can check a subset or use a smarter move.
            # Let's try swapping one element with a random one from the full universe, 
            # but prioritize dense numbers.
            
            # Randomize the order of elements to swap
            elements = list(current_s)
            random.shuffle(elements)
            
            for removed in elements:
                if time.time() > deadline - 2.0:
                    break
                
                # Try a few random additions
                # Mix of dense and random
                candidates = []
                # 5 random from dense top 100
                candidates += random.sample(dense_numbers[:100], 5)
                # 5 random from full range
                candidates += random.sample(range(1, 513), 5)
                
                for added in candidates:
                    if added == removed or added in current_s:
                        continue
                    
                    new_b = (current_b & ~(1 << (removed - 1))) | (1 << (added - 1))
                    new_score = score_bitset(new_b)
                    
                    if new_score > current_score:
                        current_s.remove(removed)
                        current_s.add(added)
                        current_b = new_b
                        current_score = new_score
                        improved = True
                        break # Restart element loop with new state
                if improved:
                    break # Restart element loop with new state
            
        return current_s, current_b, current_score

    # Multi-start strategy
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate diverse initial states
        if r == 0:
            # Pure random
            s = set(random.sample(range(1, 513), 32))
        elif r == 1:
            # Top dense
            s = set(dense_numbers[:32])
        elif r == 2:
            # Mixed dense/random
            s = set(random.sample(dense_numbers[:64], 16) + random.sample(range(1, 513), 16))
        elif r % 2 == 0:
            # Random
            s = set(random.sample(range(1, 513), 32))
        else:
            # Structured: Take blocks
            # e.g., 4 blocks of 8 numbers with gaps
            base = random.randint(1, 50)
            gap = random.randint(20, 40)
            s = set()
            for i in range(4):
                start = base + i * gap
                for j in range(8):
                    val = start + j
                    if 1 <= val <= 512:
                        s.add(val)
            # Fill remaining if needed
            while len(s) < 32:
                s.add(random.randint(1, 512))
            s = set(list(s)[:32])
        
        b = bitset_from_set(s)
        score = score_bitset(b)
        update_best(s, b, score)
        
        # Run Local Search (Hill Climbing)
        s, b, score = local_search(s, b, score)
        update_best(s, b, score)
        
        # Optional: Short SA run if time permits and we haven't reached high score
        if time.time() < deadline - 20.0 and best_score < 50:
             # Simple SA step to escape local optimum
             current_temp = 10.0
             cooling_rate = 0.99
             min_temp = 0.1
             sa_s = set(s)
             sa_b = b
             sa_score = score
             
             while time.time() < deadline - 15.0 and current_temp > min_temp:
                 removed = random.choice(list(sa_s))
                 if random.random() < 0.5:
                     added = random.choice(dense_numbers[:100])
                 else:
                     added = random.randint(1, 512)
                 
                 if added == removed or added in sa_s:
                     continue
                 
                 new_b = (sa_b & ~(1 << (removed - 1))) | (1 << (added - 1))
                 new_score = score_bitset(new_b)
                 delta = new_score - sa_score
                 
                 if delta > 0 or random.random() < (2.718281828 ** (-delta / current_temp)):
                     sa_s.remove(removed)
                     sa_s.add(added)
                     sa_b = new_b
                     sa_score = new_score
                     update_best(sa_s, sa_b, sa_score)
                 
                 current_temp *= cooling_rate
             
             # Polish the SA result
             s, b, score = local_search(sa_s, sa_b, sa_score, max_iters=2000)
             update_best(s, b, score)

    # Final check
    if best_set is None or len(best_set) != 32:
        best_set = set(range(1, 33))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
