import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    # Max possible d is (512-1)//2 = 255
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Helper to update bits from list
    def get_bits(lst):
        b = 0
        for x in lst:
            b |= (1 << x)
        return b

    # Seeds
    seeds = []
    # Seed 1: Interval [1..32]
    seeds.append(list(range(1, 33)))
    
    # Seed 2: Interval [1..31] + [33]
    seeds.append(list(range(1, 32)) + [33])
    
    # Seed 3: Random
    seeds.append(random.sample(range(1, MAXV + 1), N))
    
    # Seed 4: Two intervals
    seeds.append(list(range(1, 17)) + list(range(256, 272)))
    
    # Seed 5: Spread out
    seeds.append([i * 16 + 1 for i in range(32)]) # 1, 17, 33...
    
    for seed in seeds:
        if time.time() > deadline:
            break
            
        current_list = seed[:]
        current_list.sort()
        current_bits = get_bits(current_list)
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.9995
        local_deadline = start_time + min(20.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Pick x to remove
            x_idx = random.randint(0, N - 1)
            x = current_list[x_idx]
            
            # Pick y to add
            # Bias towards values that might help?
            # For now, random in 1..512
            y = random.randint(1, MAXV)
            if y == x:
                continue
            if (current_bits >> y) & 1:
                continue
            
            # Update bits
            new_bits = current_bits ^ (1 << x) ^ (1 << y)
            
            # Update list
            new_list = current_list[:x_idx] + current_list[x_idx+1:]
            ins_idx = bisect.bisect_left(new_list, y)
            new_list.insert(ins_idx, y)
            
            new_score = _score(new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
