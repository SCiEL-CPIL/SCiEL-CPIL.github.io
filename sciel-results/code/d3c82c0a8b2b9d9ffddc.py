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
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    while time.time() < deadline:
        # Initialize random set
        current_list = random.sample(range(1, MAXV + 1), N)
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        local_deadline = start_time + min(5.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Choose move type: 1-swap (80%) or 2-swap (20%)
            move_type = random.random()
            
            if move_type < 0.8:
                # 1-swap
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                while True:
                    y = random.randint(1, MAXV)
                    if not ((current_bits >> y) & 1):
                        break
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, y)
                new_list.insert(ins_idx, y)
                
            else:
                # 2-swap
                # Pick two distinct indices to remove
                idxs = random.sample(range(N), 2)
                x1 = current_list[idxs[0]]
                x2 = current_list[idxs[1]]
                
                # Pick two distinct new values not in set
                y1 = y2 = 0
                while y1 == 0 or y1 == y2 or ((current_bits >> y1) & 1):
                    y1 = random.randint(1, MAXV)
                while y2 == 0 or y1 == y2 or ((current_bits >> y2) & 1):
                    y2 = random.randint(1, MAXV)
                
                new_bits = current_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                
                # Update list
                # Remove x1 and x2
                # To avoid index shifting issues, sort the indices
                if idxs[0] > idxs[1]:
                    idxs[0], idxs[1] = idxs[1], idxs[0]
                
                new_list = current_list[:idxs[0]] + current_list[idxs[0]+1:idxs[1]] + current_list[idxs[1]+1:]
                
                # Insert y1 and y2
                bisect.insort(new_list, y1)
                bisect.insort(new_list, y2)
            
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
