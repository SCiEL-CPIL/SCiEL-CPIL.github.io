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
    # Max d is such that 2*d < 512, so d <= 255
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
    
    # Define different initialization strategies
    def init_random():
        return random.sample(range(1, MAXV + 1), N)
    
    def init_contiguous():
        start = random.randint(1, MAXV - N + 1)
        return list(range(start, start + N))
        
    def init_clustered():
        # Pick a few centers and add neighbors
        centers = random.sample(range(1, MAXV + 1), 4)
        S = set()
        for c in centers:
            for i in range(-8, 9):
                val = c + i
                if 1 <= val <= MAXV:
                    S.add(val)
        if len(S) < N:
            # Fill rest randomly
            remaining = list(set(range(1, MAXV + 1)) - S)
            S.update(random.sample(remaining, N - len(S)))
        return list(S)
        
    def init_ap():
        # Arithmetic progression
        a = random.randint(1, 100)
        d = random.randint(1, 10)
        S = set()
        x = a
        while len(S) < N and x <= MAXV:
            S.add(x)
            x += d
        if len(S) < N:
            remaining = list(set(range(1, MAXV + 1)) - S)
            S.update(random.sample(remaining, N - len(S)))
        return list(S)

    initializers = [init_random, init_contiguous, init_clustered, init_ap]
    
    while time.time() < deadline:
        # Choose an initializer
        init_func = random.choice(initializers)
        current_list = init_func()
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        # Limit local search time
        local_deadline = start_time + min(5.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Move type
            move_type = random.random()
            
            if move_type < 0.7:
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
                
            elif move_type < 0.9:
                # 2-swap
                idxs = random.sample(range(N), 2)
                x1 = current_list[idxs[0]]
                x2 = current_list[idxs[1]]
                
                y1 = y2 = 0
                while y1 == 0 or y1 == y2 or ((current_bits >> y1) & 1):
                    y1 = random.randint(1, MAXV)
                while y2 == 0 or y1 == y2 or ((current_bits >> y2) & 1):
                    y2 = random.randint(1, MAXV)
                
                new_bits = current_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                
                if idxs[0] > idxs[1]:
                    idxs[0], idxs[1] = idxs[1], idxs[0]
                
                new_list = current_list[:idxs[0]] + current_list[idxs[0]+1:idxs[1]] + current_list[idxs[1]+1:]
                bisect.insort(new_list, y1)
                bisect.insort(new_list, y2)
            else:
                # Shift move: Shift all elements by a small amount
                # This preserves structure
                shift = random.randint(-5, 5)
                if shift == 0:
                    continue
                new_list = []
                valid = True
                for x in current_list:
                    y = x + shift
                    if 1 <= y <= MAXV:
                        new_list.append(y)
                    else:
                        valid = False
                        break
                if not valid or len(new_list) != N:
                    # If shift moves elements out of range, try to wrap or skip
                    # For simplicity, skip this move if invalid
                    continue
                
                new_bits = 0
                for x in new_list:
                    new_bits |= (1 << x)
                
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
