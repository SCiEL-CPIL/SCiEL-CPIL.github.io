import random
import time
import math

def _score_bitset(B):
    count = 0
    for d in range(1, 256):
        if (B & (B >> d) & (B >> (2 * d))) != 0:
            count += 1
    return count

def _to_bitset(S_list):
    B = 0
    for x in S_list:
        B |= (1 << (x - 1))
    return B

def _greedy_init():
    S_list = []
    S_set = set()
    B = 0
    for _ in range(32):
        best_c = -1
        best_score = -1
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        for c in candidates:
            if c in S_set:
                continue
            new_B = B | (1 << (c - 1))
            score = _score_bitset(new_B)
            if score > best_score:
                best_score = score
                best_c = c
        S_list.append(best_c)
        S_set.add(best_c)
        B |= (1 << (best_c - 1))
    return S_list, S_set

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = 0
    best_list, _ = _greedy_init()
    best_score = _score_bitset(_to_bitset(best_list))
    best_set = best_list
    
    while time.time() < deadline:
        if random.random() < 0.2:
            S_list, S_set = _greedy_init()
        else:
            S_list = random.sample(range(1, 513), 32)
            S_set = set(S_list)
            
        B = _to_bitset(S_list)
        current_score = _score_bitset(B)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(S_list)
            
        T = 5.0
        cooling = 0.99
        max_iter = 3000
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            x = random.choice(S_list)
            while True:
                y = random.randint(1, 512)
                if y not in S_set:
                    break
            
            S_list.remove(x)
            S_list.append(y)
            S_set.remove(x)
            S_set.add(y)
            
            new_B = _to_bitset(S_list)
            new_score = _score_bitset(new_B)
            
            delta = new_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_score = new_score
                B = new_B
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(S_list)
            else:
                S_list.remove(y)
                S_list.append(x)
                S_set.remove(y)
                S_set.add(x)
                B = _to_bitset(S_list)
            
            T *= cooling
            if T < 0.01:
                break

    return best_set
