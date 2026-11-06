import time
import math
import random
import bisect
import itertools

N = 32
MAX_VAL = 512
MAX_D = 256

def _score_mask(mask):
    count = 0
    for d in range(1, MAX_D + 1):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _list_to_mask(lst):
    mask = 0
    for v in lst:
        mask |= (1 << (v - 1))
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = -1
    best_list = list(range(1, 33)) # Fallback
    best_mask = _list_to_mask(best_list)
    best_score = _score_mask(best_mask)
    
    rng = random.Random(123)
    
    # Heuristic move generation: find values that complete a 3-term progression
    def get_heuristic_candidates(current_set, current_mask, x, rng):
        # 10% random
        if rng.random() < 0.1:
            y = rng.randint(1, MAX_VAL)
            if y != x and y not in current_set:
                return y
            return None
        
        # Try to complete progressions
        candidates = []
        # For every pair (u, v) in S \ {x}, the third term of the AP could be:
        # 2v - u (if u < v) -> u, v, 2v-u
        # 2u - v (if v < u) -> v, u, 2u-v
        # (u+v)//2 (if u+v even) -> u, (u+v)//2, v
        
        # To be efficient, we don't iterate all pairs. We pick a few random ones.
        # But we can also check specific structures.
        # Let's pick 5 random pairs.
        s_without_x = [v for v in current_set if v != x]
        n = len(s_without_x)
        if n < 2:
            return None
            
        for _ in range(5):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 2)
            if j >= i:
                j += 1
            u = s_without_x[i]
            v = s_without_x[j]
            
            c1 = 2 * v - u
            if 1 <= c1 <= MAX_VAL and c1 != x and c1 not in current_set:
                candidates.append(c1)
                
            c2 = 2 * u - v
            if 1 <= c2 <= MAX_VAL and c2 != x and c2 not in current_set:
                candidates.append(c2)
                
            if (u + v) % 2 == 0:
                c3 = (u + v) // 2
                if 1 <= c3 <= MAX_VAL and c3 != x and c3 not in current_set:
                    candidates.append(c3)
                    
        if candidates:
            return rng.choice(candidates)
        
        # Fallback random
        y = rng.randint(1, MAX_VAL)
        if y != x and y not in current_set:
            return y
        return None

    # Multi-start Simulated Annealing
    # We use several different initializations
    
    inits = [
        list(range(1, 33)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256] + list(range(257, 290))[:23],
        [i * 16 + 1 for i in range(32)],
        [i * 16 + 1 for i in range(16)] + [i * 16 + 9 for i in range(16)], # Two shifted intervals
        sorted(random.sample(range(1, MAX_VAL + 1), N)),
        sorted(random.sample(range(1, MAX_VAL + 1), N)),
    ]
    
    # Add a "greedy" construction start
    # Start with empty, add elements that maximize score gain
    greedy_set = set()
    greedy_list = []
    # Simple greedy: pick 1, 2, ... then optimize? 
    # Let's just use the random ones for now.
    
    for init in inits:
        if time.time() > deadline:
            break
            
        current_list = sorted(init[:])
        if len(current_list) != N or any(x < 1 or x > MAX_VAL for x in current_list):
            current_list = sorted(random.sample(range(1, MAX_VAL + 1), N))
            
        current_set = set(current_list)
        current_mask = _list_to_mask(current_list)
        current_score = _score_mask(current_mask)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
            best_mask = current_mask
            
        # SA Parameters
        T = 20.0
        T_min = 0.5
        # Aim for ~5000 steps? 
        # Each step is fast. 
        # Let's allocate ~15s per run.
        local_deadline = start_time + 15.0
        # Actually, we have 145s total. 6 runs * 15s = 90s. Leaves 55s buffer.
        # Let's do 4 runs of 25s each? Or just loop until deadline.
        
        # Adjust cooling to reach T_min in ~5000 steps
        target_steps = 5000
        cooling = (T_min / T) ** (1.0 / target_steps)
        
        while time.time() < local_deadline:
            x_idx = rng.randint(0, N - 1)
            x = current_list[x_idx]
            
            y = get_heuristic_candidates(current_set, current_mask, x, rng)
            if y is None:
                continue
                
            new_mask = current_mask
            new_mask ^= (1 << (x - 1))
            new_mask ^= (1 << (y - 1))
            
            new_score = _score_mask(new_mask)
            delta = new_score - current_score
            
            if delta > 0 or (T > T_min and rng.random() < math.exp(delta / T)):
                # Accept
                current_score = new_score
                current_mask = new_mask
                current_set.remove(x)
                current_set.add(y)
                # Update list
                current_list.pop(x_idx)
                ins_idx = bisect.bisect_left(current_list, y)
                current_list.insert(ins_idx, y)
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
                    best_mask = current_mask
                    
            T *= cooling
            
    return sorted(best_list)

construct()
