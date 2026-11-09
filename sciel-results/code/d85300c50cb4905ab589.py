import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    N = 32
    MAX_VAL = 512
    MAX_D = 255  # max d such that 1 + 2d <= 512 => d <= 255.5

    def time_ok():
        return time.time() < deadline

    def score_mask(mask):
        count = 0
        m = mask
        for d in range(1, MAX_D + 1):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << (x - 1))
        return m

    def mask_to_list(mask):
        return [i + 1 for i in range(MAX_VAL) if (mask >> i) & 1]

    def delta_replace_single(mask, old_val, new_val):
        old_bit = 1 << (old_val - 1)
        new_bit = 1 << (new_val - 1)
        new_mask = (mask & ~old_bit) | new_bit
        
        affected_ds = set()
        
        v = old_val
        max_d_1 = (MAX_VAL - v) // 2
        for d in range(1, min(max_d_1, MAX_D) + 1):
            affected_ds.add(d)
        max_d_2 = min(v - 1, MAX_VAL - v)
        for d in range(1, max_d_2 + 1):
            affected_ds.add(d)
        max_d_3 = (v - 1) // 2
        for d in range(1, min(max_d_3, MAX_D) + 1):
            affected_ds.add(d)

        v = new_val
        max_d_1 = (MAX_VAL - v) // 2
        for d in range(1, min(max_d_1, MAX_D) + 1):
            affected_ds.add(d)
        max_d_2 = min(v - 1, MAX_VAL - v)
        for d in range(1, max_d_2 + 1):
            affected_ds.add(d)
        max_d_3 = (v - 1) // 2
        for d in range(1, min(max_d_3, MAX_D) + 1):
            affected_ds.add(d)

        delta = 0
        m_old = mask
        m_new = new_mask
        
        for d in affected_ds:
            old_has = (m_old & (m_old >> d) & (m_old >> (2 * d))) != 0
            new_has = (m_new & (m_new >> d) & (m_new >> (2 * d))) != 0
            if old_has != new_has:
                delta += 1 if new_has else -1
                
        return delta, new_mask

    def greedy_construct():
        mask = 0
        s_set = set()
        covered_ds = set()
        
        for _ in range(N):
            best_new = None
            best_gain = -1
            best_new_mask = 0
            best_new_ds = set()
            
            candidates = random.sample(range(1, MAX_VAL + 1), 100)
            
            for v in candidates:
                if v in s_set:
                    continue
                
                new_mask = mask | (1 << (v - 1))
                new_ds = set()
                for d in range(1, MAX_D + 1):
                    if (new_mask & (new_mask >> d) & (new_mask >> (2 * d))) != 0:
                        new_ds.add(d)
                
                gain = len(new_ds - covered_ds)
                
                if gain > best_gain:
                    best_gain = gain
                    best_new = v
                    best_new_mask = new_mask
                    best_new_ds = new_ds
            
            if best_new is None:
                available = [v for v in range(1, MAX_VAL + 1) if v not in s_set]
                best_new = random.choice(available)
                best_new_mask = mask | (1 << (best_new - 1))
                best_new_ds = set()
                for d in range(1, MAX_D + 1):
                    if (best_new_mask & (best_new_mask >> d) & (best_new_mask >> (2 * d))) != 0:
                        best_new_ds.add(d)

            mask = best_new_mask
            s_set.add(best_new)
            covered_ds = best_new_ds
            
        return mask, score_mask(mask), mask_to_list(mask)

    def simulated_annealing(init_mask, init_score, init_list, max_time_per_run=10.0):
        import math
        current_mask = init_mask
        current_score = init_score
        current_list = init_list[:]
        current_set = set(current_list)
        
        best_mask = current_mask
        best_score = current_score
        best_list = current_list[:]
        
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.01
        
        start_sa_time = time.time()
        
        while time_ok() and (time.time() - start_sa_time) < max_time_per_run:
            if T < min_T:
                break
                
            if random.random() < 0.7:
                idx = random.randint(0, N - 1)
                old_val = current_list[idx]
                
                new_val = random.randint(1, MAX_VAL)
                if new_val == old_val or new_val in current_set:
                    continue
                
                delta, new_mask = delta_replace_single(current_mask, old_val, new_val)
                new_score = current_score + delta
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    current_set.discard(old_val)
                    current_set.add(new_val)
                    current_list[idx] = new_val
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        best_list = current_list[:]
                        
            else:
                idxs = random.sample(range(N), 2)
                old_val1 = current_list[idxs[0]]
                old_val2 = current_list[idxs[1]]
                
                new_val1 = random.randint(1, MAX_VAL)
                new_val2 = random.randint(1, MAX_VAL)
                
                if new_val1 == new_val2:
                    continue
                if new_val1 in current_set or new_val2 in current_set:
                    continue
                if new_val1 == old_val1 or new_val1 == old_val2 or new_val2 == old_val1 or new_val2 == old_val2:
                    continue
                
                new_mask = current_mask
                new_mask &= ~(1 << (old_val1 - 1))
                new_mask &= ~(1 << (old_val2 - 1))
                new_mask |= (1 << (new_val1 - 1))
                new_mask |= (1 << (new_val2 - 1))
                
                new_score = score_mask(new_mask)
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_score = new_score
                    current_set.discard(old_val1)
                    current_set.discard(old_val2)
                    current_set.add(new_val1)
                    current_set.add(new_val2)
                    current_list[idxs[0]] = new_val1
                    current_list[idxs[1]] = new_val2
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = current_mask
                        best_list = current_list[:]
            
            T *= cooling_rate
            
        return best_mask, best_score, best_list

    random.seed(42)
    
    global_best_mask = 0
    global_best_score = 0
    global_best_list = []
    
    num_restarts = 5
    time_per_restart = 140.0 / num_restarts
    
    for i in range(num_restarts):
        if not time_ok():
            break
            
        if i == 0:
            mask, score, lst = greedy_construct()
        else:
            lst = random.sample(range(1, MAX_VAL + 1), N)
            mask = list_to_mask(lst)
            score = score_mask(mask)
        
        m, s, l = simulated_annealing(mask, score, lst, max_time_per_run=time_per_restart)
        
        if s > global_best_score:
            global_best_score = s
            global_best_mask = m
            global_best_list = l[:]
            
        print(f"Restart {i}: Score {s}", file=sys.stderr)
        
    print(f"Final Score: {global_best_score}", file=sys.stderr)
    
    if not global_best_list:
        return list(range(1, 33))
        
    return global_best_list
