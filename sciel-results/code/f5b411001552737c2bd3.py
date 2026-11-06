import random
import time
import math

def construct():
    n = 256
    deadline = time.time() + 80
    
    def count_3aps(S):
        count = 0
        for a in range(1, n + 1):
            if not S[a]:
                continue
            max_d = (n - a) // 2
            for d in range(1, max_d + 1):
                if S[a + d] and S[a + 2 * d]:
                    count += 1
        return count
    
    def has_4ap_with(S, x):
        # x as 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if S[x + d] and S[x + 2 * d] and S[x + 3 * d]:
                return True
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, x):
            if x - d >= 1 and S[x - d] and S[x + d] and S[x + 2 * d]:
                return True
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, (x - 1) // 2 + 1):
            if S[x - 2 * d] and S[x - d] and S[x + d]:
                return True
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            if S[x - 3 * d] and S[x - 2 * d] and S[x - d]:
                return True
        return False
    
    def delta_3aps_add(S, x):
        delta = 0
        for d in range(1, (n - x) // 2 + 1):
            if S[x + d] and S[x + 2 * d]:
                delta += 1
        for d in range(1, x):
            if x - d >= 1 and S[x - d] and S[x + d]:
                delta += 1
        for d in range(1, (x - 1) // 2 + 1):
            if S[x - 2 * d] and S[x - d]:
                delta += 1
        return delta
    
    def delta_3aps_remove(S, x):
        delta = 0
        for d in range(1, (n - x) // 2 + 1):
            if S[x + d] and S[x + 2 * d]:
                delta -= 1
        for d in range(1, x):
            if x - d >= 1 and S[x - d] and S[x + d]:
                delta -= 1
        for d in range(1, (x - 1) // 2 + 1):
            if S[x - 2 * d] and S[x - d]:
                delta -= 1
        return delta
    
    def delta_3aps_swap(S, x_out, x_in):
        loss = 0
        for d in range(1, (n - x_out) // 2 + 1):
            if S[x_out + d] and S[x_out + 2 * d]:
                loss += 1
        for d in range(1, x_out):
            if x_out - d >= 1 and S[x_out - d] and S[x_out + d]:
                loss += 1
        for d in range(1, (x_out - 1) // 2 + 1):
            if S[x_out - 2 * d] and S[x_out - d]:
                loss += 1
        
        gain = 0
        for d in range(1, (n - x_in) // 2 + 1):
            a1 = x_in + d
            a2 = x_in + 2 * d
            if a1 != x_out and S[a1] and a2 != x_out and S[a2]:
                gain += 1
        for d in range(1, x_in):
            a1 = x_in - d
            a2 = x_in + d
            if a1 != x_out and S[a1] and a2 != x_out and S[a2]:
                gain += 1
        for d in range(1, (x_in - 1) // 2 + 1):
            a1 = x_in - 2 * d
            a2 = x_in - d
            if a1 != x_out and S[a1] and a2 != x_out and S[a2]:
                gain += 1
        
        return gain - loss
    
    def valid_swap(S, x_out, x_in):
        def in_set_no_x(val, x_out):
            if val == x_out:
                return False
            return S[val]
        
        for d in range(1, (n - x_in) // 3 + 1):
            if in_set_no_x(x_in + d, x_out) and in_set_no_x(x_in + 2 * d, x_out) and in_set_no_x(x_in + 3 * d, x_out):
                return False
        for d in range(1, x_in):
            if x_in - d >= 1 and in_set_no_x(x_in - d, x_out) and in_set_no_x(x_in + d, x_out) and in_set_no_x(x_in + 2 * d, x_out):
                return False
        for d in range(1, (x_in - 1) // 2 + 1):
            if in_set_no_x(x_in - 2 * d, x_out) and in_set_no_x(x_in - d, x_out) and in_set_no_x(x_in + d, x_out):
                return False
        for d in range(1, (x_in - 1) // 3 + 1):
            if in_set_no_x(x_in - 3 * d, x_out) and in_set_no_x(x_in - 2 * d, x_out) and in_set_no_x(x_in - d, x_out):
                return False
        return True
    
    def greedy_construct():
        S = [False] * (n + 1)
        for x in range(1, n + 1):
            if not has_4ap_with(S, x):
                S[x] = True
        return S
    
    def random_greedy():
        S = [False] * (n + 1)
        order = list(range(1, n + 1))
        random.shuffle(order)
        for x in order:
            if not has_4ap_with(S, x):
                S[x] = True
        return S
    
    def local_search(S, max_iters=8000):
        current = S[:]
        current_score = count_3aps(current)
        best = S[:]
        best_score = current_score
        
        temp = 1.0
        cooling = 0.9998
        min_temp = 0.0005
        iters = 0
        
        while iters < max_iters and time.time() < deadline:
            iters += 1
            move_type = random.random()
            accepted = False
            
            if move_type < 0.25:
                candidates = [x for x in range(1, n + 1) if not current[x]]
                if candidates:
                    x = random.choice(candidates)
                    if not has_4ap_with(current, x):
                        delta = delta_3aps_add(current, x)
                        if delta >= 0 or random.random() < math.exp(delta / temp):
                            current[x] = True
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best = current[:]
            elif move_type < 0.5:
                in_set = [x for x in range(1, n + 1) if current[x]]
                if in_set:
                    x = random.choice(in_set)
                    delta = delta_3aps_remove(current, x)
                    if delta >= 0 or random.random() < math.exp(delta / temp):
                        current[x] = False
                        current_score += delta
                        accepted = True
                        if current_score > best_score:
                            best_score = current_score
                            best = current[:]
            else:
                in_set = [x for x in range(1, n + 1) if current[x]]
                out_set = [x for x in range(1, n + 1) if not current[x]]
                if in_set and out_set:
                    x_out = random.choice(in_set)
                    x_in = random.choice(out_set)
                    if valid_swap(current, x_out, x_in):
                        delta = delta_3aps_swap(current, x_out, x_in)
                        if delta >= 0 or random.random() < math.exp(delta / temp):
                            current[x_out] = False
                            current[x_in] = True
                            current_score += delta
                            accepted = True
                            if current_score > best_score:
                                best_score = current_score
                                best = current[:]
            
            if not accepted:
                temp *= cooling
                if temp < min_temp:
                    temp = 0.3
            else:
                temp *= cooling
        
        return best, best_score
    
    best_overall = None
    best_overall_score = -1
    
    S0 = greedy_construct()
    best, score = local_search(S0, max_iters=5000)
    if score > best_overall_score:
        best_overall = best
        best_overall_score = score
    
    while time.time() < deadline:
        S_rand = random_greedy()
        best, score = local_search(S_rand, max_iters=3000)
        if score > best_overall_score:
            best_overall = best
            best_overall_score = score
    
    result = sorted([x for x in range(1, n + 1) if best_overall[x]])
    
    if not result:
        return [1]
    
    return result
