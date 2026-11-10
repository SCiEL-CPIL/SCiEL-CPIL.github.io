import random
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin for 90s limit
    
    # Weights: LCM(1..30) / i for i=1..30
    # LCM of 1..30
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 31):
        L = lcm(L, i)
    
    # w[i] = L / (i+1) for i=0..29 (term i contributes c[i]/(i+1), so integer weight is L/(i+1))
    w = [L // (i + 1) for i in range(30)]
    
    best_abs = None
    best_c = None
    
    def eval_sum(c):
        s = 0
        for i in range(30):
            s += c[i] * w[i]
        return s
    
    def local_search(c, max_iters=2000):
        """Hill climbing: flip one coefficient at a time to reduce |sum|"""
        cur = list(c)
        cur_sum = eval_sum(cur)
        cur_abs = abs(cur_sum)
        if cur_abs == 0:
            return None, 0  # invalid
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            best_delta = 0
            best_idx = -1
            best_val = -1
            for i in range(30):
                for new_val in (-1, 0, 1):
                    if new_val == cur[i]:
                        continue
                    delta = (new_val - cur[i]) * w[i]
                    new_sum = cur_sum + delta
                    new_abs = abs(new_sum)
                    if new_abs == 0:
                        continue
                    if new_abs < cur_abs:
                        best_delta = delta
                        best_idx = i
                        best_val = new_val
                        cur_abs = new_abs
                        improved = True
            if best_idx >= 0:
                cur_sum += best_delta
                cur[best_idx] = best_val
        if cur_abs == 0:
            return None, 0
        return cur, cur_abs
    
    def simulated_annealing(c_init, iters=5000):
        """SA with geometric cooling"""
        cur = list(c_init)
        cur_sum = eval_sum(cur)
        cur_abs = abs(cur_sum)
        if cur_abs == 0:
            return None, 0
        
        best = list(cur)
        best_abs = cur_abs
        best_sum = cur_sum
        
        T = 1e-6 * L  # initial temperature
        T_min = 1.0
        cooling = 0.9995
        
        for it in range(iters):
            # Pick random index and new value
            i = random.randint(0, 29)
            old_val = cur[i]
            new_val = random.choice([-1, 0, 1])
            if new_val == old_val:
                continue
            
            delta = (new_val - old_val) * w[i]
            new_sum = cur_sum + delta
            new_abs = abs(new_sum)
            
            if new_abs == 0:
                # Reject zero sums
                continue
            
            if new_abs < cur_abs:
                # Accept
                cur[i] = new_val
                cur_sum = new_sum
                cur_abs = new_abs
                if cur_abs < best_abs:
                    best_abs = cur_abs
                    best = list(cur)
                    best_sum = cur_sum
            else:
                # Metropolis acceptance
                if new_abs <= T_min:
                    accept_prob = 1.0
                else:
                    # P = exp(-(new_abs - cur_abs) / T)
                    diff = (new_abs - cur_abs) / T
                    if diff < 700:  # avoid underflow
                        accept_prob = math.exp(-diff)
                    else:
                        accept_prob = 0.0
                
                if random.random() < accept_prob:
                    cur[i] = new_val
                    cur_sum = new_sum
                    cur_abs = new_abs
            
            T = max(T * cooling, T_min)
        
        if best_abs == 0:
            return None, 0
        return best, best_abs
    
    # Initial solution: trivial
    c_init = [0] * 28 + [1, -1]
    c, abs_val = local_search(c_init, max_iters=500)
    if c is not None:
        best_c = c
        best_abs = abs_val
    
    # Multiple restarts with random initializations
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random initialization
        c_rand = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Quick local search to get a decent starting point
        c_ls, abs_ls = local_search(c_rand, max_iters=300)
        if c_ls is None:
            c_ls = c_rand
            abs_ls = abs(eval_sum(c_rand))
        
        # Simulated annealing
        c_sa, abs_sa = simulated_annealing(c_ls, iters=3000)
        if c_sa is not None and (best_abs is None or abs_sa < best_abs):
            best_abs = abs_sa
            best_c = c_sa
        
        # Final local search polish on the best SA result
        c_polish, abs_polish = local_search(c_sa, max_iters=1000)
        if c_polish is not None and (best_abs is None or abs_polish < best_abs):
            best_abs = abs_polish
            best_c = c_polish
    
    # If somehow we ended up with zero sum, fix it
    if best_c is None or eval_sum(best_c) == 0:
        # Fallback: find a nonzero solution
        best_c = [0] * 28 + [1, -1]
        best_abs = abs(eval_sum(best_c))
    
    return best_c
