import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    A = [L // i for i in range(1, 31)]
    
    best_abs_N = float('inf')
    best_coeffs = None
    
    def get_N(c):
        return sum(ci * ai for ci, ai in zip(c, A))
        
    def greedy_init():
        c = [0] * 30
        N = 0
        # Process largest terms first
        indices = sorted(range(30), key=lambda i: A[i], reverse=True)
        for i in indices:
            # We want to minimize |N + c_i A_i|
            # Options for c_i: -1, 0, 1
            # N is current sum
            # If we pick 0, sum stays N
            # If we pick 1, sum becomes N + A[i]
            # If we pick -1, sum becomes N - A[i]
            # We pick the one that minimizes |new_sum|
            
            # Since A[i] > 0:
            if N > 0:
                # N is positive, we want to subtract
                # Compare |N - A[i]| vs |N|
                if abs(N - A[i]) < abs(N):
                    c[i] = -1
                    N -= A[i]
                else:
                    c[i] = 0
                    # N unchanged
            elif N < 0:
                # N is negative, we want to add
                if abs(N + A[i]) < abs(N):
                    c[i] = 1
                    N += A[i]
                else:
                    c[i] = 0
            else:
                # N == 0
                # Any choice keeps |N| small? No, picking non-zero makes it A[i]
                # So 0 is best to keep it 0
                c[i] = 0
        
        # If we ended up with N=0, we need to perturb
        # This greedy might result in exactly 0.
        # If N==0, flip a small term
        if N == 0:
             # Find smallest A[i] where c[i] can be changed to make N non-zero
             # Actually, just return c, and let local search handle it?
             # Local search rejects N=0 moves.
             # So we need a valid start.
             # Let's just pick a random small index and set it to 1
             idx = random.randint(0, 29)
             c[idx] = 1
             N = A[idx]
             
        return c, N

    def local_search(c, N, time_limit):
        # c is list, N is current sum
        while time.time() < time_limit:
            improved = False
            # Try single bit flips
            for i in range(30):
                old_val = c[i]
                # Try other 2 values
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * A[i]
                    new_N = N + delta
                    if new_N == 0:
                        continue
                    if abs(new_N) < abs(N):
                        c[i] = new_val
                        N = new_N
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Stuck. Kick.
                # Change 2 random indices
                i1, i2 = random.sample(range(30), 2)
                c[i1] = random.choice([-1, 0, 1])
                c[i2] = random.choice([-1, 0, 1])
                N = get_N(c)
                if N == 0:
                    # Kick again
                    c[random.randint(0,29)] = random.choice([-1, 0, 1])
                    N = get_N(c)
        
        return N, c

    while time.time() < deadline:
        c, N = greedy_init()
        
        # Limit each local search to 2 seconds to allow restarts
        search_limit = min(deadline, time.time() + 2.0)
        N, c_res = local_search(c, N, search_limit)
        
        if N != 0 and abs(N) < best_abs_N:
            best_abs_N = abs(N)
            best_coeffs = c_res[:]
            
        if best_abs_N == 1:
            break
            
    if best_coeffs is None:
        return [0] * 28 + [1, -1]
        
    return best_coeffs
