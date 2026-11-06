import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    MIN_N = 10000
    MAX_N = 1000000
    
    # Compute largest prime factor for all numbers up to MAX_N
    # Using a sieve: for each prime p, mark all multiples with lpf = p
    # Since we iterate p in increasing order, the last prime to divide n is the largest
    
    lpf_arr = [0] * (MAX_N + 1)
    for i in range(2, MAX_N + 1):
        if lpf_arr[i] == 0:  # i is prime
            for j in range(i, MAX_N + 1, i):
                lpf_arr[j] = i
    
    # Scan for the best consecutive pair
    best_n = MIN_N
    best_val = float('inf')
    
    for n in range(MIN_N, MAX_N):
        if time.time() > deadline - 5:
            break
        
        max_lpf = max(lpf_arr[n], lpf_arr[n+1])
        
        if max_lpf < best_val:
            best_val = max_lpf
            best_n = n
    
    # Local refinement: check a wider window around the best found
    # and also try to find even better pairs by checking neighborhoods
    search_window = 200
    local_start = max(MIN_N, best_n - search_window)
    local_end = min(MAX_N - 1, best_n + search_window)
    
    for n in range(local_start, local_end + 1):
        if time.time() > deadline - 2:
            break
        
        max_lpf = max(lpf_arr[n], lpf_arr[n+1])
        
        if max_lpf < best_val:
            best_val = max_lpf
            best_n = n
    
    return int(best_n)
