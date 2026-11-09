import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    # List of small primes to test as the maximum prime factor P
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate a boolean array of p-smooth numbers <= limit
    # A number is p-smooth if its largest prime factor is <= p
    def generate_smooth_array(primes_list, limit):
        smooth = bytearray(limit + 1)
        smooth[1] = 1
        
        # Use nested loops: for each prime, multiply existing smooth numbers
        # We process primes in order, and for each prime p, we update smooth[]
        # by checking if smooth[x] is set and x*p <= limit, then set smooth[x*p]
        
        for p in primes_list:
            # For each multiple of p, check if the quotient is smooth
            # Actually, better approach: iterate through all x, if smooth[x] and x*p <= limit, mark smooth[x*p]
            # But this can be done more efficiently by iterating x from 1 to limit//p
            
            # We need to be careful: we should only use smooth numbers that were marked
            # BEFORE this prime was introduced (to avoid using p multiple times in one pass incorrectly)
            # Actually, the standard approach is:
            # For each prime p in order:
            #   For x from 1 to limit//p:
            #     if smooth[x] is set, then smooth[x*p] should be set
            # But this would allow p^2, p^3, etc. which is correct.
            # However, we need to make sure we're not using the newly marked values in the same iteration
            # for further multiplications by the same p. 
            # 
            # The correct way: for each prime p, we want to mark all numbers of the form
            # (smooth_number_not_involving_p) * p^k for k >= 1.
            # 
            # Simpler: use a list of current smooth numbers and expand.
            
            pass
        
        # Let's use a different approach: start with [1], and for each prime,
        # multiply all existing smooth numbers by powers of that prime.
        
        smooth_list = [1]
        for p in primes_list:
            new_smooths = []
            for x in smooth_list:
                val = x
                while val * p <= limit:
                    val *= p
                    new_smooths.append(val)
            # Add new smooths to smooth_list
            smooth_list.extend(new_smooths)
            for val in new_smooths:
                smooth[val] = 1
        
        return smooth

    for idx_p, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate all numbers whose prime factors are a subset of primes <= p
        current_primes = primes[:idx_p + 1]
        
        smooth_arr = generate_smooth_array(current_primes, N)
        
        # Check if there is a pair a, b where both are smooth and a + b = N
        # Linear scan for efficiency
        for a in range(1, N):
            if smooth_arr[a] and smooth_arr[N - a]:
                return [a, N - a]
        
        # If no pair found with max prime factor p, try next prime
    
    # Fallback: Sieve LPF and scan
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break
                
    return [best_a, best_b]
