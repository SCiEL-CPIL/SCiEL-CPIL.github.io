import time
import numpy as np
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    # Strategy: Use a highly composite base B and enumerate powerful numbers
    # of the form B * k^2 where k is composed of primes dividing B.
    # This creates dense structure for sum collisions.
    
    # Highly composite number: product of small prime powers
    # B = 2^a * 3^b * 5^c * 7^d * ...
    # We want B to be powerful (all exponents >= 2)
    
    # Let's use B = (2*3*5*7*11*13)^2 = 30030^2 = 901800900
    # This is powerful and has many divisors.
    
    base = 2 * 3 * 5 * 7 * 11 * 13  # 30030
    B = base * base  # 901800900, this is powerful
    
    # Powerful numbers of form B * m where m is powerful
    # Actually, we need all powerful numbers up to 10^18.
    # A powerful number n = product p_i^{e_i} where e_i >= 2.
    # So n = k^2 * s where s is squarefree.
    
    # Better approach: enumerate all powerful numbers up to some limit L
    # using the structure: n = a^2 * b^3 where b is squarefree.
    
    # For efficiency, let's use FFT with a larger limit
    # But first, let's try the target-specific approach:
    
    # Generate powerful numbers up to L
    L = 10**8
    
    powerfuls = set()
    
    # Method: n = a^2 * b^3 where b is squarefree
    # b^3 <= L => b <= L^(1/3)
    # a^2 <= L / b^3 => a <= sqrt(L / b^3)
    
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    if time.time() > deadline - 60:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    # Now, instead of full FFT which is memory intensive,
    # let's use a smarter approach: focus on a specific N
    # and count representations by checking pairs.
    
    # But to find the best N, we need to search.
    # Let's use FFT but with careful memory management.
    
    # FFT approach for convolution
    n_fft = 1
    while n_fft < 2 * L + 1:
        n_fft *= 2
        
    # Check if we have enough memory
    # n_fft * 8 bytes for complex
    if n_fft > 2**26:  # 64M elements, 512MB for complex
        # Fall back to smaller L
        L = 5 * 10**7
        n_fft = 1
        while n_fft < 2 * L + 1:
            n_fft *= 2
        
        # Regenerate powerfuls for smaller L
        powerfuls = set()
        max_b = int(L ** (1.0/3.0)) + 2
        for b in range(1, max_b + 1):
            b3 = b * b * b
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= L:
                    powerfuls.add(n)
        powerfuls_list = sorted(powerfuls)
    
    A = np.zeros(n_fft, dtype=np.float64)
    A[powerfuls_list] = 1.0
    
    F = np.fft.rfft(A)
    C = F * F
    conv = np.fft.irfft(C, n=n_fft)
    
    if time.time() > deadline - 30:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    conv_int = np.rint(conv[:2*L+1]).astype(np.int64)
    
    if len(conv_int) < 3:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    max_count = int(np.max(conv_int[2:]))
    best_N = int(np.argmax(conv_int[2:])) + 2
    
    # Extract pairs
    powerfuls_set = set(powerfuls_list)
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    pairs.sort()
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': best_N, 'pairs': pairs}
