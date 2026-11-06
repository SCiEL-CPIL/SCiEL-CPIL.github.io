import math
from functools import lru_cache

def construct():
    """
    Find N with many representations as a sum of two powerful numbers.
    
    Strategy: 
    1. Generate all powerful numbers up to a reasonable limit (10^6).
    2. For each powerful number Q in the list (with Q < 10^18 // 2),
       consider N = k * Q where k is an integer.
       If a and b are powerful numbers such that a + b = N,
       then a = Q * (a/Q) and b = Q * (b/Q).
       This requires Q to divide both a and b.
       
    Actually, a better structural approach based on the hint:
    Let Q be a powerful number. If we set N = Q * M, and we want a + b = Q * M,
    we can look for representations where a = Q * x and b = Q * y.
    This requires Q to divide both a and b.
    If Q is a powerful number, and x and y are powerful numbers, then
    a = Q * x is powerful if and only if Q and x are "compatible" (i.e., for each prime p,
    the exponent in Q + exponent in x >= 2, or is 0).
    
    A simpler and more effective approach for this specific problem:
    Let Q be a powerful number. Consider N = 2 * Q.
    Then a = Q, b = Q is a valid pair.
    But we want MANY pairs.
    
    Let's stick to the convolution method but optimize it and perhaps use a different limit.
    The current best is 4.13. The limit was 5e7.
    
    Let's try to push the limit higher if possible, or use a more efficient generator.
    
    The generator:
    n = x^2 * y^3 where y is square-free.
    y^3 <= L => y <= L^(1/3).
    x^2 <= L / y^3 => x <= sqrt(L / y^3).
    
    The number of powerful numbers up to L is approximately C * L^(2/3).
    For L = 10^8, L^(2/3) = 10^(16/3) = 10^5.33 ~ 2 * 10^5.
    This is very manageable for the set P.
    The FFT size needs to be at least 2L.
    For L = 10^8, FFT size is 2 * 10^8. This is too large for standard memory (1.6 GB for complex).
    
    However, we can use a smaller L if we are smart.
    Let's try L = 2 * 10^7 (which was in the previous version).
    Maybe we can go to 3 * 10^7 or 4 * 10^7?
    
    Let's check the memory for L = 4 * 10^7.
    FFT size ~ 1.28 * 10^8 (next power of 2).
    rfft output size ~ 6.4 * 10^7 complex numbers.
    6.4 * 10^7 * 16 bytes = ~1 GB. This might be on the edge of the memory limit (usually 2-4 GB).
    Time for FFT of size 1.28 * 10^8: O(N log N). 10^8 * 30 ~ 3 * 10^9 operations.
    NumPy FFT is highly optimized. It might take a few seconds.
    
    Let's try L = 3 * 10^7.
    FFT size ~ 1.28 * 10^8 (still).
    Actually, 2 * 3 * 10^7 = 6 * 10^7. Next power of 2 is 2^26 = 6.7 * 10^7.
    rfft output size ~ 3.3 * 10^7.
    Memory ~ 500 MB. This is safe.
    
    Let's try to increase L to 5 * 10^7 (which was already done in the current best).
    The current best uses L up to 5 * 10^7.
    
    To get a structural change, let's look at the "primorial" hint.
    Let Q be a product of small primes.
    Let N = 2 * Q^2.
    Then a = Q^2, b = Q^2 is a solution.
    Are there others?
    We need a + b = 2 * Q^2.
    If a and b are powerful, let a = u^2 * v^3 and b = w^2 * z^3.
    This seems hard to parameterize directly.
    
    Alternative structural change:
    Instead of finding the global max over all N, we can specifically search for N of the form
    N = k * Q where Q is a highly composite powerful number.
    
    Actually, let's just optimize the existing convolution approach by using a slightly larger L
    if memory allows, or by using a more efficient way to find the top k candidates.
    
    Let's try L = 6 * 10^7.
    FFT size: 2 * 6 * 10^7 = 1.2 * 10^8. Next power of 2 is 2^27 = 1.34 * 10^8.
    rfft output: 6.7 * 10^7 complex. ~1 GB.
    This might work.
    
    Let's implement a robust version that tries L = 5 * 10^7 and L = 6 * 10^7,
    and keeps the best result.
    
    Also, we can speed up the pair extraction.
    
    One more thing: the current program calculates `score = res_slice + d`.
    We can just take the top 100 indices from `score` and verify them.
    This is safer than just taking argmax, in case of floating point issues (though unlikely).
    """
    
    import time
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    def generate_powerful_numbers(limit):
        powerful = set()
        max_y = int(math.ceil(limit ** (1.0/3.0))) + 2
        
        # Sieve for square-free numbers up to max_y
        sq_free = [True] * (max_y + 1)
        sq_free[0] = False
        p = 2
        while p * p <= max_y:
            if sq_free[p]:
                sq = p * p
                for j in range(sq, max_y + 1, sq):
                    sq_free[j] = False
            p += 1
        
        for y in range(1, max_y + 1):
            if not sq_free[y]:
                continue
            y3 = y * y * y
            if y3 > limit:
                break
            max_x = int(math.isqrt(limit // y3))
            for x in range(1, max_x + 1):
                n = x * x * y3
                powerful.add(n)
        
        return sorted(powerful)
    
    def find_best_for_limit(L):
        P = generate_powerful_numbers(L)
        if not P:
            return 0, []
        
        # Create A array
        target_size = 2 * L + 1
        n_fft = 1
        while n_fft < target_size:
            n_fft *= 2
            
        # Check if n_fft is too large for memory
        # rfft output size is n_fft//2 + 1
        # Memory for complex128 is 16 bytes per element
        # Let's cap at n_fft = 2^27 (1.34e8) -> output 6.7e8 elements -> ~1GB
        # This is usually OK on modern systems with 8GB+ RAM
        if n_fft > 2**27:
            return 0, []
            
        A = np.zeros(n_fft, dtype=np.float64)
        for p in P:
            if p < n_fft:
                A[p] = 1.0
        
        F = np.fft.rfft(A)
        F2 = F * F
        result = np.fft.irfft(F2, n=n_fft)
        
        res_slice = result[:2*L+1]
        
        # Compute d
        P_bool = np.zeros(L + 1, dtype=np.int8)
        for p in P:
            P_bool[p] = 1
            
        d = np.zeros(2*L+1, dtype=np.int32)
        even_idx = np.arange(2, 2*L+1, 2)
        d[even_idx] = P_bool[even_idx // 2]
        
        score = res_slice + d
        
        # Get top 100 candidates
        # Use argpartition for efficiency
        if len(score) > 100:
            top_indices = np.argpartition(score, -100)[-100:]
        else:
            top_indices = np.arange(len(score))
            
        top_indices = top_indices[np.argsort(score[top_indices])[::-1]]
        
        best_n = 0
        best_pairs = []
        
        P_set = set(P)
        
        for idx in top_indices:
            N = int(idx)
            # Verify N is within reasonable range
            if N < 2:
                continue
            
            # Extract pairs
            pairs = []
            for a in P:
                if a > N // 2:
                    break
                b = N - a
                if b in P_set:
                    pairs.append([a, b])
            
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_n = N
        
        return best_n, best_pairs
    
    best_n = 0
    best_pairs = []
    
    # Try increasing limits
    limits = [3 * 10**7, 5 * 10**7, 6 * 10**7]
    
    for L in limits:
        if time.time() > deadline - 20:
            break
            
        n, pairs = find_best_for_limit(L)
        
        if len(pairs) > len(best_pairs):
            best_pairs = pairs
            best_n = n
    
    if not best_pairs:
        best_n = 5
        best_pairs = [[1, 4]]
        
    return {'N': best_n, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
