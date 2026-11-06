import time
import numpy as np
from math import isqrt, gcd

def _is_powerful(n):
    """Check if n is a powerful number (every prime factor appears with exponent >= 2)."""
    if n <= 1:
        return True
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            # Count exponent of d in x
            exp = 0
            while x % d == 0:
                x //= d
                exp += 1
            if exp == 1:
                return False
        d += 1
    # x is now a prime factor (if > 1)
    if x > 1:
        return False  # exponent is 1
    return True

def _factorize_check_powerful(n):
    """Check if n is powerful by exact factorization. Returns True if powerful."""
    if n <= 1:
        return True
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            exp = 0
            while x % d == 0:
                x //= d
                exp += 1
            if exp == 1:
                return False
        d += 1
    if x > 1:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerfuls = set()
    max_b = int(limit ** (1.0 / 3.0)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerfuls.add(val)
    return sorted(powerfuls)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate powerful numbers up to L
    L = 10**7
    powerfuls = _generate_powerful(L)
    powerfuls_set = set(powerfuls)
    n_pow = len(powerfuls)
    
    if n_pow < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    elapsed_gen = time.time() - start_time
    
    # Strategy: Generate candidate N values from sums of powerful numbers
    # Focus on N values in range [2, 2*L] where we can verify both a and b are <= L
    # Also try larger N values where b = N - a might be > L but still powerful
    
    # Phase 1: Find best N in [2, 2*L] using dense counting
    # Use numpy for efficient counting
    # Create array of powerful numbers as indices
    pow_arr = np.array(powerfuls, dtype=np.int64)
    
    # For N in [2, 2*L], count pairs (a,b) with a+b=N, a<=b, both powerful
    # Use FFT convolution for efficiency
    from scipy.signal import fftconvolve
    
    size = 2 * L + 1
    arr = np.zeros(size, dtype=np.float64)
    arr[pow_arr] = 1.0
    
    if time.time() < deadline - 80:
        counts = fftconvolve(arr, arr, mode='full')
        # counts[i] = number of pairs (a,b) with a+b=i
        # We want a <= b, so for i > L, all pairs have a < b
        # For i <= L, we need to adjust: count includes (a,b) and (b,a) for a != b
        # Actually fftconvolve gives sum over all a,b in our set with a+b=i
        # For a <= b: if a != b, each pair counted twice; if a == b, counted once
        # So for i: total = counts[i], diagonal = 1 if i/2 is powerful and i even, else 0
        # pairs_with_a_le_b = (counts[i] + diagonal) / 2
        
        # Find best N
        best_N = 2
        best_count = 0
        
        for N in range(2, 2 * L + 1):
            c = int(counts[N])
            diag = 1 if (N % 2 == 0 and (N // 2) in powerfuls_set) else 0
            pair_count = (c + diag) // 2
            if pair_count > best_count:
                best_count = pair_count
                best_N = N
        
        # Get pairs for best_N
        pairs = []
        for a in powerfuls:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        pairs.sort()
    else:
        # Fallback: sample-based search
        best_N = 2
        best_count = 0
        pairs = [[1, 1]]
    
    elapsed_fft = time.time() - start_time
    
    # Phase 2: Try larger N values where b might be > L
    # For N > 2*L, b = N - a > L for small a
    # We can check if b is powerful using factorization
    # Candidate N values: sums of two powerful numbers, but we can't enumerate all
    # Instead, try N = a + b where a is small powerful and b is a "large" powerful
    
    # Generate some large powerful numbers for b
    # Powerful numbers can be as large as 10^18
    # Form: a^2 * b^3 <= 10^18
    # Let's generate powerful numbers in ranges
    
    if time.time() < deadline - 30:
        # Try specific candidate N values
        # Heuristic: N values that are sums of "round" powerful numbers
        # e.g., N = p^2 + q^2 for primes p, q
        # Or N = p^2 * r^3 + q^2 * s^3
        
        # Generate candidate N values
        candidates = set()
        
        # Sum of two small powerfuls (already covered in Phase 1)
        # For larger N, try: a (small powerful) + b (large powerful)
        # where b is powerful and b <= 10^18 - a
        
        # Generate large powerful numbers
        # b = c^2 * d^3 where c^2 * d^3 is in [L+1, 10^18]
        # d^3 <= 10^18 => d <= 10^6
        # c^2 <= 10^18 / d^3 => c <= 10^9 / d^1.5
        
        large_powerfuls = set()
        max_d = 10**6
        for d in range(2, max_d + 1):
            d3 = d ** 3
            if d3 > 10**18:
                break
            max_c = isqrt(10**18 // d3)
            if max_c < 1:
                break
            # Sample c values to avoid too many
            # c^2 * d^3 should be > L
            min_c = isqrt((L + 1) // d3) + 1
            if min_c > max_c:
                continue
            # Sample c values
            step = max(1, (max_c - min_c) // 1000)
            for c in range(min_c, max_c + 1, step):
                val = c * c * d3
                if L < val <= 10**18:
                    large_powerfuls.add(val)
        
        # Also add perfect squares of large numbers
        max_c_sq = isqrt(10**18)
        for c in range(isqrt(L + 1) + 1, min(max_c_sq, isqrt(L + 1) + 100000)):
            val = c * c
            if L < val <= 10**18:
                large_powerfuls.add(val)
        
        large_powerfuls = sorted(large_powerfuls)
        
        # Try candidate N = a + b where a is small powerful, b is large powerful
        # Sample combinations
        import random
        random.seed(42)
        
        n_large = len(large_powerfuls)
        n_samples = min(50000, n_pow * n_large // 100) if n_large > 0 else 0
        
        candidate_Ns = set()
        for _ in range(n_samples):
            a_idx = random.randint(0, n_pow - 1)
            b_idx = random.randint(0, n_large - 1) if n_large > 0 else 0
            a = powerfuls[a_idx]
            b = large_powerfuls[b_idx] if n_large > 0 else 0
            N = a + b
            if 2 <= N <= 10**18:
                candidate_Ns.add(N)
        
        # Also add some structured candidates
        for a in powerfuls[:100]:
            for b in large_powerfuls[:100]:
                N = a + b
                if 2 <= N <= 10**18:
                    candidate_Ns.add(N)
        
        candidate_Ns = sorted(candidate_Ns)
        
        # Check each candidate for representations
        best_N2 = best_N
        best_count2 = len(pairs)
        best_pairs2 = pairs
        
        for N_cand in candidate_Ns:
            if time.time() > deadline - 15:
                break
            
            # Count representations: a + b = N_cand, a <= b, both powerful
            # a can be from powerfuls (<= L) or large_powerfuls
            # b = N_cand - a
            
            count = 0
            cand_pairs = []
            
            # Check a from powerfuls
            for a in powerfuls:
                if a > N_cand // 2:
                    break
                b = N_cand - a
                if b < a:
                    break
                if b in powerfuls_set:
                    count += 1
                    cand_pairs.append([a, b])
                elif b > L and b <= 10**18:
                    # Check if b is powerful
                    if _is_powerful(b):
                        count += 1
                        cand_pairs.append([a, b])
            
            # Also check a from large_powerfuls (for a > L)
            for a in large_powerfuls:
                if a > N_cand // 2:
                    break
                b = N_cand - a
                if b < a:
                    break
                if b in powerfuls_set:
                    if a not in powerfuls_set:  # avoid double counting
                        count += 1
                        cand_pairs.append([a, b])
                elif b > L and b <= 10**18:
                    if _is_powerful(b):
                        if a not in powerfuls_set and b not in powerfuls_set:
                            count += 1
                            cand_pairs.append([a, b])
            
            if count > best_count2:
                best_count2 = count
                best_N2 = N_cand
                best_pairs2 = cand_pairs
        
        if best_N2 != best_N:
            best_N = best_N2
            pairs = best_pairs2
    
    # Final verification
    if not pairs:
        pairs = [[1, 1]]
        best_N = 2
    
    return {'N': best_N, 'pairs': pairs}
