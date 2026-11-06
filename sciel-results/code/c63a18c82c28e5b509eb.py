import time
import math
import random

def sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    # Precompute primes up to 10^6 for trial division fallback
    primes = sieve_primes(10**6)
    prime_set = set(primes)
    
    def phi_trial(n):
        """Compute totient using trial division by precomputed primes."""
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
                if temp == 1:
                    break
        if temp > 1:
            result -= result // temp
        return result
    
    def segmented_phi_window(start, length):
        """
        Compute phi for all numbers in [start, start+length-1] using segmented SPF.
        Returns list of phi values.
        """
        if start == 1:
            # Handle 1 specially
            result = [1]
            if length == 1:
                return result
            start = 2
            length -= 1
        
        # Segmented sieve for smallest prime factor
        spf = [0] * length  # 0 means prime (or unmarked)
        
        limit = int(math.isqrt(start + length - 1))
        
        for p in primes:
            if p > limit:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            for m in range(first, start + length, p):
                idx = m - start
                if spf[idx] == 0:
                    spf[idx] = p
        
        # Compute phi values
        phi_vals = []
        for i in range(length):
            n = start + i
            if n == 1:
                phi_vals.append(1)
                continue
            temp = n
            result = n
            p = spf[i]
            while temp > 1:
                if p == 0:
                    p = temp  # temp is prime
                result -= result // p
                while temp % p == 0:
                    temp //= p
                # Find next prime factor
                if temp > 1:
                    # Need to find next prime factor of temp
                    # Use trial division for remaining part
                    next_p = 0
                    for q in primes:
                        if q * q > temp:
                            break
                        if temp % q == 0:
                            next_p = q
                            break
                    if next_p == 0:
                        next_p = temp
                    p = next_p
            phi_vals.append(result)
        return phi_vals
    
    def find_longest_distinct_run(phi_vals):
        """Find the longest contiguous subarray with all distinct values."""
        n = len(phi_vals)
        if n == 0:
            return 0, 0
        best_len = 1
        best_start = 0
        left = 0
        seen = {}
        for right in range(n):
            v = phi_vals[right]
            if v in seen and seen[v] >= left:
                left = seen[v] + 1
            seen[v] = right
            if right - left + 1 > best_len:
                best_len = right - left + 1
                best_start = left
        return best_start, best_len
    
    def check_run_trial(s, k):
        """Check if phi(s), ..., phi(s+k-1) are all distinct using trial division."""
        seen = set()
        for i in range(k):
            n = s + i
            v = phi_trial(n)
            if v in seen:
                return False
            seen.add(v)
        return True
    
    def find_best_run():
        best_s = 5
        best_k = 3
        
        # Strategy 1: Try to find a long run by scanning a large window
        # Use segmented phi to compute phi for a window, then find longest distinct subarray
        window_sizes = [2000, 5000, 10000]
        
        # Candidate start positions
        candidate_starts = []
        
        # Near 10^12
        for offset in range(0, 20000, 100):
            s_try = 10**12 - 200 - offset
            if s_try >= 1:
                candidate_starts.append(s_try)
        
        # Mid-range positions
        for base in [10**10, 5*10**10, 10**11, 5*10**11, 9*10**11]:
            for offset in range(0, 5000, 200):
                s_try = base - offset
                if s_try >= 1:
                    candidate_starts.append(s_try)
        
        # Some smaller numbers (known to have good properties)
        for s_try in range(1, 1000):
            candidate_starts.append(s_try)
        
        # Random sampling
        for _ in range(50):
            s_try = random.randint(1, 10**12 - 200)
            candidate_starts.append(s_try)
        
        # Shuffle for variety
        random.shuffle(candidate_starts)
        
        # Try each candidate with segmented phi
        for s_try in candidate_starts:
            if time.time() > deadline - 30:
                break
            
            # Compute phi for a window starting at s_try
            win_size = min(2000, 10**12 - s_try + 1)
            if win_size < 200:
                continue
            
            try:
                phi_vals = segmented_phi_window(s_try, win_size)
            except Exception:
                continue
            
            # Find longest distinct run in this window
            local_start, local_len = find_longest_distinct_run(phi_vals)
            
            if local_len > best_k:
                best_k = local_len
                best_s = s_try + local_start
                
                # If we found a very long run, try to extend it
                if best_k >= 150:
                    # Try to extend left
                    extend_left = 0
                    seen_vals = set(phi_vals[local_start:local_start + best_k])
                    for ext in range(1, 200):
                        if s_try + local_start - ext < 1:
                            break
                        n_ext = s_try + local_start - ext
                        v_ext = phi_trial(n_ext)
                        if v_ext in seen_vals:
                            break
                        seen_vals.add(v_ext)
                        extend_left += 1
                        if time.time() > deadline - 20:
                            break
                    if extend_left > 0:
                        best_k += extend_left
                        best_s -= extend_left
                    
                    # Try to extend right
                    extend_right = 0
                    end_idx = local_start + best_k
                    for ext in range(1, 200):
                        if end_idx + ext >= win_size:
                            break
                        v_ext = phi_vals[end_idx + ext]
                        if v_ext in seen_vals:
                            break
                        seen_vals.add(v_ext)
                        extend_right += 1
                        if time.time() > deadline - 20:
                            break
                    if extend_right > 0:
                        best_k += extend_right
                    
                    # Cap at 200
                    best_k = min(best_k, 200)
                    
                    if best_k == 200:
                        break
        
        # Strategy 2: If we haven't found a great run, try targeted search for specific k
        if best_k < 100:
            for k_try in range(min(best_k + 50, 200), best_k, -1):
                if time.time() > deadline - 10:
                    break
                # Try a few more specific positions
                for s_try in [10**12 - k_try, 10**11 - k_try // 2, 10**10 - k_try // 4]:
                    if s_try < 1:
                        continue
                    if check_run_trial(s_try, k_try):
                        best_s = s_try
                        best_k = k_try
                        break
                if best_k >= k_try:
                    break
        
        return best_s, best_k
    
    s, k = find_best_run()
    return {'start': s, 'length': k}
