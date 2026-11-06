import time
import random
import math
import sys

def construct():
    deadline = time.time() + 145  # Safety margin for 150s limit
    max_range = 10**12
    best_k = 1
    best_s = 1
    
    def check_run_fast(phi_arr, start, max_len=200):
        """Given phi array for [start, start+max_len-1], find longest prefix with distinct values"""
        seen = set()
        k = 0
        for i in range(max_len):
            val = phi_arr[i]
            if val in seen:
                break
            seen.add(val)
            k += 1
        return k
    
    def segmented_phi(start, length):
        """Compute phi(n) for n in [start, start+length-1] using segmented sieve"""
        if length <= 0:
            return []
        
        end = start + length - 1
        # Need primes up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        
        # Generate primes up to limit using simple sieve
        if limit < 2:
            primes = []
        else:
            sieve = bytearray(b'\x01') * (limit + 1)
            sieve[0] = 0
            if limit >= 1:
                sieve[1] = 0
            for i in range(2, int(math.isqrt(limit)) + 1):
                if sieve[i]:
                    sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
            primes = [i for i in range(2, limit + 1) if sieve[i]]
        
        # Initialize phi array
        phi = list(range(start, end + 1))
        
        # For each prime p, mark multiples and adjust phi
        for p in primes:
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first > end:
                continue
            for j in range(first, end + 1, p):
                idx = j - start
                if phi[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
        
        return phi
    
    # Strategy 1: Segmented sieve scan of small-to-medium range
    # Scan in chunks to find long runs
    chunk_size = 500  # Check windows of 500, look for runs up to 200
    scan_end = min(5000000, max_range)  # Scan up to 5 million initially
    
    pos = 1
    while pos <= scan_end and time.time() < deadline - 10:
        remaining = scan_end - pos + 1
        if remaining <= 0:
            break
        actual_chunk = min(chunk_size, remaining)
        
        phi_arr = segmented_phi(pos, actual_chunk)
        
        # Sliding window to find max distinct-totient run
        # For efficiency, check all starting positions in this chunk
        best_in_chunk = 0
        best_pos_in_chunk = 0
        
        # Use sliding window with hash set
        seen = set()
        left = 0
        for right in range(len(phi_arr)):
            val = phi_arr[right]
            while val in seen and left <= right:
                seen.remove(phi_arr[left])
                left += 1
            seen.add(val)
            window_len = right - left + 1
            if window_len > best_in_chunk:
                best_in_chunk = window_len
                best_pos_in_chunk = left + pos
            if window_len >= 200:
                break
        
        if best_in_chunk > best_k:
            best_k = best_in_chunk
            best_s = best_pos_in_chunk
            if best_k >= 200:
                break
        
        # Move by chunk_size but with overlap to catch runs crossing boundaries
        pos += chunk_size - 199  # Overlap by 199 to catch boundary runs
        if pos < 1:
            pos = 1
    
    # Strategy 2: Random targeted search in larger ranges using segmented sieve
    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        
        if attempts % 4 == 0:
            # Random in large range
            s = random.randint(10**6, max_range - 500)
        elif attempts % 4 == 1:
            # Near primorials
            primorials = [30030, 510510, 9699690, 223092870, 6469693230]
            p = random.choice(primorials)
            multiplier = random.randint(1, max(1, max_range // p - 500))
            s = p * multiplier + random.randint(-100, 100)
            s = max(1, min(s, max_range - 500))
        elif attempts % 4 == 2:
            # Random in medium range
            s = random.randint(10**5, 10**9)
        else:
            # Random in very large range
            s = random.randint(10**9, max_range - 500)
        
        if time.time() > deadline - 3:
            break
        
        # Use segmented sieve for a window around s
        window_start = max(1, s - 100)
        window_len = min(500, max_range - window_start + 1)
        
        phi_arr = segmented_phi(window_start, window_len)
        
        # Find best run in this window
        seen = set()
        left = 0
        best_in_window = 0
        best_pos_in_window = 0
        for right in range(len(phi_arr)):
            val = phi_arr[right]
            while val in seen and left <= right:
                seen.remove(phi_arr[left])
                left += 1
            seen.add(val)
            window_len2 = right - left + 1
            if window_len2 > best_in_window:
                best_in_window = window_len2
                best_pos_in_window = left + window_start
            if window_len2 >= 200:
                break
        
        if best_in_window > best_k:
            best_k = best_in_window
            best_s = best_pos_in_window
            if best_k >= 200:
                break
            
            # Try to extend: check a larger window around the found position
            if time.time() < deadline - 3:
                ext_start = max(1, best_s - 200)
                ext_len = min(600, max_range - ext_start + 1)
                phi_ext = segmented_phi(ext_start, ext_len)
                seen2 = set()
                left2 = 0
                best_ext = 0
                best_ext_pos = 0
                for right2 in range(len(phi_ext)):
                    val2 = phi_ext[right2]
                    while val2 in seen2 and left2 <= right2:
                        seen2.remove(phi_ext[left2])
                        left2 += 1
                    seen2.add(val2)
                    wl = right2 - left2 + 1
                    if wl > best_ext:
                        best_ext = wl
                        best_ext_pos = left2 + ext_start
                    if wl >= 200:
                        break
                if best_ext > best_k:
                    best_k = best_ext
                    best_s = best_ext_pos
                    if best_k >= 200:
                        break
    
    return {'start': best_s, 'length': best_k}
