import time
import math

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # Primes up to 10^6 for segmented sieve
    # We need primes up to sqrt(max_n). max_n is 10^12, so sqrt is 10^6.
    primes = sieve_primes(10**6)
    
    def compute_phi_range(L, R):
        """
        Computes phi(n) for n in [L, R] using segmented sieve.
        Returns a list of phi values.
        """
        W = R - L + 1
        rem = [L + i for i in range(W)]
        phi = [L + i for i in range(W)]
        
        for p in primes:
            if p * p > R:
                break
            m0 = ((L + p - 1) // p) * p
            for m in range(m0, R + 1, p):
                idx = m - L
                if rem[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
                        
        for i in range(W):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
                
        return phi

    def find_max_run(phi_list, start_offset):
        seen = {}
        left = 0
        max_len = 0
        max_start = start_offset
        
        for right in range(len(phi_list)):
            val = phi_list[right]
            if val in seen:
                if seen[val] >= left:
                    left = seen[val] + 1
            seen[val] = right
            
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start_offset + left
                
        return max_len, max_start

    best_k = 1
    best_s = 1
    
    # Phase 1: Small numbers
    if time.time() < deadline:
        phi_small = compute_phi_range(1, 100000)
        k, s = find_max_run(phi_small, 1)
        if k > best_k:
            best_k = k
            best_s = s
            
    # Phase 2: Segmented scan
    CHUNK_SIZE = 2 * 10**6
    L = 10**6
    # Scan up to 10^8
    while L + CHUNK_SIZE - 1 <= 10**8:
        if time.time() > deadline - 5:
            break
            
        R = L + CHUNK_SIZE - 1
        phi_chunk = compute_phi_range(L, R)
        k, s = find_max_run(phi_chunk, L)
        if k > best_k:
            best_k = k
            best_s = s
            
        L += CHUNK_SIZE
        
    return {'start': best_s, 'length': best_k}
