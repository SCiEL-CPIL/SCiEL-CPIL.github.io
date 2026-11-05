import time
import math

def construct():
    start_time = time.time()
    TIME_LIMIT = 120.0
    # We need to search for s such that s+k-1 <= 10^12.
    # The seed only searched up to 20,000.
    # Let's search up to 10^6 or 10^7 depending on time.
    # A sieve for phi up to N takes O(N log log N).
    # For N=10^6, it's very fast. For N=10^7, it's still fast (<1s).
    # For N=10^8, it might take a few seconds.
    # Let's aim for N = 10^6 first, then extend if time permits.
    
    # Strategy:
    # 1. Compute phi(n) for n in [1, N] using a sieve.
    # 2. Find the longest run of consecutive integers with distinct phi values in [1, N].
    # 3. If time remains, extend N or search other ranges.
    #    However, the property "distinct totients" is local.
    #    If we find a run of length k at start s, that's a valid solution.
    #    The maximum k is likely to be found in the first few million integers?
    #    Actually, phi(n) is roughly n * e^-gamma. Collisions are frequent.
    #    The longest run of distinct values is likely to be short (e.g., 10-30).
    #    The seed found k=1 at best? No, seed score 1.0 means it returned something.
    #    Let's check what the seed found.
    #    Seed searched 1..20000.
    #    Let's just do a robust sieve up to 10^6 or 10^7.
    
    # Determine N based on time.
    # Sieve phi for N=10^6 is ~0.1s.
    # Sieve phi for N=10^7 is ~1s.
    # Sieve phi for N=10^8 is ~10s.
    # Let's try N=10^6 first.
    
    N = 10**6
    
    # 1. Sieve for phi
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # 2. Find longest run of distinct phi values
    # We want max k such that phi[s], ..., phi[s+k-1] are distinct.
    # Use a sliding window with a hash map or set.
    # Since phi values are up to N, we can use a list or set.
    # Set is fine for O(1) lookup.
    
    best_k = 1
    best_s = 1
    
    # Sliding window
    left = 1
    seen = {}
    max_val = N  # phi(n) <= n
    
    # We iterate right from 1 to N
    # We maintain a window [left, right] where all phi values are distinct.
    # If phi[right] is in seen, we move left until it's not.
    
    for right in range(1, N + 1):
        val = phi[right]
        
        # If val is already in the window, shrink from left
        if val in seen:
            # Move left until the previous occurrence of val is out of window
            while left <= seen[val]:
                del seen[phi[left]]
                left += 1
        
        seen[val] = right
        current_len = right - left + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left
            
        # Check time
        if time.time() - start_time > TIME_LIMIT - 10.0:
            break
            
    # If we have time, we could search further, but usually the longest runs
    # of distinct totients are not extremely long.
    # For example, phi(1)=1, phi(2)=1 (collision at 2).
    # phi(3)=2, phi(4)=2 (collision at 4).
    # phi(5)=4, phi(6)=2 (collision with 3? No, 2 is used. phi(3)=2, phi(6)=2).
    # Let's trace:
    # 1: {1} len 1
    # 2: {1, 1} -> collision. window [2,2] {1} len 1. Best 1.
    # 3: {1, 2} len 2. Best 2.
    # 4: phi(4)=2. Collision with 3. Move left past 3. Window [4,4] {2} len 1.
    # 5: phi(5)=4. Window [4,5] {2,4} len 2.
    # 6: phi(6)=2. Collision with 4. Move left past 4. Window [6,6] {2} len 1.
    # 7: phi(7)=6. Window [6,7] {2,6} len 2.
    # 8: phi(8)=4. Collision with 5? No, 5 is out. Wait.
    # Let's re-evaluate carefully.
    
    # The algorithm above is correct.
    
    return {'start': best_s, 'length': best_k}
