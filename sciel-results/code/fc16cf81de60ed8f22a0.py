import time
import sys

def construct():
    N = 256
    # The condition 3 * min(S) > 2 * max(S) guarantees validity for S = [K, N]
    # We want to maximize N - K + 1, so minimize K.
    # K must satisfy 3K > 2N => K > 2N/3
    # K_min = floor(2N/3) + 1
    
    K = (2 * N) // 3 + 1
    
    # Verify the condition for safety
    if 3 * K <= 2 * N:
        K += 1
        
    S = list(range(K, N + 1))
    
    # Optional: Try to add elements smaller than K
    # This is a heuristic improvement attempt
    # Check if any x in [1, K-1] can be added to S without removing any elements from S
    # Condition: for all b, c in S with b < c (or b<=c), x must not divide b+c
    # Since x < b, c, we just check if any b+c is a multiple of x.
    
    # Precompute sums of pairs in S?
    # S has ~86 elements. Pairs ~ 86*86/2 = 3600.
    # Checking each x in 1..170 against 3600 sums is fast.
    
    # However, adding x might create conflicts with other added x's?
    # If we add multiple small numbers, they might conflict with each other?
    # Condition: a < b <= c.
    # If we add x1 < x2 < K.
    # x1 < x2 < b. x1 must not divide x2+b.
    # x1 < x2 < c. x1 must not divide x2+c.
    # x2 < b <= c. x2 must not divide b+c (already checked for original S? No, x2 is new).
    # Actually, if we add a set X of small numbers, we need to check all triples.
    # Given the time limit and the low probability of gaining many small numbers, 
    # and the high cost of checking, we might skip this or do a very limited search.
    
    # Let's try a greedy addition of small numbers.
    # Start with S_tail.
    # Candidates: 1 to K-1.
    # Sort candidates by "difficulty" (e.g., number of multiples in sum range)?
    # Or just iterate.
    
    # Actually, the "Knowledge" suggests the jump to 86 was the main gain.
    # I will return the tail.
    
    return S

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
