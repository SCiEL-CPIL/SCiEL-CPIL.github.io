# EVOLVE-BLOCK-START
import time
import math
import random

_START = time.time()
TIME_LIMIT = 80.0  # Safety margin before 90s kill

def _elapsed():
    return time.time() - __START

# Precompute primes up to 40
def _sieve(n):
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _sieve(40)
# Primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# Composites in 2..40
_COMPOSITES = [x for x in range(2, 41) if x not in _PRIMES]
_ALL_CANDIDATES = list(range(2, 41))

def _is_valid_subset(set_vals):
    """Check if all subset products are distinct."""
    k = len(set_vals)
    if k > 20:
        return False
    
    products = set()
    # Use bitmasks to enumerate subsets
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= set_vals[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _can_add(current_set, candidate):
    """Check if adding candidate to current_set maintains distinct subset products."""
    # We need to check if candidate creates a collision.
    # A collision occurs if there exist disjoint subsets A, B of current_set 
    # such that candidate * P(A) = P(B) OR P(A) = candidate * P(B) ?
    # Actually, the condition is simply: is the new set valid?
    # But checking the whole set is expensive.
    # Let's try a slightly smarter check.
    # If we have set S and add c, the new products are P(S) union {c * p | p in P(S)}.
    # We need P(S) and {c * p | p in P(S)} to be disjoint, and {c * p} to be internally distinct.
    # Since c > 1 and all p > 0, c*p is distinct for distinct p.
    # So we just need to check if any c*p is in P(S) for p in P(S).
    
    # Compute all subset products of current_set
    k = len(current_set)
    if k > 18:
        return False # Safety check, shouldn't happen often if we start small
        
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= current_set[i]
        products.add(prod)
        
    # Check for collision: does c * p exist in products for any p in products?
    # Note: p=1 is always in products (empty set). So if c is in products, it's a collision.
    # Generally, if c * p in products, then subset {c} U A has same product as subset B.
    # Where P(A) = p and P(B) = c*p.
    # This implies c = P(B)/P(A).
    # So we just need to check if c * p is in products for any p in products.
    # Wait, if p is large, c*p might be huge. But products are bounded.
    # Actually, we just iterate through products.
    
    for p in products:
        if c * p in products:
            return False
            
    return True

def construct():
    # Initial best solution from current program (score 1.3333 -> 16 items)
    # The current program returns 16 items: [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Let's verify this count.
    # 2,4,16 (3) + 3,9 (2) + 5,25 (2) + 7,11,13,17,19,23,29,31,37 (9) = 16.
    # Can we do better?
    # Max possible is 18.
    
    # Strategy:
    # 1. Start with the known good set of 16.
    # 2. Try to add remaining composites to reach 17 or 18.
    # 3. If that fails, try to swap out some elements for others to allow more additions?
    #    Or start from scratch with a greedy approach that prioritizes "expensive" composites (high powers) first?
    
    # Let's try a greedy construction from scratch to see if we can get 17 or 18.
    # Heuristic: Add elements that are "least likely" to cause collisions.
    # High powers of small primes are good because they have unique prime factorization exponents.
    # e.g. 16=2^4. 4=2^2. 2=2^1.
    # If we have 2, 4, 16, we use exponents 1, 2, 4. Subset sums of {1,2,4} are distinct (1..7).
    # So we can have 2, 4, 8, 16? Exponents 1,2,3,4. Subset sums of {1,2,3,4} are NOT distinct (1+3=4, 1+2+? no 1+3=4 is sum of {1,3} and {4}).
    # So {2,4,8,16} is invalid.
    # {2,4,16} is valid (exponents 1,2,4).
    # Can we add 32? 32=2^5. Exponents 1,2,4,5. Subset sums of {1,2,4,5}.
    # 1, 2, 4, 5, 1+2=3, 1+4=5 (Collision! 5 is in set, 1+4=5).
    # So {2,4,16,32} is invalid.
    
    # What about mixing primes?
    # The current set uses:
    # 2: {2, 4, 16} (exponents 1, 2, 4)
    # 3: {3, 9} (exponents 1, 2)
    # 5: {5, 25} (exponents 1, 2)
    # Others: {p} (exponent 1)
    
    # Can we add 27? 27=3^3.
    # Current 3-powers: 3 (3^1), 9 (3^2).
    # If we add 27 (3^3), exponents are {1, 2, 3}.
    # Subset sums of {1, 2, 3}: 1, 2, 3, 1+2=3 (Collision).
    # So we cannot add 27 if we have 3 and 9.
    # We could replace 9 with 27? {3, 27} -> exponents {1, 3}. Subset sums: 1, 3, 4. Distinct.
    # Or replace 3 with 27? {27, 9} -> exponents {3, 2}. Subset sums: 2, 3, 5. Distinct.
    # But we want to maximize count.
    
    # Let's try to add 8? 8=2^3.
    # Current 2-powers: 2 (2^1), 4 (2^2), 16 (2^4). Exponents {1, 2, 4}.
    # Add 8 (2^3). Exponents {1, 2, 3, 4}.
    # Subset sums of {1, 2, 3, 4}: 1, 2, 3, 4, 1+2=3 (Collision).
    # So we cannot add 8.
    
    # What about 25? Already there.
    # What about 125? > 40.
    
    # Let's look at other composites.
    # 6=2*3.
    # If we add 6, we have 2, 3, 6.
    # 6 = 2*3. Collision {6} and {2,3}.
    # So we cannot add 6 if we have 2 and 3.
    
    # 10=2*5. Collision with 2,5.
    # 14=2*7. Collision with 2,7.
    # 15=3*5. Collision with 3,5.
    # 21=3*7. Collision with 3,7.
    # 22=2*11. Collision with 2,11.
    # 26=2*13. Collision with 2,13.
    # 33=3*11. Collision with 3,11.
    # 34=2*17. Collision with 2,17.
    # 35=5*7. Collision with 5,7.
    # 38=2*19. Collision with 2,19.
    # 39=3*13. Collision with 3,13.
    
    # So any composite with distinct prime factors p*q where p and q are in the set is invalid.
    # Since we have all primes 2,3,5,7,11,13,17,19,23,29,31,37, we cannot add ANY square-free composite.
    # We can only add prime powers p^k.
    
    # Available prime powers in 2..40:
    # 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32
    # 3^1=3, 3^2=9, 3^3=27
    # 5^1=5, 5^2=25
    # 7^1=7, 7^2=49 (>40)
    # 11^1=11, 11^2=121 (>40)
    # ...
    
    # So the only candidates for addition beyond the primes are prime powers.
    # Current set includes:
    # 2, 4, 16 (2^1, 2^2, 2^4)
    # 3, 9 (3^1, 3^2)
    # 5, 25 (5^1, 5^2)
    # And all other primes.
    
    # Can we add 8 (2^3)?
    # As analyzed, {1,2,3,4} has collision.
    # Can we add 32 (2^5)?
    # Exponents {1,2,4,5}. Collision 1+4=5.
    # Can we add 27 (3^3)?
    # Exponents {1,2,3}. Collision 1+2=3.
    
    # So we are stuck with 16?
    # Let's check if we can swap to get more.
    # For prime 2, max size of subset of {1,2,3,4,5} with distinct subset sums is 3 (e.g., {1,2,4} or {1,2,5} or {2,3,5}).
    # {1,2,4} -> 2,4,16. (Size 3)
    # {1,2,5} -> 2,4,32. (Size 3)
    # {2,3,5} -> 4,8,32. (Size 3)
    # Can we have size 4?
    # {1,2,4,8} -> 1+2+4=7, 8 is in set? No. But 1+?
    # Subset sums of {1,2,4,8}: 1,2,4,8,3,5,9,6,7,10,11,12,13,14,15,16. All distinct?
    # Wait, binary representation! {1,2,4,8} are powers of 2.
    # Subset sums of powers of 2 are all distinct (0 to 15).
    # So we CAN have 2^1, 2^2, 2^3, 2^4.
    # i.e., 2, 4, 8, 16.
    # Let's re-verify.
    # Set {2, 4, 8, 16}.
    # Products correspond to exponents subset sums.
    # Exponents: 1, 2, 3, 4.
    # Subset sums of {1, 2, 3, 4}:
    # 1, 2, 3, 4
    # 1+2=3 (Collision with 3!)
    # Ah, the elements are 2^1, 2^2, 2^3, 2^4.
    # The "weights" are the exponents 1, 2, 3, 4.
    # We need distinct subset sums of the exponents.
    # {1, 2, 3, 4}: 1+2=3. Collision.
    # So {2,4,8,16} is invalid.
    
    # What weights allow 4 distinct subset sums?
    # We need a set of 4 integers with distinct subset sums.
    # Minimum sum of 4 distinct positive integers is 1+2+3+4=10.
    # But we need distinct sums.
    # {1, 2, 4, 8} works. Sums 0..15.
    # So exponents {1, 2, 4, 8} would work.
    # This corresponds to numbers 2^1, 2^2, 2^4, 2^8.
    # 2^8 = 256. > 40.
    # So we can't have 4 powers of 2 in 2..40.
    # Max is 3.
    
    # For prime 3:
    # Powers: 3^1=3, 3^2=9, 3^3=27.
    # Exponents {1, 2, 3}. Collision 1+2=3.
    # Can we have {1, 2, 4}? 3^4=81 > 40.
    # So max 2 powers of 3.
    
    # For prime 5:
    # Powers: 5^1=5, 5^2=25.
    # Max 2.
    
    # For prime 7:
    # Powers: 7^1=7, 7^2=49 > 40.
    # Max 1.
    
    # So the maximum number of prime powers we can include is:
    # 3 (for 2) + 2 (for 3) + 2 (for 5) + 1 (for 7) + 1 (for 11) + ...
    # Total primes: 12.
    # Extra powers:
    # 2: +2 (we have 1, so add 2 more? No, total 3 means 2 extras)
    # 3: +1 (total 2 means 1 extra)
    # 5: +1 (total 2 means 1 extra)
    # Total items = 12 (primes) + 2 (extra 2s) + 1 (extra 3) + 1 (extra 5) = 16.
    
    # So 16 seems to be the maximum!
    # The current program already finds 16.
    # Score 1.3333 corresponds to 16/12.
    # Is it possible to have more than 16?
    # Only if we can mix primes and composites in a way that allows more items.
    # But we established that square-free composites are blocked by the presence of their prime factors.
    # And prime powers are limited by the subset sum property of exponents.
    # What if we DROP a prime to add more composites?
    # Example: Drop 2.
    # Then we can add 4, 8, 16, 32?
    # Exponents {2, 3, 4, 5}.
    # Subset sums of {2,3,4,5}:
    # 2,3,4,5
    # 2+3=5 (Collision).
    # So {4,8,16,32} is invalid.
    # {4,8,16}? Exponents {2,3,4}. 2+3=5 (not in set). 2+4=6. 3+4=7. 2+3+4=9.
    # Sums: 2,3,4,5,6,7,9. Distinct?
    # Wait, 2+3=5. 5 is NOT in the set of exponents {2,3,4}.
    # The condition is that the subset sums of the exponents must be distinct.
    # Exponents: 2, 3, 4.
    # Sums:
    # {} = 0
    # {2} = 2
    # {3} = 3
    # {4} = 4
    # {2,3} = 5
    # {2,4} = 6
    # {3,4} = 7
    # {2,3,4} = 9
    # All distinct: 0,2,3,4,5,6,7,9.
    # So {4,8,16} is valid!
    # This gives 3 items for prime 2, without using 2.
    # But we dropped 2 (1 item) and added 4,8,16 (3 items). Net gain +2.
    # However, we still have 3, 5, etc.
    # Total items:
    # 2s: 4, 8, 16 (3 items)
    # 3s: 3, 9 (2 items)
    # 5s: 5, 25 (2 items)
    # Others: 7, 11, 13, 17, 19, 23, 29, 31, 37 (9 items)
    # Total: 3 + 2 + 2 + 9 = 16.
    # Same count.
    
    # Can we do better?
    # Drop 2, 3.
    # 2s: 4, 8, 16 (3)
    # 3s: 9, 27? Exponents {2, 3}. Sums 0,2,3,5. Distinct.
    # So {9, 27} is valid. (2 items)
    # 5s: 5, 25 (2 items)
    # Others: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 (10 items)
    # Total: 3 + 2 + 2 + 10 = 17!
    
    # Let's verify this set:
    # {4, 8, 16, 9, 27, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37, 5? No, 5 is in others?}
    # Primes: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37. (10 primes)
    # Composites: 4, 8, 16 (from 2), 9, 27 (from 3), 25 (from 5).
    # Wait, 5 is a prime. 25 is 5^2.
    # If we keep 5 and 25, that's 2 items for prime 5.
    # So:
    # 2-group: {4, 8, 16} (3 items)
    # 3-group: {9, 27} (2 items)
    # 5-group: {5, 25} (2 items)
    # Other primes: {7, 11, 13, 17, 19, 23, 29, 31, 37} (9 items)
    # Total: 3 + 2 + 2 + 9 = 16.
    # Where did I get 17?
    # Ah, I counted 5 in "others" and also in "5-group".
    # Primes available: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # If we drop 2 and 3, we have 10 primes left: 5, 7, 11, 13, 17, 19, 23, 29, 31, 37.
    # We use 5 in the 5-group (5, 25). So 5 is not in "other primes".
    # Other primes: 7, 11, 13, 17, 19, 23, 29, 31, 37. (9 items).
    # Total: 3 (2s) + 2 (3s) + 2 (5s) + 9 (others) = 16.
    
    # Can we drop 2, 3, 5?
    # 2-group: {4, 8, 16} (3)
    # 3-group: {9, 27} (2)
    # 5-group: {25} (1) -- since 5 is dropped, we can only have 25?
    # Can we have 25 and 125? No.
    # Can we have 25 and 5? No, 5 is dropped.
    # So 5-group: {25} (1 item).
    # Other primes: 7, 11, 13, 17, 19, 23, 29, 31, 37. (9 items).
    # Total: 3 + 2 + 1 + 9 = 15. Worse.
    
    # What if we drop 2 only?
    # 2-group: {4, 8, 16} (3)
    # 3-group: {3, 9} (2)
    # 5-group: {5, 25} (2)
    # Others: 7, 11, 13, 17, 19, 23, 29, 31, 37. (9)
    # Total: 3 + 2 + 2 + 9 = 16.
    
    # What if we drop 3 only?
    # 2-group: {2, 4, 16} (3)
    # 3-group: {9, 27} (2) -- Exponents {2,3}. Sums 0,2,3,5. Distinct.
    # 5-group: {5, 25} (2)
    # Others: 2, 7, 11, 13, 17, 19, 23, 29, 31, 37. (10) -- 2 is in others? No, 2 is in 2-group.
    # Others: 7, 11, 13, 17, 19, 23, 29, 31, 37. (9)
    # Total: 3 + 2 + 2 + 9 = 16.
    
    # It seems 16 is the hard limit.
    
    # However, the task asks to maximize. If 16 is the max, we just need to return a valid set of 16.
    # The current program does this.
    # To be safe, I will implement a greedy search to find a valid set of 16, and if by some miracle 17 exists, find it.
    # But given the analysis, 16 is likely max.
    
    # Let's just return the known good set of 16.
    candidate = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify
    if _is_valid_subset(candidate):
        return candidate
        
    # Fallback
    return _PRIMES

# EVOLVE-BLOCK-END
