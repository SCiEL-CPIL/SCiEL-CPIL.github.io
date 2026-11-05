import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    # 1. Precomputation
    # Sieve for primes up to 256
    limit = 256
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    
    primes_set = set(primes)
    
    # Prime factor masks for O(1) coprime checks
    # Map each prime to a bit index
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    pf_mask = [0] * (limit + 1)
    for n in range(1, limit + 1):
        mask = 0
        for p in primes:
            if n % p == 0:
                mask |= (1 << prime_to_idx[p])
        pf_mask[n] = mask

    def is_pairwise_coprime_mask(nums):
        """Check if a list of numbers is pairwise coprime using masks."""
        combined_mask = 0
        for n in nums:
            if n > limit:
                return False
            m = pf_mask[n]
            if combined_mask & m:
                return False
            combined_mask |= m
        return True

    def check_score(A, B):
        """Returns the count of distinct pairwise coprime sums in A+B."""
        sums = [a + b for a in A for b in B]
        distinct_sums = list(set(sums))
        
        # We need to find the maximum subset of distinct_sums that is pairwise coprime.
        # Since we want to maximize cardinality, and the max is 9, 
        # if all 9 are distinct and pairwise coprime, score is 9.
        # If not, we might have a smaller valid set.
        # However, the problem asks to "Maximize the cardinality of A+B" 
        # where "All DISTINCT values in A+B must be pairwise coprime".
        # This implies the *entire set of distinct sums* must be pairwise coprime.
        # So if any pair in the distinct sums is not coprime, the configuration is invalid?
        # Re-reading: "All DISTINCT values in A+B must be pairwise coprime."
        # This is a constraint on the valid output.
        # The score is the cardinality of A+B (distinct sums).
        # So we need to find A, B such that the set S = {a+b | a in A, b in B} 
        # is pairwise coprime. Then score = |S|.
        
        if len(distinct_sums) != len(sums):
             # Repeated representations are allowed, so distinct count < 9 is possible.
             pass
             
        # Check if ALL distinct sums are pairwise coprime
        if is_pairwise_coprime_mask(distinct_sums):
            return len(distinct_sums)
        else:
            return 0 # Invalid configuration for scoring purposes, or low score?
            # The task says "Maximize the cardinality... All DISTINCT values... must be pairwise coprime".
            # This suggests we only count valid configurations.
            # If a configuration fails the coprime check, it's not a valid candidate for the max.
            # But the evaluator measures "raw value". If I return an invalid set, does it score 0?
            # "an invalid result... scores 0".
            # So I must return a valid set.
            # If I can't find a valid set with score 9, I should return the best valid set found.
            # If I return a set that is NOT pairwise coprime, it scores 0.
            # So check_score should return 0 if not valid, else len(distinct_sums).

    # 2. Search Strategy
    # We want score 9. This requires 9 distinct sums, all pairwise coprime.
    # As established, all sums must be odd -> A odd, B even (or vice versa).
    
    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    best_score = 0
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    
    # Fallback valid solution (must be pairwise coprime sums)
    # Let's find a quick valid one to start with
    # A=[1,3,5], B=[2,4,6] -> Sums: 3,5,7, 5,7,9, 7,9,11. Distinct: {3,5,7,9,11}. 
    # 3 and 9 share factor 3. Invalid.
    # A=[1,3,7], B=[2,4,8] -> Sums: 3,5,9, 5,7,11, 9,11,15. Distinct: {3,5,7,9,11,15}.
    # 3,9,15 share 3. Invalid.
    # A=[1,5,7], B=[2,4,8] -> 3,5,9, 7,9,13, 9,13,15. 3,9,15 share 3.
    # A=[1,5,11], B=[2,4,8] -> 3,5,9, 7,9,13, 13,15,19. 3,9,15 share 3.
    # A=[1,7,13], B=[4,10,16] -> 5,11,17, 11,17,23, 17,23,29. Distinct: {5,11,17,23,29}. All prime. Valid. Score 5.
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = check_score(best_A, best_B) # Should be 5

    # Phase 1: Search for 9 distinct PRIME sums.
    # This is the most likely path to score 9.
    # Iterate A (odd triples). For each A, find valid B (even) where all sums are prime.
    
    # To speed up, we can precompute for each odd a, which even b make a+b prime.
    # But b is shared across a1, a2, a3.
    
    # Randomized search with pruning
    rng = random.Random(42)
    
    # Precompute prime status for all possible sums
    # sum in [3, 256]
    
    # Optimization: 
    # For a fixed A=(a1,a2,a3), a candidate b is "good" if a1+b, a2+b, a3+b are all prime.
    # We collect all such b's. If we have >=3, we pick 3 and check if the 9 sums are distinct.
    # Since they are all prime, distinctness is the only other check.
    
    # We will iterate through A combinations.
    # To be thorough but fast, we will sample A's and also do a structured sweep.
    
    # Structured sweep on smaller A values first (primes are denser relatively? no, but smaller sums are easier to check)
    # Actually, let's just iterate all A combinations from a subset of odds.
    # There are 64 odds. C(64,3) = 41664 combinations.
    # For each, we scan 64 evens. 41664 * 64 = ~2.6 million checks.
    # Each check is 3 prime lookups. Very fast in Python if optimized.
    
    all_A = list(combinations(odds, 3))
    # Shuffle to find a solution quickly if it exists early, or sort by sum?
    # Let's sort by max element to try smaller numbers first?
    # Or just randomize to avoid bias.
    rng.shuffle(all_A)
    
    found_9 = False
    for A in all_A:
        if time.time() > deadline - 10:
            break
            
        a1, a2, a3 = A
        viable_b = []
        
        for b in evens:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            # Sums must be <= 256
            if s3 > limit:
                continue
            if is_prime[s1] and is_prime[s2] and is_prime[s3]:
                viable_b.append(b)
                
        if len(viable_b) < 3:
            continue
            
        # Try all triples from viable_b
        # viable_b is usually small.
        for i in range(len(viable_b)):
            if time.time() > deadline - 5:
                break
            for j in range(i + 1, len(viable_b)):
                for k in range(j + 1, len(viable_b)):
                    B = [viable_b[i], viable_b[j], viable_b[k]]
                    # Check distinctness of all 9 sums
                    sums = []
                    valid = True
                    for a in A:
                        for b in B:
                            s = a + b
                            if s in sums:
                                valid = False
                                break
                            sums.append(s)
                        if not valid:
                            break
                    if valid:
                        # All 9 sums are distinct primes -> Pairwise Coprime
                        found_9 = True
                        best_A = list(A)
                        best_B = B
                        best_score = 9
                        break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break

    if found_9:
        return [best_A, best_B]

    # Phase 2: If 9 primes not found, search for 9 distinct pairwise coprime sums (not necessarily prime).
    # This is harder. We use the bitmask method.
    # We can use a similar structure: A odd, B even.
    # For each A, find B such that the 9 sums are pairwise coprime.
    
    # To speed this up, we can use a greedy construction of B.
    # Pick b1. Calculate sums s11, s12, s13.
    # Pick b2. Calculate sums s21, s22, s23.
    # Check if s2* are coprime with s1* and each other.
    # Pick b3. Check s3* against s1*, s2*.
    
    # We will iterate A again, but this time we don't restrict to primes.
    # We will use a random subset of A's to save time if Phase 1 failed.
    
    # Let's try a more targeted search for Phase 2.
    # Random restarts with local improvement?
    # Or just iterate A and greedily pick B.
    
    # Greedy B selection for a given A:
    def find_B_greedy(A, time_limit):
        a1, a2, a3 = A
        # Candidate B's are even numbers
        # We want to pick b1, b2, b3
        # such that all 9 sums are pairwise coprime.
        
        # Precompute masks for sums for all potential b's?
        # For each b, we have 3 sums.
        # Let's just iterate b1, b2, b3.
        
        # To prune, we can precompute for each b the "mask" of its 3 sums.
        # But sums depend on A.
        
        # Let's try b1 in evens.
        # For each b1, check if sums are valid (coprime among themselves).
        # If yes, keep b1 as candidate.
        # For each b2 in evens (b2 != b1), check sums against b1 sums.
        # If yes, keep (b1, b2).
        # For each b3 in evens (b3 != b1, b2), check sums against b1, b2 sums.
        
        # This is O(N^3) for each A. N=64. 64^3 = 262k.
        # With 40k A's, that's 10 billion ops. Too slow.
        
        # We need to limit A's or B's.
        # Let's try only a subset of A's for Phase 2.
        # Or, use the prime-based viable_b list from Phase 1, but relax the "prime" constraint to "coprime to others".
        
        # Actually, if Phase 1 failed, it means no A has 3 distinct b's making 9 distinct primes.
        # It might still have 3 b's making 9 distinct coprime numbers (e.g., 1, 9, 25...).
        
        # Let's do a random search for Phase 2 with a time limit.
        pass

    # Alternative Phase 2: Random search with bitmask validation
    # We already have a fallback score of 5.
    # Can we get 6, 7, 8?
    # If we can't get 9, maximizing 5->8 is good.
    
    # Let's do a quick random search for better scores (6-8) if 9 is not found.
    # Use the bitmask check.
    
    if best_score < 9 and time.time() < deadline - 5:
        for _ in range(50000):
            if time.time() > deadline - 5:
                break
            
            # Random A (odd), Random B (even)
            A = rng.sample(odds, 3)
            B = rng.sample(evens, 3)
            
            score = check_score(A, B)
            if score > best_score:
                best_score = score
                best_A = A
                best_B = B
                if best_score == 9:
                    break

    # Final check
    # Ensure we return a valid list
    if best_score == 0:
        # Return a known valid small set
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {A}, B: {B}")
    sums = [a+b for a in A for b in B]
    print(f"Sums: {sums}")
    print(f"Distinct: {len(set(sums))}")
