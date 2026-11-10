import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Max sum is 128 + 128 = 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # We want to find A and B such that all 9 sums are distinct primes.
    # Iterate over possible A
    # Optimization: We can fix a_1=1 to reduce symmetry? No, range is fixed.
    # But we can iterate a1, a2, a3 in increasing order.
    
    # Precompute valid b candidates for a given A?
    # For a fixed A = (a1, a2, a3), a number b is "valid" if a1+b, a2+b, a3+b are all prime.
    # We need 3 distinct such b's, say b1, b2, b3, such that the 9 sums a_i+b_j are all distinct.
    
    # To speed up, we can iterate a1, a2, a3
    # 128 choose 3 is ~350k.
    # For each A, finding valid b's takes 128 checks.
    # Total ~45M checks. Might be tight in Python.
    # Let's add early termination and deadline check.
    
    # We can also prune: if a1+b, a2+b, a3+b are prime, they must be >= 2.
    # So b >= 2 - min(A).
    
    for a1 in range(1, 127):
        if time.time() > deadline:
            # Return a known valid solution if we have one, else the seed
            # The seed is valid with size 5.
            return [[1, 7, 13], [4, 10, 16]]
            
        for a2 in range(a1 + 1, 128):
            for a3 in range(a2 + 1, 128):
                A = [a1, a2, a3]
                
                # Find all b in [1, 128] such that a1+b, a2+b, a3+b are prime
                valid_bs = []
                for b in range(1, 129):
                    if is_prime[a1+b] and is_prime[a2+b] and is_prime[a3+b]:
                        valid_bs.append(b)
                
                if len(valid_bs) < 3:
                    continue
                
                # We need to pick 3 distinct b's from valid_bs
                # such that the 9 sums are distinct.
                # If valid_bs has exactly 3, just check them.
                # If more, we can try the first few combinations.
                
                # Check distinctness condition:
                # Sums are distinct iff for all i!=k, j!=l, a_i + b_j != a_k + b_l
                # a_i - a_k != b_l - b_j
                
                # Let's just generate the sums for a few candidate B sets.
                # To keep it fast, we only check the first 10 combinations of 3 from valid_bs
                # or all if len(valid_bs) is small.
                
                bs_to_check = valid_bs
                if len(bs_to_check) > 10:
                    bs_to_check = bs_to_check[:10]
                
                for B in combinations(bs_to_check, 3):
                    sums = []
                    valid = True
                    for a in A:
                        for b in B:
                            s = a + b
                            sums.append(s)
                    
                    # Check if all 9 are distinct
                    if len(set(sums)) != 9:
                        continue
                    
                    # Check if all are prime (already checked individually, but just in case of logic error)
                    # Actually, we ensured a_i+b_j are prime for each b in valid_bs.
                    # So they are prime.
                    
                    return [A, list(B)]
    
    # If no 9-sum solution found, return the seed (size 5)
    return [[1, 7, 13], [4, 10, 16]]
