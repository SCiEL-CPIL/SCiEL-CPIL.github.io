"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    primes = _primes_upto(1000)
    # We want sum(1/p) <= 1. Greedy by smallest p first (largest 1/p)
    # But we need to cover 1..L. Let's think about which primes to use.
    
    # The seed uses primes from 13 upward, assigning residue (i+1) % p to the i-th such prime.
    # This covers 1, 2, 3, ... up to the number of primes used.
    # Sum of 1/p for primes 13..997 is about 0.92 (as noted in seed).
    
    # To maximize L, we want to cover as many consecutive integers starting from 1.
    # Each class (p, r) covers all n such that n ≡ r (mod p).
    # For covering 1..L, we need for each k in 1..L, some (p, r) with k ≡ r (mod p).
    
    # Key insight: if we have a class (p, 0), it covers p, 2p, 3p, ...
    # If we have (p, 1), it covers 1, p+1, 2p+1, ...
    # If we have (p, 2), it covers 2, p+2, ...
    
    # The seed's approach: for the i-th prime p_i (starting from 13), assign residue (i+1) % p_i.
    # So prime 13 gets residue 1%13=1, covers 1, 14, 27, ...
    # Prime 17 gets residue 2%17=2, covers 2, 19, 36, ...
    # Prime 19 gets residue 3%19=3, covers 3, 22, 41, ...
    # etc.
    
    # This covers 1, 2, 3, ... up to the number of primes used, because for the i-th prime,
    # residue i is less than p_i (since p_i >= 13 > i for small i), so i % p_i = i.
    # So integer i is covered by the i-th prime.
    
    # How many primes from 13 to 997? Let's count.
    # Primes: 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, ...
    # Up to 997.
    
    # Sum of reciprocals: sum_{p in [13,997]} 1/p.
    # This is approximately log(log(997)) - log(log(13)) ≈ log(6.9) - log(2.56) ≈ 1.93 - 0.94 ≈ 0.99.
    # Actually let me be more careful. Sum of 1/p for p <= x is log(log(x)) + B where B ≈ 0.261.
    # Sum for p <= 997: log(log(997)) + 0.261 ≈ log(6.904) + 0.261 ≈ 1.932 + 0.261 ≈ 2.193
    # Sum for p < 13 (i.e., p=2,3,5,7,11): 1/2+1/3+1/5+1/7+1/11 = 0.5+0.333+0.2+0.143+0.091 ≈ 1.267
    # So sum for 13..997 ≈ 2.193 - 1.267 ≈ 0.926. Yes, about 0.92.
    
    # Number of primes from 13 to 997: pi(997) - pi(11) = 168 - 5 = 163.
    # So the seed covers 1..163.
    
    # Can we do better? We have budget sum(1/p) <= 1.
    # If we use smaller primes, 1/p is larger, so we use budget faster.
    # But smaller primes cover more integers in 1..L per unit of "budget"?
    
    # Actually, the question is: what's the maximum L such that we can cover 1..L with distinct primes <= 1000, sum 1/p <= 1?
    
    # Let me think about this differently. For each integer k in 1..L, we need at least one prime p and residue r=k%p such that (p,r) is in our set.
    
    # A simple strategy: for each k, pick a prime p_k that divides (k - r_k) for some r_k, i.e., p_k | (k - r_k). But r_k = k mod p_k, so this is automatic. The constraint is that we use distinct primes and sum 1/p <= 1.
    
    # Actually, each prime can cover multiple integers. If we use prime p with residue r, it covers r, r+p, r+2p, ... all in 1..L.
    
    # So the problem is: choose a set of (p, r) pairs with distinct p <= 1000, sum 1/p <= 1, maximizing L such that every k in 1..L is covered.
    
    # Greedy approach: start with L=1, and try to extend. But this is complex.
    
    # Alternative: use the seed's approach but try to optimize residues.
    # The seed assigns residue i to the i-th prime (from 13). This means each prime covers exactly one small integer (itself, roughly).
    # But a prime p with residue r covers r, r+p, r+2p, ... So if r is small, it covers many integers up to L.
    
    # For example, if we use p=2, r=1, it covers 1, 3, 5, 7, 9, ... all odd numbers.
    # If we use p=2, r=0, it covers 2, 4, 6, 8, ... all even numbers.
    # With p=2, r=1 and p=3, r=2: odd numbers and numbers ≡2 mod 3.
    # 1: odd ✓
    # 2: ≡2 mod 3 ✓
    # 3: odd ✓
    # 4: even, not odd, 4 mod 3 = 1 ≠ 2. Not covered!
    
    # So we need a more careful approach.
    
    # Let me try a different strategy: use small primes with carefully chosen residues.
    # This is essentially the covering system problem.
    
    # Known result: the smallest covering system with distinct moduli has moduli 2,3,4,6,12 (or similar).
    # But we're restricted to primes.
    
    # Let me try to use a greedy/backtracking approach to find a good set.
    
    # Actually, let me first compute what the seed achieves and then try to improve.
    
    # Seed: primes from 13 to 997, residues 1,2,3,...,163 (mod respective primes).
    # This covers 1..163 exactly (each i is covered by the i-th prime since i < p_i for i <= 163 and p_i >= 13... wait, for i=1, p=13, 1<13 ✓. For i=163, p is the 163rd prime >= 13. The 163rd prime overall is 997, and the 163rd prime >= 13 is the (163+5)=168th prime = 997. So p=997, i=163, 163 < 997 ✓.)
    
    # So L_seed = 163.
    
    # Can we do better? Let's try using smaller primes.
    # If we use p=2 with some residue, it covers about half the integers.
    # If we use p=2, r=1: covers all odd numbers.
    # Then we need to cover even numbers 2,4,6,...,2L.
    # For even numbers, we can use p=3. 2 mod 3 = 2, 4 mod 3 = 1, 6 mod 3 = 0, 8 mod 3 = 2, ...
    # If we use p=3, r=2: covers 2, 5, 8, 11, 14, 17, 20, 23, ... (numbers ≡2 mod 3)
    # If we use p=3, r=1: covers 1, 4, 7, 10, 13, 16, 19, 22, ... (numbers ≡1 mod 3)
    # If we use p=3, r=0: covers 3, 6, 9, 12, 15, 18, 21, 24, ... (numbers ≡0 mod 3)
    
    # But we can only use p=3 once. So we pick one residue.
    
    # This is getting complex. Let me try a computational approach.
    
    # Strategy: use a greedy algorithm that tries to extend L by adding primes.
    # Or, use a local search / simulated annealing.
    
    # Given time constraints, let me try a simple improvement:
    # 1. Start with the seed's solution (L=163).
    # 2. Try to add more coverage by using smaller primes that fit in the remaining budget.
    # 3. Or, try to reassign residues to cover more.
    
    # Actually, let me try a completely different approach:
    # Use a set of small primes and find residues that cover 1..L for the largest L.
    
    # Let me try: use primes 2,3,5,7,11,13,... and assign residues to maximize coverage.
    
    # For a given set of primes, the problem of finding residues to cover 1..L is a CSP.
    # We can use backtracking.
    
    # But first, which primes to use? The sum of 1/p must be <= 1.
    # Sum of 1/p for p=2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,...
    # 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59+1/61+1/67+1/71+1/73+1/79+1/83+1/89+1/97
    # ≈ 0.5+0.333+0.2+0.143+0.091+0.077+0.059+0.053+0.043+0.034+0.032+0.027+0.024+0.023+0.021+0.019+0.017+0.016+0.015+0.014+0.014+0.013+0.012+0.011+0.010
    # Let me sum: 0.5+0.333=0.833, +0.2=1.033. Already over 1!
    
    # So we can't use 2,3,5 all together. Sum of 1/2+1/3+1/5 = 1.033 > 1.
    
    # So we need to be selective.
    
    # Let me try: use p=2 (1/2=0.5), p=3 (1/3≈0.333), total 0.833. Remaining budget 0.167.
    # Next smallest prime is 5, 1/5=0.2 > 0.167. Can't use 5.
    # Next: 7, 1/7≈0.143 <= 0.167. Use it. Total: 0.833+0.143=0.976. Remaining: 0.024.
    # Next: 11, 1/11≈0.091 > 0.024. Can't use.
    # So with {2,3,7}, sum ≈ 0.976.
    
    # With {2,3,7}, can we cover a large L?
    # p=2, r=a: covers a, a+2, a+4, ...
    # p=3, r=b: covers b, b+3, b+6, ...
    # p=7, r=c: covers c, c+7, c+14, ...
    
    # We need to choose a in {0,1}, b in {0,1,2}, c in {0,...,6} to maximize L.
    
    # Let me enumerate. For each (a,b,c), find the largest L such that all 1..L are covered.
    
    # This is a finite search. Let me do it.
    
    # Actually, let me write code to do this for various prime sets.
    
    # But given the time limit, let me try a heuristic:
    # 1. Try the seed's solution.
    # 2. Try a few hand-crafted solutions with small primes.
    # 3. Return the best.
    
    # Let me compute the seed's L first.
    seed_primes = [p for p in primes if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_primes)]
    seed_sum = sum(Fraction(1, p) for p in seed_primes)
    
    # Compute L for a given set of classes
    def compute_L(classes, max_L=100000):
        covered = [False] * (max_L + 1)
        for p, r in classes:
            for n in range(r if r > 0 else p, max_L + 1, p):
                if 1 <= n <= max_L:
                    covered[n] = True
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    # This is O(L * num_classes) which could be slow for L=100000.
    # Let me optimize: for each class, mark covered positions.
    
    def compute_L_fast(classes, max_L=100000):
        covered = bytearray(max_L + 1)
        for p, r in classes:
            start = r if r > 0 else p
            if start < 1:
                start = r + p * ((1 - r + p - 1) // p)
            # Mark all n >= 1 with n ≡ r (mod p)
            for n in range(start, max_L + 1, p):
                covered[n] = 1
        L = 0
        for i in range(1, max_L + 1):
            if covered[i]:
                L = i
            else:
                break
        return L
    
    L_seed = compute_L_fast(seed_classes)
    
    # Now try some alternatives
    best_L = L_seed
    best_classes = seed_classes
    
    # Try {2, 3, 7}
    for a in range(2):
        for b in range(3):
            for c in range(7):
                classes = [[2, a], [3, b], [7, c]]
                s = Fraction(1,2) + Fraction(1,3) + Fraction(1,7)
                if s <= 1:
                    L = compute_L_fast(classes)
                    if L > best_L:
                        best_L = L
                        best_classes = classes
    
    # Try {2, 3, 11}
    s2311 = Fraction(1,2) + Fraction(1,3) + Fraction(1,11)
    if s2311 <= 1:
        for a in range(2):
            for b in range(3):
                for c in range(11):
                    classes = [[2, a], [3, b], [11, c]]
                    L = compute_L_fast(classes)
                    if L > best_L:
                        best_L = L
                        best_classes = classes
    
    # Try {2, 5, 7}
    s257 = Fraction(1,2) + Fraction(1,5) + Fraction(1,7)
    if s257 <= 1:
        for a in range(2):
            for b in range(5):
                for c in range(7):
                    classes = [[2, a], [5, b], [7, c]]
                    L = compute_L_fast(classes)
                    if L > best_L:
                        best_L = L
                        best_classes = classes
    
    # Try {3, 5, 7, 11}
    s35711 = Fraction(1,3) + Fraction(1,5) + Fraction(1,7) + Fraction(1,11)
    if s35711 <= 1:
        for a in range(3):
            for b in range(5):
                for c in range(7):
                    for d in range(11):
                        classes = [[3, a], [5, b], [7, c], [11, d]]
                        L = compute_L_fast(classes)
                        if L > best_L:
                            best_L = L
                            best_classes = classes
    
    # Try {2, 3, 7, 13}
    s23713 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,13)
    if s23713 <= 1:
        for a in range(2):
            for b in range(3):
                for c in range(7):
                    for d in range(13):
                        classes = [[2, a], [3, b], [7, c], [13, d]]
                        L = compute_L_fast(classes)
                        if L > best_L:
                            best_L = L
                            best_classes = classes
    
    # Try {2, 3, 7, 17}
    s23717 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,17)
    if s23717 <= 1:
        for a in range(2):
            for b in range(3):
                for c in range(7):
                    for d in range(17):
                        classes = [[2, a], [3, b], [7, c], [17, d]]
                        L = compute_L_fast(classes)
                        if L > best_L:
                            best_L = L
                            best_classes = classes
    
    # Check time
    if time.time() > deadline - 10:
        return best_classes
    
    # Try larger sets
    # {2, 3, 7, 13, 19}
    s2371319 = Fraction(1,2) + Fraction(1,3) + Fraction(1,7) + Fraction(1,13) + Fraction(1,19)
    if s2371319 <= 1:
        for a in range(2):
            for b in range(3):
                for c in range(7):
                    for d in range(13):
                        for e in range(19):
                            classes = [[2, a], [3, b], [7, c], [13, d], [19, e]]
                            L = compute_L_fast(classes)
                            if L > best_L:
                                best_L = L
                                best_classes = classes
                            if time.time() > deadline - 10:
                                return best_classes
    
    return best_classes

# EVOLVE-BLOCK-END
