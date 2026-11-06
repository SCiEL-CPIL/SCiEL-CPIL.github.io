import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin for safety

    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)

    # Candidate prime sets with sum(1/p) <= 1
    # Set A: {3, 5} + primes >= 13. Sum ~ 1/3 + 1/5 + sum(1/p for p>=13)
    # sum(1/p for p>=13) is approx 0.92 - 1/2 - 1/3 - 1/5 = 0.92 - 0.5 - 0.333 - 0.2 = -0.113?
    # Wait, sum of reciprocals of primes diverges.
    # Let's calculate exactly.
    # sum(1/p for p in [2..997]) ~ 2.198
    # sum(1/p for p in [2,3,5,7,11]) = 1/2+1/3+1/5+1/7+1/11 = 0.5+0.333+0.2+0.1428+0.0909 = 1.266
    # sum(1/p for p >= 13) = 2.198 - 1.266 = 0.932
    # Set A: {3,5} + p>=13. Sum = 1/3+1/5+0.932 = 0.333+0.2+0.932 = 1.465 > 1. INVALID.
    # My previous calculation in the prompt was wrong or I misread the set.
    # The prompt says: "Set A: {3, 5} + primes >= 13. Sum ~ 0.97."
    # Let's re-verify.
    # Maybe the prompt meant {3, 5} + primes >= some larger threshold?
    # Or maybe I am summing wrong.
    # Let's just compute valid sets programmatically to be safe.

    def is_valid_set(P):
        s = sum(Fraction(1, p) for p in P)
        return s <= 1

    # We want sets with high density of small primes to cover 1..L efficiently.
    # Small primes cover more numbers per unit of "budget" (1/p).
    # So we want to include 2, 3, 5, 7, 11, 13... as much as possible.
    # But sum(1/p) <= 1.
    # If we include 2, sum(1/2) = 0.5. Remaining budget 0.5.
    # Next smallest primes: 3 (0.333), 5 (0.2). 0.5+0.333+0.2 = 1.033 > 1.
    # So {2,3,5} is invalid.
    # {2,3} sum = 0.833. Remaining 0.166. Next prime 5 (0.2) too big. 7 (0.142) fits.
    # {2,3,7} sum = 0.833+0.142=0.975. Remaining 0.025. 11 (0.09) too big.
    # So {2,3,7} + primes p where 1/p <= 0.025 => p >= 40.
    # Set 1: {2, 3, 7} + [p for p in primes if p >= 41]
    
    # What if we don't use 2?
    # {3,5} sum = 0.533. Remaining 0.466.
    # 7 (0.142), 11 (0.09), 13 (0.077), 17 (0.058), 19 (0.052), 23 (0.043), 29 (0.034), 31 (0.032), 37 (0.027), 41 (0.024), 43 (0.023), 47 (0.021), 53 (0.018), 59 (0.017), 61 (0.016), 67 (0.015), 71 (0.014), 73 (0.0137), 79 (0.0126), 83 (0.012), 89 (0.0112), 97 (0.0103)...
    # Let's just build sets greedily by adding smallest primes until sum > 1, then backtrack or try different combos.
    # Actually, the structure of the covering system is crucial.
    # A covering system needs to cover every integer.
    # If we have a set of primes P, we need to find residues r_p such that union of (r_p mod p) covers 1..L.
    
    # Strategy:
    # 1. Identify a few valid prime sets.
    # 2. For each set, run a heuristic/SA to find residues that maximize L.
    # 3. Return the best L found.

    # Let's generate valid sets.
    # We want to maximize the "coverage power".
    # Smaller primes are generally better for covering consecutive integers because they have smaller periods and thus can be arranged to hit more spots in a short range?
    # Actually, a prime p covers 1/p of the integers.
    # To cover 1..L, we need the union to be dense.
    
    # Let's try to construct sets that include 2, 3, 5 if possible.
    # {2,3,5} is invalid.
    # {2,3,7,11} sum = 0.5+0.333+0.142+0.0909 = 1.066 > 1.
    # {2,3,7,13} sum = 0.5+0.333+0.142+0.0769 = 1.052 > 1.
    # {2,3,7,17} sum = 0.5+0.333+0.142+0.0588 = 1.034 > 1.
    # {2,3,7,19} sum = 0.5+0.333+0.142+0.0526 = 1.028 > 1.
    # {2,3,7,23} sum = 0.5+0.333+0.142+0.0434 = 1.018 > 1.
    # {2,3,7,29} sum = 0.5+0.333+0.142+0.0344 = 1.009 > 1.
    # {2,3,7,31} sum = 0.5+0.333+0.142+0.0322 = 1.007 > 1.
    # {2,3,7,37} sum = 0.5+0.333+0.142+0.0270 = 0.998 <= 1. VALID.
    # So Set 1: {2, 3, 7, 37} + [p for p in primes if p > 37 and sum + 1/p <= 1]
    # Current sum = 0.998. Remaining 0.002.
    # Next prime 41 (0.024) too big.
    # So Set 1 is just {2, 3, 7, 37}.
    
    # What about {2, 3, 11, 13}?
    # 0.5+0.333+0.0909+0.0769 = 1.0008 > 1.
    # {2, 3, 11, 17}?
    # 0.5+0.333+0.0909+0.0588 = 0.9827. Valid.
    # Remaining 0.0173.
    # Next prime 19 (0.052) too big.
    # So Set 2: {2, 3, 11, 17}.
    
    # What about {2, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, ...}?
    # If we drop 3, we save 0.333.
    # Sum(1/p for p>=2) ~ 2.2.
    # We need sum <= 1.
    # Let's just write a function to find good sets.
    
    def find_valid_sets(primes):
        sets = []
        # Try to include 2
        # Greedy: add smallest primes until sum > 1, then remove the largest one added?
        # Or just enumerate a few known good structures.
        
        # Structure 1: Include 2.
        # Try to add 3, 5, 7, 11, 13...
        # We know {2,3,7,37} is valid.
        # We know {2,3,11,17} is valid.
        
        # Let's try to maximize the number of small primes.
        # If we include 2,3, we have 0.833 used.
        # We can add at most one more prime p such that 1/p <= 0.166 => p >= 7.
        # If we add 7, sum=0.975. Remaining 0.025. p >= 40.
        # If we add 11, sum=0.924. Remaining 0.076. p >= 14. So we can add 13?
        # {2,3,11,13} sum = 1.0008 > 1. No.
        # {2,3,11,17} sum = 0.982. Remaining 0.018. p >= 56. So 59?
        # 1/59 = 0.0169. 0.982+0.0169 = 0.999. Valid.
        # So {2,3,11,17,59} is valid.
        
        # Let's systematically generate sets.
        # We will try to include 2.
        # Then try to include 3.
        # Then try to include next smallest primes.
        
        # Actually, let's just use the sets from the prompt's "inspiration" or similar.
        # The prompt's current program uses:
        # [3, 5] + [p for p in primes if p >= 13]
        # [2] + [p for p in primes if p >= 67]
        # [3, 5, 7] + [p for p in primes if p >= 150]
        
        # Let's check validity of these.
        # Set A: {3,5} + p>=13.
        # Sum(1/p for p>=13) = 0.932 (from before).
        # 1/3+1/5 = 0.533.
        # Total = 1.465 > 1. INVALID.
        # So the current program's Set A is invalid?
        # The current program checks validity:
        # `if s <= 1: valid_sets.append(P)`
        # So it would discard Set A.
        
        # Set B: {2} + p>=67.
        # Sum(1/p for p>=67).
        # Sum(1/p for p<67) = 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29+1/31+1/37+1/41+1/43+1/47+1/53+1/59+1/61
        # = 0.5+0.333+0.2+0.142+0.0909+0.0769+0.0588+0.0526+0.0434+0.0344+0.0322+0.0270+0.0243+0.0232+0.0212+0.0188+0.0169+0.0163
        # Sum approx: 0.5+0.333=0.833. +0.2=1.033.
        # So sum(1/p for p<67) > 1.
        # Total sum(1/p for p<1000) ~ 2.2.
        # Sum(1/p for p>=67) = 2.2 - 1.033 = 1.167.
        # Set B sum = 1/2 + 1.167 = 1.667 > 1. INVALID.
        
        # Set C: {3,5,7} + p>=150.
        # Sum(1/p for p<150) is even larger.
        # So all these sets are likely invalid?
        
        # This implies the current program might be running on very few or no valid sets from that list, or my manual sum is wrong.
        # Let's calculate sum(1/p for p in [2..997]) more carefully.
        # It is known that sum_{p<=1000} 1/p is approx 2.198.
        # sum_{p<=30} 1/p = 1/2+1/3+1/5+1/7+1/11+1/13+1/17+1/19+1/23+1/29
        # = 0.5+0.333+0.2+0.1428+0.0909+0.0769+0.0588+0.0526+0.0434+0.0344
        # = 1.533
        # So sum_{p>30} 1/p = 2.198 - 1.533 = 0.665.
        
        # Let's re-evaluate Set B: {2} + p>=67.
        # sum_{p>=67} 1/p = 2.198 - sum_{p<67} 1/p.
        # sum_{p<67} 1/p = sum_{p<=61} 1/p.
        # sum_{p<=30} = 1.533.
        # 37: 0.0270, 41: 0.0243, 43: 0.0232, 47: 0.0212, 53: 0.0188, 59: 0.0169, 61: 0.0163.
        # Sum(37..61) = 0.027+0.024+0.023+0.021+0.019+0.017+0.016 = 0.147.
        # sum_{p<67} = 1.533 + 0.147 = 1.68.
        # sum_{p>=67} = 2.198 - 1.68 = 0.518.
        # Set B sum = 1/2 + 0.518 = 1.018 > 1. Still invalid.
        
        # So the current program's candidate sets are all invalid?
        # If `valid_sets` is empty, it falls back to:
        # `primes_seed = [p for p in primes if p >= 13]`
        # `return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]`
        # This set is {13, 17, ..., 997}.
        # Sum = sum_{p>=13} 1/p = 2.198 - sum_{p<13} 1/p.
        # sum_{p<13} = 1/2+1/3+1/5+1/7+1/11 = 1.266.
        # Sum = 2.198 - 1.266 = 0.932 <= 1. VALID.
        
        # So the current program effectively runs on the set {13, 17, ..., 997} if the others are invalid.
        # But this set does NOT include 2, 3, 5, 7, 11.
        # Covering 1..L with only primes >= 13 is hard because the periods are large.
        # The density of coverage is sum(1/p) = 0.932.
        # But the structure matters.
        
        # To get a higher score (L >= 999), we need a better set of primes.
        # We need to include small primes like 2, 3, 5.
        # Let's find valid sets that include small primes.
        
        # We established {2,3,7,37} is valid (sum 0.998).
        # {2,3,11,17,59} is valid (sum 0.999).
        # {2,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97, ...}
        # If we drop 3, we save 0.333.
        # Sum(1/p for p in {2,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97})
        # = 0.5 + 0.2 + 0.142 + 0.0909 + 0.0769 + 0.0588 + 0.0526 + 0.0434 + 0.0344 + 0.0322 + 0.0270 + 0.0243 + 0.0232 + 0.0212 + 0.0188 + 0.0169 + 0.0163 + 0.0149 + 0.0140 + 0.0137 + 0.0126 + 0.0120 + 0.0112 + 0.0103
        # Sum approx:
        # 0.5+0.2=0.7
        # +0.142=0.842
        # +0.091=0.933
        # +0.077=1.010 > 1.
        # So we can't include 13 if we have 2,5,7,11.
        # {2,5,7,11} sum = 0.933.
        # Remaining 0.067.
        # Next prime 13 (0.077) too big.
        # 17 (0.058) fits.
        # {2,5,7,11,17} sum = 0.933+0.058=0.991.
        # Remaining 0.009.
        # 19 (0.052) too big.
        # So {2,5,7,11,17} is valid.
        
        # Let's create a list of valid sets to try.
        sets_to_try = []
        
        # Set 1: {2, 3, 7, 37}
        P1 = [2, 3, 7, 37]
        # Can we add more?
        # Sum = 0.998. No.
        sets_to_try.append(P1)
        
        # Set 2: {2, 3, 11, 17, 59}
        P2 = [2, 3, 11, 17, 59]
        # Sum = 0.999. No.
        sets_to_try.append(P2)
        
        # Set 3: {2, 5, 7, 11, 17}
        P3 = [2, 5, 7, 11, 17]
        # Sum = 0.991.
        # Can we add 19? 0.991+0.052 > 1. No.
        sets_to_try.append(P3)
        
        # Set 4: {3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, ...}
        # Drop 2. Save 0.5.
        # Sum(1/p for p in {3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97})
        # = 1.010 (from before, minus 0.5? No, the 1.010 was for {2,5,7,11,13}.
        # Let's recalc sum for {3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97}
        # Sum(1/p for p in {3..97}) = Sum(1/p for p in {2..97}) - 1/2.
        # Sum(1/p for p in {2..97}) = 2.198 - sum(1/p for p in {101..997}).
        # This is getting complicated. Let's just use the program to check.
        
        # Let's define a function to generate sets by greedily adding primes.
        def generate_set(start_primes, primes_list):
            P = list(start_primes)
            s = sum(Fraction(1, p) for p in P)
            for p in primes_list:
                if p in P:
                    continue
                if s + Fraction(1, p) <= 1:
                    P.append(p)
                    s += Fraction(1, p)
            return P

        # Generate sets with different starting points
        # Start with 2
        P_start_2 = [2]
        # Try to add 3
        P_2_3 = generate_set([2, 3], primes)
        sets_to_try.append(P_2_3)
        
        # Try to add 5 instead of 3
        P_2_5 = generate_set([2, 5], primes)
        sets_to_try.append(P_2_5)
        
        # Start with 3
        P_3 = generate_set([3], primes)
        sets_to_try.append(P_3)
        
        # Start with 3, 5
        P_3_5 = generate_set([3, 5], primes)
        sets_to_try.append(P_3_5)
        
        # Start with 3, 5, 7
        P_3_5_7 = generate_set([3, 5, 7], primes)
        sets_to_try.append(P_3_5_7)
        
        # Remove duplicates (as sets)
        unique_sets = []
        seen = set()
        for P in sets_to_try:
            t = tuple(sorted(P))
            if t not in seen:
                seen.add(t)
                unique_sets.append(P)
        
        # Filter valid sets
        valid_sets = []
        for P in unique_sets:
            if is_valid_set(P):
                valid_sets.append(P)
        
        if not valid_sets:
            # Fallback
            primes_seed = [p for p in primes if p >= 13]
            return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]

        best_L_global = 0
        best_result_global = []
        
        L_MAX = 10000
        
        # Allocate time
        time_per_set = (deadline - time.time()) / len(valid_sets) - 2 # 2s margin per set
        
        for P in valid_sets:
            if time.time() > deadline - 5:
                break
            
            set_start = time.time()
            set_deadline = set_start + time_per_set
            
            P_sorted = sorted(P)
            
            # Greedy Init
            residues = {p: 0 for p in P}
            cover_count = [0] * (L_MAX + 1)
            
            def update_cover(p, r, delta):
                first = r if r > 0 else p
                if first > L_MAX:
                    return
                for n in range(first, L_MAX + 1, p):
                    cover_count[n] += delta

            assigned_primes = set()
            for n in range(1, L_MAX + 1):
                if cover_count[n] == 0:
                    chosen_p = None
                    for p in P_sorted:
                        if p not in assigned_primes:
                            chosen_p = p
                            break
                    if chosen_p is None:
                        break
                    r = n % chosen_p
                    residues[chosen_p] = r
                    assigned_primes.add(chosen_p)
                    update_cover(chosen_p, r, 1)
                if time.time() > set_deadline:
                    break

            def get_L():
                for n in range(1, L_MAX + 1):
                    if cover_count[n] == 0:
                        return n - 1
                return L_MAX

            current_L = get_L()
            best_set_L = current_L
            best_set_residues = residues.copy()
            
            # SA
            T = 5.0
            T_min = 0.1
            cooling = 0.999
            
            while time.time() < set_deadline:
                p = random.choice(P)
                r_old = residues[p]
                r_new = random.randint(0, p - 1)
                
                if r_new == r_old:
                    continue
                
                update_cover(p, r_old, -1)
                update_cover(p, r_new, 1)
                
                new_L = get_L()
                
                if new_L > current_L:
                    current_L = new_L
                    residues[p] = r_new
                    if current_L > best_set_L:
                        best_set_L = current_L
                        best_set_residues = residues.copy()
                else:
                    delta = new_L - current_L
                    if delta == 0:
                        if random.random() < 0.5:
                            residues[p] = r_new
                    else:
                        prob = math.exp(delta / T)
                        if random.random() < prob:
                            residues[p] = r_new
                            current_L = new_L
                        else:
                            update_cover(p, r_new, -1)
                            update_cover(p, r_old, 1)
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            if best_set_L > best_L_global:
                best_L_global = best_set_L
                best_result_global = [[p, best_set_residues[p]] for p in best_set_residues]
                
                if best_L_global >= 5000:
                    break

        if not best_result_global:
            primes_seed = [p for p in primes if p >= 13]
            return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
            
        return best_result_global
