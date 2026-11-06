import time
import random
from math import gcd
from itertools import combinations, product

def _pairwise_coprime_count(sums):
    """Count distinct sums that are pairwise coprime."""
    distinct = sorted(set(sums))
    count = 0
    for i in range(len(distinct)):
        ok = True
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                ok = False
                break
        if ok:
            # Check all pairs from i onward with i
            pass
    # Better: find max subset that is pairwise coprime
    # But the task says "ALL DISTINCT values in A+B must be pairwise coprime"
    # So we need ALL distinct sums to be pairwise coprime
    # The score is the cardinality of A+B (number of distinct sums)
    # but only if they are ALL pairwise coprime
    # Wait, re-reading: "All DISTINCT values in A+B must be pairwise coprime. Maximize the cardinality of A+B"
    # So the constraint is that all distinct sums must be pairwise coprime, and we maximize the count.
    # If not all are pairwise coprime, the result is invalid (score 0)
    
    # Let me re-read: "Return two lists A,B, each containing exactly 3 distinct integers in [1,128]. 
    # All DISTINCT values in A+B must be pairwise coprime. Maximize the cardinality of A+B"
    
    # So the constraint is a hard constraint: all distinct sums must be pairwise coprime.
    # The objective is to maximize the number of distinct sums.
    
    # For evaluation, I think the score is just the number of distinct sums, 
    # but only if the pairwise coprime constraint is satisfied.
    # Otherwise score is 0.
    
    return len(distinct)

def _is_valid_pairwise_coprime(sums):
    """Check if all distinct sums are pairwise coprime."""
    distinct = sorted(set(sums))
    for i in range(len(distinct)):
        for j in range(i+1, len(distinct)):
            if gcd(distinct[i], distinct[j]) != 1:
                return False
    return True

def _score(A, B):
    """Score: number of distinct sums if all pairwise coprime, else 0."""
    sums = [a + b for a in A for b in B]
    if _is_valid_pairwise_coprime(sums):
        return len(set(sums))
    return 0

def _evaluate(A, B):
    """Evaluate and return (score, sums_list)."""
    sums = [a + b for a in A for b in B]
    if _is_valid_pairwise_coprime(sums):
        return len(set(sums)), sums
    return 0, sums

def _find_AB_for_target_sums(target_sums):
    """
    Given a set of target sums, try to find A and B (each 3 distinct in [1,128])
    such that A+B contains exactly these target sums (or a subset thereof).
    
    Strategy: For each target sum s, we need a+b=s for some a in A, b in B.
    We'll try small values for A and see if B can be determined.
    """
    targets = sorted(target_sums)
    
    # Try all combinations of 3 values for A from a small range
    # The sums range from min(A)+min(B) to max(A)+max(B)
    # If targets are in [4, 20], try A values in [1, 10]
    
    min_target = min(targets)
    max_target = max(targets)
    
    # A elements must be at least 1, B elements at least 1
    # So a <= s-1 for each target s
    # Also a >= s - 127 for each target s (since b <= 128)
    
    # Try A values from 1 to min(128, max_target - 1)
    # But we want small A to keep B reasonable
    
    max_A_val = min(50, max_target - 2)  # Leave room for B >= 1
    min_A_val = max(1, min_target - 128)
    
    if max_A_val < min_A_val:
        return None
    
    # Try combinations of 3 values for A
    a_range = list(range(min_A_val, max_A_val + 1))
    
    for a1, a2, a3 in combinations(a_range, 3):
        A = [a1, a2, a3]
        
        # For each target sum, find which a it could pair with
        # and what b would be
        # b = s - a, must be in [1, 128] and distinct
        
        # Build a mapping: for each target s, find possible (a, b) pairs
        possible_B = set()
        valid = True
        
        for s in targets:
            found_for_s = False
            for a in A:
                b = s - a
                if 1 <= b <= 128:
                    possible_B.add(b)
                    found_for_s = True
            if not found_for_s:
                valid = False
                break
        
        if not valid:
            continue
        
        # Now try to pick 3 distinct B values from possible_B
        # such that A+B covers all targets
        B_candidates = sorted(possible_B)
        
        if len(B_candidates) < 3:
            continue
        
        # Try combinations of 3 from B_candidates
        for b1, b2, b3 in combinations(B_candidates, 3):
            B = [b1, b2, b3]
            
            # Compute A+B
            sums = set(a + b for a in A for b in B)
            
            # Check if sums == targets (or at least targets is subset)
            # Actually we want sums to be exactly the pairwise coprime set
            if sums == set(targets):
                return (A, B)
            
            # Or if targets is a subset of sums, but sums must still be pairwise coprime
            if set(targets).issubset(sums) and _is_valid_pairwise_coprime(list(sums)):
                # This is also valid, but we prefer exact match
                pass
    
    return None

def _generate_coprime_targets(k, min_val=4, max_val=200):
    """Generate k pairwise coprime numbers in [min_val, max_val]."""
    # Use primes and prime powers
    primes = [p for p in range(2, max_val + 1) if all(p % d != 0 for d in range(2, int(p**0.5) + 1))]
    
    # Start with small primes
    candidates = primes[:50]
    
    # Try to pick k pairwise coprime numbers
    # Since they're all prime, they're automatically pairwise coprime
    # But we can also use prime powers for more variety
    
    # For now, just pick k distinct primes
    if len(primes) < k:
        return None
    
    return primes[:k]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    best_score = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    def try_candidate(A, B):
        nonlocal best_score, best_A, best_B
        score, _ = _evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 1: Direct search for high-cardinality pairwise coprime sumsets
    # Key insight: if we want k distinct sums, we need k pairwise coprime numbers
    # in the range [2, 256] (since min sum is 1+1=2, max is 128+128=256)
    
    # Precompute all pairwise coprime subsets of size k from [2, 256]
    # For k=6,7,8,9,10, try to find A,B that produce them
    
    # First, try to find a good set of target sums
    # Then work backwards to find A and B
    
    # Generate candidate target sets
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Generate many candidate target sets and try to realize them
    # Focus on sums in a reasonable range [4, 30] to keep A,B small
    
    # Strategy: enumerate pairs (A, B) where A and B are small
    # and check if the sumset is pairwise coprime with high cardinality
    
    # More efficient: iterate over A values, for each compute what B values
    # would give pairwise coprime sums
    
    # Let's do a smarter search:
    # Fix A = [a1, a2, a3] with small values
    # For each candidate b in B, the sums are a1+b, a2+b, a3+b
    # We need all 9 sums (or however many are distinct) to be pairwise coprime
    
    # Key constraint: if a1+b and a2+b share a factor, that's bad
    # gcd(a1+b, a2+b) = gcd(a1+b, a2-a1)
    # So for each pair (a_i, a_j) in A, and each b, we need
    # gcd(a_i + b, a_j + b) = gcd(a_i + b, a_j - a_i) = 1
    
    # This means: for each b, a_i + b must be coprime to (a_j - a_i) for all j
    
    # Let's exploit this structure
    
    def search_with_fixed_A(A, deadline):
        """Given A, find B that maximizes pairwise coprime sumset."""
        a1, a2, a3 = A[0], A[1], A[2]
        
        # Differences
        d12 = a2 - a1
        d13 = a3 - a1
        d23 = a3 - a2
        
        # For a given b, the sums are:
        # s1 = a1 + b
        # s2 = a2 + b = s1 + d12
        # s3 = a3 + b = s1 + d13
        
        # We need gcd(s1, s2) = gcd(s1, d12) = 1
        # gcd(s1, s3) = gcd(s1, d13) = 1
        # gcd(s2, s3) = gcd(s2, d23) = gcd(s1 + d12, d23) = 1
        
        # Also need cross terms between different b values
        
        # Let's just try b values and check
        
        best_local_score = 0
        best_local_B = None
        
        # Try b values from 1 to 128
        # For efficiency, precompute which b values give "good" individual triples
        good_b_candidates = []
        for b in range(1, 129):
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            
            # Check if these three are pairwise coprime
            if gcd(s1, s2) != 1 or gcd(s1, s3) != 1 or gcd(s2, s3) != 1:
                continue
            
            # Also check they're in valid range
            if s1 < 2 or s3 > 256:
                continue
            
            good_b_candidates.append(b)
        
        if len(good_b_candidates) < 3:
            return 0, None
        
        # Now try combinations of 3 from good_b_candidates
        # But we need to check cross-term coprimality
        
        # Limit to top candidates for efficiency
        # Prioritize b values that give sums with small prime factors (harder to share)
        # Actually, prioritize b values that give prime sums
        
        # Sort by number of prime factors of the sums (fewer is better)
        def count_prime_factors(n):
            count = 0
            d = 2
            while d * d <= n:
                while n % d == 0:
                    count += 1
                    n //= d
                d += 1
            if n > 1:
                count += 1
            return count
        
        # Score each candidate b by how "prime" its sums are
        scored_b = []
        for b in good_b_candidates:
            s1 = a1 + b
            s2 = a2 + b
            s3 = a3 + b
            pf = count_prime_factors(s1) + count_prime_factors(s2) + count_prime_factors(s3)
            scored_b.append((pf, b))
        
        scored_b.sort()
        
        # Try top 50 candidates
        top_b = [b for _, b in scored_b[:50]]
        
        for b1, b2, b3 in combinations(top_b, 3):
            B = [b1, b2, b3]
            
            # Compute all sums
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) < best_local_score:
                continue
            
            if _is_valid_pairwise_coprime(list(sums)):
                if len(sums) > best_local_score:
                    best_local_score = len(sums)
                    best_local_B = B
        
        return best_local_score, best_local_B
    
    # Try various A sets
    # A should have elements that create "nice" differences
    # Small differences are better (fewer prime factors to avoid)
    
    # Generate candidate A sets
    A_candidates = []
    
    # Consecutive or near-consecutive
    for start in range(1, 20):
        A_candidates.append([start, start+1, start+2])
        A_candidates.append([start, start+1, start+3])
        A_candidates.append([start, start+2, start+3])
        A_candidates.append([start, start+2, start+4])
        A_candidates.append([start, start+3, start+4])
    
    # Spread out
    for a1 in range(1, 15):
        for a2 in range(a1+2, 30):
            for a3 in range(a2+2, 50):
                A_candidates.append([a1, a2, a3])
    
    # Remove duplicates and limit
    A_candidates = list(set(tuple(A) for A in A_candidates))
    A_candidates = [list(A) for A in A_candidates[:500]]
    
    for A in A_candidates:
        if time.time() > deadline:
            break
        
        score, B = search_with_fixed_A(A, deadline)
        if score > best_score and B is not None:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 2: Random restarts with local improvement
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        
        A = random.sample(range(1, 30), 3)
        A.sort()
        score, B = search_with_fixed_A(A, deadline)
        if score > best_score and B is not None:
            best_score = score
            best_A = A[:]
            best_B = B[:]
    
    # Strategy 3: Local search on the best found
    if best_score >= 5:
        def local_search(A, B, max_iters=20000):
            nonlocal best_score, best_A, best_B
            current_A = A[:]
            current_B = B[:]
            current_score, _ = _evaluate(current_A, current_B)
            
            for iteration in range(max_iters):
                if time.time() > deadline:
                    break
                
                improved = False
                # Try changing one element in A or B
                for set_idx in range(2):
                    for pos in range(3):
                        old_val = current_A[pos] if set_idx == 0 else current_B[pos]
                        # Try nearby values and some random ones
                        candidates = set()
                        for delta in range(-5, 6):
                            new_val = old_val + delta
                            if 1 <= new_val <= 128 and new_val != old_val:
                                candidates.add(new_val)
                        # Add some random candidates
                        for _ in range(10):
                            candidates.add(random.randint(1, 128))
                        
                        for new_val in sorted(candidates):
                            if set_idx == 0:
                                if new_val in current_A and new_val != old_val:
                                    continue
                                new_A = current_A[:]
                                new_A[pos] = new_val
                                new_B = current_B
                            else:
                                if new_val in current_B and new_val != old_val:
                                    continue
                                new_A = current_A
                                new_B = current_B[:]
                                new_B[pos] = new_val
                            
                            score, _ = _evaluate(new_A, new_B)
                            if score > current_score:
                                current_A = new_A
                                current_B = new_B
                                current_score = score
                                if score > best_score:
                                    best_score = score
                                    best_A = current_A[:]
                                    best_B = current_B[:]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            return current_score
        
        local_search(best_A, best_B)
    
    # Final verification
    final_score, _ = _evaluate(best_A, best_B)
    if final_score < 5:
        # Fallback to known valid solution
        return [[1, 7, 13], [4, 10, 16]]
    
    return [best_A, best_B]
