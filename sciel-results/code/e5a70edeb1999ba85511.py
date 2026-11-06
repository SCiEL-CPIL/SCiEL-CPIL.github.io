"""Program to find optimal A, B for pairwise-coprime sumset maximization."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations, product

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _pairwise_coprime(vals):
    vs = list(set(vals))
    n = len(vs)
    for i in range(n):
        for j in range(i+1, n):
            if _gcd(vs[i], vs[j]) != 1:
                return False
    return True

def _eval(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if len(sums) <= 1:
        return len(sums)
    if _pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def _is_valid(A, B):
    """Check if A and B are valid: 3 distinct integers in [1,128]."""
    if len(A) != 3 or len(B) != 3:
        return False
    if len(set(A)) != 3 or len(set(B)) != 3:
        return False
    if any(x < 1 or x > 128 for x in A) or any(x < 1 or x > 128 for x in B):
        return False
    return True

def _search():
    """Search for optimal A, B using constraint-based backtracking over sum values."""
    deadline = time.time() + 75  # 75s budget, leaving 15s safety margin
    
    # Precompute sieve for primality
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    
    # Generate candidate sum values: primes and prime powers in [2, 256]
    # A sum s can be decomposed as s = a + b with a,b in [1,128]
    # i.e., 2 <= s <= 256
    candidates = []
    for s in range(2, 257):
        # Include primes
        if is_prime[s]:
            candidates.append(s)
        else:
            # Include prime powers p^k where p is prime and k >= 2
            # and s is coprime to all other candidates we might pick
            # For now, include all numbers that are powers of a single prime
            n = s
            p = None
            while n % 2 == 0:
                n //= 2
                if p is None:
                    p = 2
                elif p != 2:
                    break
            else:
                if n == 1:
                    # power of 2
                    candidates.append(s)
                    continue
            n = s
            p = None
            for prime in range(3, int(math.isqrt(s)) + 1, 2):
                if s % prime == 0:
                    if p is not None and p != prime:
                        break
                    p = prime
                    n = s
                    while n % prime == 0:
                        n //= prime
                    if n == 1:
                        candidates.append(s)
                        break
            else:
                # Check if s itself is a prime power of a prime > sqrt(s)
                # Actually if no small factor found and s > 1, s might be prime (handled) or p^1
                pass
    
    # Remove duplicates
    candidates = list(set(candidates))
    candidates.sort()
    
    # Build decomposition table: for each sum s, list of (a, b) pairs with a,b in [1,128]
    decomp = {}
    for s in candidates:
        pairs = []
        for a in range(max(1, s - 128), min(128, s - 1) + 1):
            b = s - a
            if 1 <= b <= 128:
                pairs.append((a, b))
        decomp[s] = pairs
    
    # Build compatibility graph: two sums s1, s2 are compatible if gcd(s1, s2) == 1
    # Also need them to be "co-decomposable" with small A, B sets
    
    # Strategy: Backtrack to select 9 pairwise-coprime sums, then check if they
    # can be covered by a 3x3 grid (3 A values, 3 B values)
    
    # First, find all maximal cliques in the coprime graph among candidates
    # This is expensive, so we'll use a heuristic approach
    
    # Phase 1: Try to find 9 pairwise-coprime sums that form a 3x3 grid
    # A 3x3 grid has sums a_i + b_j for i,j in {0,1,2}
    # Key property: (a_i + b_j) - (a_i + b_k) = b_j - b_k
    # and (a_i + b_j) - (a_k + b_j) = a_i - a_k
    
    # Let's enumerate A sets and for each, find compatible B sets
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _eval(best_A, best_B)
    
    # Phase 1: Systematic search focusing on making sums prime
    # For each A, find B values that make all sums prime, then check pairwise coprime
    
    # Precompute for each a in [1,128], which b values make a+b prime
    prime_b_for_a = {}
    for a in range(1, 129):
        prime_b_for_a[a] = []
        for b in range(1, 129):
            s = a + b
            if 2 <= s <= 256 and is_prime[s]:
                prime_b_for_a[a].append(b)
    
    # Try all A from a structured set
    # Focus on A values where the differences a2-a1, a3-a1, a3-a2 are "nice"
    a_range = range(1, 61)
    
    # Phase 1a: Try A with small values, B chosen to make sums prime
    for a1 in a_range:
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, min(a1 + 30, 129)):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, min(a2 + 30, 129)):
                if time.time() > deadline:
                    break
                
                A = [a1, a2, a3]
                
                # Find B values that make all three sums prime
                common_b = set(prime_b_for_a[a1]) & set(prime_b_for_a[a2]) & set(prime_b_for_a[a3])
                
                if len(common_b) < 3:
                    continue
                
                b_list = sorted(common_b)
                
                # Try combinations of 3 from b_list
                if len(b_list) <= 20:
                    b_combos = list(combinations(b_list, 3))
                else:
                    b_combos = [tuple(rng.sample(b_list, 3)) for _ in range(100)]
                    b_combos = list(set(b_combos))
                
                for B_tuple in b_combos:
                    if time.time() > deadline:
                        break
                    B = list(B_tuple)
                    score = _eval(A, B)
                    if score > best_score:
                        best_score = score
                        best_A = A[:]
                        best_B = B[:]
                        if best_score >= 9:
                            return best_A, best_B
    
    # Phase 1b: Try A with larger spread
    for a1 in range(1, 30):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 5, min(a1 + 50, 129)):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 5, min(a2 + 50, 129)):
                if time.time() > deadline:
                    break
                
                A = [a1, a2, a3]
                
                # Find B values that make all three sums prime
                common_b = set(prime_b_for_a[a1]) & set(prime_b_for_a[a2]) & set(prime_b_for_a[a3])
                
                if len(common_b) < 3:
                    # Try relaxed: at least 2 sums prime
                    # Or try all b and check pairwise coprime
                    b_list = list(range(1, 129))
                else:
                    b_list = sorted(common_b)
                
                if len(b_list) <= 15:
                    b_combos = list(combinations(b_list, 3))
                else:
                    b_combos = [tuple(rng.sample(b_list, 3)) for _ in range(50)]
                    b_combos = list(set(b_combos))
                
                for B_tuple in b_combos:
                    if time.time() > deadline:
                        break
                    B = list(B_tuple)
                    score = _eval(A, B)
                    if score > best_score:
                        best_score = score
                        best_A = A[:]
                        best_B = B[:]
                        if best_score >= 9:
                            return best_A, best_B
    
    # Phase 2: Backtracking over sum values
    # Select pairwise-coprime sums, then try to fit them into a 3x3 grid
    
    # Build coprime compatibility matrix
    n_cands = len(candidates)
    coprime = [[False] * n_cands for _ in range(n_cands)]
    for i in range(n_cands):
        for j in range(i + 1, n_cands):
            coprime[i][j] = coprime[j][i] = (_gcd(candidates[i], candidates[j]) == 1)
    
    # For backtracking, we need to check if a set of sums can be a 3x3 grid
    # A set of 9 sums {s_ij} is a 3x3 grid if there exist a_1,a_2,a_3,b_1,b_2,b_3
    # such that s_ij = a_i + b_j for all i,j
    
    def _can_form_grid(sums_set):
        """Check if a set of 9 sums can form a 3x3 grid."""
        if len(sums_set) != 9:
            return False
        
        sums = list(sums_set)
        # Try all permutations of assigning sums to grid positions
        # Too expensive, so use a heuristic:
        # In a 3x3 grid, s_11 + s_22 + s_33 = s_12 + s_21 + s_33 = ... (many relations)
        # Actually: s_ij + s_kl - s_il - s_kj = 0 for all i,j,k,l
        
        # Simpler: try to find a_1, b_1 such that a_1 + b_1 is in sums
        # Then a_2 = some other sum - b_1, etc.
        
        # For each pair (s1, s2) as potential (a1+b1, a1+b2):
        # b2 - b1 = s2 - s1
        # For each s3 in sums: a2 + b1 = s3, so a2 = s3 - b1
        # Then a2 + b2 should also be in sums: (s3 - b1) + (b1 + (s2-s1)) = s3 + s2 - s1
        
        for s1 in sums:
            for s2 in sums:
                if s1 == s2:
                    continue
                db = s2 - s1  # b2 - b1
                if db == 0:
                    continue
                
                # Try each possible b1 in [1, 128]
                # a1 = s1 - b1 must be in [1, 128]
                min_b1 = max(1, s1 - 128)
                max_b1 = min(128, s1 - 1)
                
                for b1 in range(min_b1, max_b1 + 1):
                    a1 = s1 - b1
                    b2 = b1 + db
                    if not (1 <= b2 <= 128):
                        continue
                    
                    # Now we have a1, b1, b2. Need to find a2, a3, b3
                    # a2 + b1 must be in sums, a3 + b1 must be in sums
                    # And a2 + b2, a3 + b2, a2 + b3, a3 + b3 must be in sums
                    
                    remaining = [s for s in sums if s != s1 and s != s2]
                    # a2 + b1 in remaining, a3 + b1 in remaining
                    # a2 + b2 in remaining, a3 + b2 in remaining
                    # a2 + b3 in sums, a3 + b3 in sums
                    
                    # a2 = (a2+b1) - b1
                    # a3 = (a3+b1) - b1
                    
                    # Try pairs from remaining as (a2+b1, a3+b1)
                    for i in range(len(remaining)):
                        for j in range(i + 1, len(remaining)):
                            s_a2b1 = remaining[i]
                            s_a3b1 = remaining[j]
                            a2 = s_a2b1 - b1
                            a3 = s_a3b1 - b1
                            if not (1 <= a2 <= 128 and 1 <= a3 <= 128):
                                continue
                            if a2 == a1 or a3 == a1 or a2 == a3:
                                continue
                            
                            # Check a2+b2 and a3+b2
                            s_a2b2 = a2 + b2
                            s_a3b2 = a3 + b2
                            if s_a2b2 not in sums_set or s_a3b2 not in sums_set:
                                continue
                            if s_a2b2 == s1 or s_a2b2 == s2 or s_a2b2 == s_a2b1 or s_a2b2 == s_a3b1:
                                continue
                            if s_a3b2 == s1 or s_a3b2 == s2 or s_a3b2 == s_a2b1 or s_a3b2 == s_a3b1:
                                continue
                            
                            # Remaining 3 sums must be a2+b3, a3+b3, and one of the above
                            # Actually we have 9 sums total:
                            # s1=a1+b1, s2=a1+b2, s_a2b1=a2+b1, s_a3b1=a3+b1,
                            # s_a2b2=a2+b2, s_a3b2=a3+b2, and 3 more: a1+b3, a2+b3, a3+b3
                            
                            # The 3 remaining sums are the ones not yet assigned
                            used = {s1, s2, s_a2b1, s_a3b1, s_a2b2, s_a3b2}
                            remaining3 = [s for s in sums_set if s not in used]
                            if len(remaining3) != 3:
                                continue
                            
                            # remaining3 = {a1+b3, a2+b3, a3+b3}
                            # a1+b3 = a1 + b3
                            # a2+b3 = a2 + b3
                            # a3+b3 = a3 + b3
                            # So (a2+b3) - (a1+b3) = a2 - a1
                            # and (a3+b3) - (a1+b3) = a3 - a1
                            
                            # Try each element of remaining3 as a1+b3
                            for s_a1b3 in remaining3:
                                b3 = s_a1b3 - a1
                                if not (1 <= b3 <= 128):
                                    continue
                                if b3 == b1 or b3 == b2:
                                    continue
                                
                                s_a2b3 = a2 + b3
                                s_a3b3 = a3 + b3
                                if s_a2b3 not in sums_set or s_a3b3 not in sums_set:
                                    continue
                                if s_a2b3 != s_a1b3 and s_a3b3 != s_a1b3:
                                    # Check they're the other two
                                    if {s_a2b3, s_a3b3} != set(remaining3) - {s_a1b3}:
                                        continue
                                
                                # Verify all distinct
                                all_sums = [s1, s2, s_a2b1, s_a3b1, s_a2b2, s_a3b2, s_a1b3, s_a2b3, s_a3b3]
                                if len(set(all_sums)) != 9:
                                    continue
                                
                                # Verify all in [2, 256]
                                if any(s < 2 or s > 256 for s in all_sums):
                                    continue
                                
                                return ([a1, a2, a3], [b1, b2, b3])
        
        return None
    
    # Phase 2: Backtracking to find 9 pairwise-coprime sums that form a grid
    # Use the candidate list and coprime matrix
    
    # Sort candidates by degree in coprime graph (most connected first)
    degrees = [sum(coprime[i]) for i in range(n_cands)]
    order = sorted(range(n_cands), key=lambda i: -degrees[i])
    
    # Backtracking search for a clique of size 9
    # This is expensive, so limit the search
    
    found_grid = None
    backtrack_limit = 100000
    count = 0
    
    def _backtrack_clique(current, start_idx):
        nonlocal found_grid, count
        if time.time() > deadline:
            return
        if count > backtrack_limit:
            return
        
        if len(current) == 9:
            result = _can_form_grid(set(candidates[i] for i in current))
            if result:
                found_grid = result
                return
            return
        
        for idx in range(start_idx, n_cands):
            if count > backtrack_limit:
                return
            count += 1
            i = order[idx]
            if not all(coprime[i][j] for j in current):
                continue
            current.append(i)
            _backtrack_clique(current, idx + 1)
            if found_grid:
                return
            current.pop()
    
    # Start backtracking with a few different starting points
    for start_i in order[:50]:
        if found_grid or time.time() > deadline or count > backtrack_limit:
            break
        _backtrack_clique([start_i], order.index(start_i) + 1)
    
    if found_grid:
        A, B = found_grid
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
    
    # Phase 3: Random search
    for _ in range(100000):
        if time.time() > deadline:
            break
        A = list(rng.sample(range(1, 129), 3))
        B = list(rng.sample(range(1, 129), 3))
        score = _eval(A, B)
        if score > best_score:
            best_score = score
            best_A = A[:]
            best_B = B[:]
            if best_score >= 9:
                return best_A, best_B
    
    # Phase 4: Local search
    for _ in range(100000):
        if time.time() > deadline:
            break
        if rng.random() < 0.5:
            idx = rng.randint(0, 2)
            new_val = rng.randint(1, 128)
            if new_val not in best_A:
                old = best_A[idx]
                best_A[idx] = new_val
                score = _eval(best_A, best_B)
                if score >= best_score:
                    best_score = score
                    if best_score >= 9:
                        return best_A, best_B
                else:
                    best_A[idx] = old
        else:
            idx = rng.randint(0, 2)
            new_val = rng.randint(1, 128)
            if new_val not in best_B:
                old = best_B[idx]
                best_B[idx] = new_val
                score = _eval(best_A, best_B)
                if score >= best_score:
                    best_score = score
                    if best_score >= 9:
                        return best_A, best_B
                else:
                    best_B[idx] = old
    
    return best_A, best_B

# Pre-compute the best answer
_best_result = _search()

def construct():
    return _best_result
# EVOLVE-BLOCK-END
