import math
import time
import sys

def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    """
    deadline = time.time() + 80
    sys.setrecursionlimit(10000)
    
    # Candidate universe: 2..256
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Map value to index
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Build conflict graph:
    # Two candidates x, y are in conflict if there exists a third candidate z (z != x, z != y)
    # such that x | y*z or y | x*z.
    # If we select both x and y, and also select any z that causes the conflict, we have a violation.
    # Wait, the condition is: for ALL distinct a,b,c in S, a !| b*c.
    # If x, y, z are in S, and x | y*z, then it's invalid.
    # So if x, y are in S, we cannot include ANY z such that x | y*z or y | x*z.
    # This means x and y are "incompatible" with a set of other nodes.
    # This is not a standard pairwise conflict graph where edge (x,y) means they can't both be in S.
    # Here, edge (x,y) doesn't mean x,y can't be in S. It means if x,y are in S, then certain z's are forbidden.
    # This is a 3-uniform hypergraph independent set problem (or similar).
    # Specifically, a triple {a,b,c} is "bad" if a|bc or b|ac or c|ab.
    # We want to find a subset S such that no triple from S is bad.
    # This is exactly the Maximum Independent Set in a 3-uniform hypergraph where hyperedges are the bad triples.
    # Or, equivalently, we can define a graph on the candidates where... no, it's not a graph.
    
    # Let's precompute the "badness" of triples or use a different approach.
    # Since N=255, the total number of triples is C(255, 3) ~ 2.7M.
    # We can iterate through all triples and mark them as bad.
    # Then we need to find the largest subset with no bad triple.
    
    # Optimization:
    # For a fixed pair (a, b), which c's are bad?
    # c is bad if a | b*c, b | a*c, or c | a*b.
    # Let's precompute for each pair (a, b) the set of c's that form a bad triple.
    # But storing sets for all pairs is heavy.
    
    # Alternative: Branch and Bound on the hypergraph.
    # State: (candidates_available, current_solution)
    # Upper bound: len(candidates_available) + len(current_solution)
    # If upper bound <= best, prune.
    
    # To make B&B efficient, we need a good ordering and strong pruning.
    # Heuristic: Pick a candidate x.
    # If we include x, we must remove all y, z such that {x,y,z} is bad.
    # This is complex.
    
    # Let's try a different perspective.
    # The condition "a does not divide b*c" for all distinct a,b,c.
    # If we pick a set of numbers that are all "large" (e.g., > 128), does it help?
    # If a > 128, b > 128, c > 128, then b*c > 16384.
    # a divides b*c?
    # If a is prime > 128, it divides b*c only if a divides b or a divides c.
    # Since b,c < 256 and a > 128, a cannot divide b or c (unless b=c=a, but they are distinct).
    # So any set of distinct primes in (128, 256] is a valid set!
    # How many primes in (128, 256]?
    # Primes: 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251.
    # Count: 23 primes.
    # Can we add more numbers?
    # What if we add a composite number x > 128?
    # If x is composite, say x = p*q.
    # If p, q are in the set, and x is in the set, then x | p*q is true (x = p*q).
    # So we cannot have x and its factors p, q in the set.
    # If we stick to primes > 128, we are safe from this specific issue.
    # But what if we mix small and large?
    
    # Let's test the set of primes > 128 first.
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
        
    primes_large = [x for x in candidates if is_prime(x) and x > 128]
    # Size: 23.
    # Current best seed score 1.0556 corresponds to raw value?
    # Seed returns range(129, 257) size 128? No, seed does greedy removal.
    # If seed score is 1.0556, and base is 128? 128 * 1.0556 = 135.1.
    # Or maybe the seed returns ~135 items.
    # My prime set is only 23. That's much worse.
    # The "antichain" property (no x divides y) is not required, but it helps.
    # In [129, 256], if x divides y, then y=2x.
    # 129*2 = 258 > 256. So [129, 256] is an antichain.
    # For an antichain, a | b*c implies a/gcd(a,b) | c.
    # Since a,b > 128, gcd(a,b) is small? Not necessarily.
    # But if a,b,c are all in [129, 256], can a | b*c?
    # b*c is at most 256*255 ~ 65000.
    # a is at least 129.
    # So a | b*c is possible.
    # Example: a=144, b=144 (distinct required), b=145, c=144? No distinct.
    # Let's check the seed's result size.
    # The seed starts with 128 elements and removes conflicts.
    # If it removes few, size is close to 128.
    
    # Let's implement a smarter B&B or Local Search on the full set [2, 256].
    # Or better, start with [129, 256] and try to add elements from [2, 128].
    # Adding small elements is risky because small a easily divides b*c.
    # But small a is "safe" as a divisor? No, small a is a "bad" divisor.
    # We want to AVOID having small a in the set if it divides many products.
    # Actually, if a is small, it's likely to divide b*c. So we should EXCLUDE small numbers?
    # If we exclude small numbers, we are left with large numbers.
    # Large numbers are "safe" divisors (hard to divide b*c).
    # But large numbers are "bad" as multiplicands?
    # If x is large, and a is in set, a | x*c.
    # If a is also large, a | x*c is harder.
    # So a set of large numbers is likely valid.
    # The seed starts with [129, 256].
    # Let's verify if [129, 256] is valid.
    # If it is valid, size is 128.
    # If not, we remove some.
    # Can we add ANY number from [2, 128]?
    # Let's try to add numbers to the seed's result.
    
    # Strategy:
    # 1. Start with [129, 256].
    # 2. Remove conflicting elements (as in seed) to get a valid base set S.
    # 3. Try to add elements from [2, 128] in a greedy manner, checking validity.
    # 4. Use local search (swap) to improve.
    
    S = list(range(129, 257))
    
    def is_valid_set(S):
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (b*c) % a == 0 or (a*c) % b == 0 or (a*b) % c == 0:
                        return False
        return True
        
    def get_conflict_pairs(S):
        # Returns list of (a,b) pairs that are part of at least one bad triple with some c in S.
        # Actually, for removal, we want to identify elements involved in bad triples.
        bad_elems = set()
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                for k in range(j+1, len(S)):
                    a, b, c = S[i], S[j], S[k]
                    if (b*c) % a == 0 or (a*c) % b == 0 or (a*b) % c == 0:
                        bad_elems.add(a)
                        bad_elems.add(b)
                        bad_elems.add(c)
        return bad_elems

    # Phase 1: Clean up [129, 256]
    # We need a fast validity check. O(N^3) is too slow for repeated checks if N is large.
    # But N starts at 128. 128^3 = 2M. Python loop is slow.
    # We need to optimize the check.
    
    def has_bad_triple(S):
        # Returns True if any bad triple exists.
        # Optimization: Precompute divisors?
        # a | b*c  <=>  a/gcd(a,b) | c.
        # For each pair (a,b), compute req = a // gcd(a,b).
        # Check if any c in S (c!=a, c!=b) is divisible by req.
        # We can store S in a set for O(1) lookup? No, we need to check divisibility.
        # We can store multiples of req in S?
        
        S_set = set(S)
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                # Check if a | b*c for some c
                g = math.gcd(a, b)
                req = a // g
                if req == 1:
                    # a | b. Since a!=b, this implies a is a proper divisor of b.
                    # Then a | b*c for ANY c.
                    # So if there is any c distinct from a,b, we have a bad triple.
                    if len(S) > 2:
                        return True
                    continue
                
                # Check if any c in S is divisible by req
                # c must be distinct from a, b.
                # Iterate through S? O(N). Total O(N^3).
                # Can we do better?
                # Precompute for each possible req (2..256) the list of multiples in S?
                # But S changes.
                
                # Let's stick to O(N^3) but with early exit.
                # 128^3 is 2M iterations. In Python, this might take 1-2 seconds.
                # If we do this 100 times, it's 100-200s. Too slow.
                
                # We need a faster check.
                # Idea: For each a, precompute the set of "forbidden" pairs (b,c)?
                # Or, for each a, precompute the set of b such that a/gcd(a,b) has a multiple in S.
                
                # Let's try a different approach:
                # For each c in S, which pairs (a,b) are bad with c?
                # a | b*c  <=>  a/gcd(a,b) | c.
                # Let d = a/gcd(a,b). Then d | c.
                # So for a fixed c, a and b are bad if there exists d | c such that d = a/gcd(a,b).
                # This is complex.
                
                # Alternative:
                # Just use the O(N^3) check but optimize the inner loop.
                # If we only do this a few times, it's fine.
                # But for greedy addition, we might do it 128 times.
                
                # Let's assume the seed's cleanup is fast enough if we just do it once.
                # The seed does it in a loop.
                # Let's replicate the seed's logic but faster.
        
        return False # Placeholder
        
    # Let's implement a faster "find_bad_triple" using bitsets or precomputation.
    # Precompute for each x in 2..256, the list of divisors.
    # divisors[x] = [d for d in range(1, x+1) if x % d == 0]
    
    # Actually, let's just use the seed's approach but with a tighter deadline.
    # The seed score is 1.0556. If seed returns 135, then 135/128 = 1.054.
    # If seed returns 136, 136/128 = 1.0625.
    # Let's find out what the seed returns.
    # I will run the seed logic mentally or just assume it's around 135.
    
    # New Plan:
    # 1. Start with S = [129..256].
    # 2. Use a fast conflict detector to remove elements until valid.
    # 3. Try to add elements from [2..128] sorted by "safety" (largest first).
    # 4. Local search: Try swapping an element in S with one outside to increase size? No, size is fixed if we swap.
    #    We want to MAXIMIZE size. So we only ADD.
    #    If we can't add, we are done.
    
    # Fast Conflict Detector:
    # For a set S, check if valid.
    # We can precompute for each a in S, the set of b in S such that a/gcd(a,b) has a multiple in S (excluding a,b).
    
    def check_validity(S):
        S_set = set(S)
        n = len(S)
        # For each a, b
        for i in range(n):
            a = S[i]
            for j in range(i+1, n):
                b = S[j]
                g = math.gcd(a, b)
                req_a = a // g
                req_b = b // g
                
                # Check if req_a has a multiple in S (distinct from a,b)
                # If req_a == 1, a|b. Bad if len(S)>2.
                if req_a == 1:
                    if n > 2:
                        return False
                    # If n==2, no c, so ok.
                    # But we need to check b|a*c too?
                    # If a|b, then b|a*c is b|b*k*c? No.
                    # If a|b, let b=ka.
                    # b | a*c  <=>  ka | ac  <=>  k | c.
                    # So if k | c for some c, it's bad.
                    
                # We need to check both a|bc and b|ac.
                # a|bc <=> req_a | c.
                # b|ac <=> req_b | c.
                
                # So we need to check if there exists c in S, c!=a, c!=b, such that
                # c % req_a == 0 OR c % req_b == 0.
                
                # Optimization:
                # Precompute for each possible req (2..256) the count of multiples in S?
                # But we need to exclude a and b.
                
                # Let's just iterate c for now, but break early.
                found = False
                for c in S:
                    if c == a or c == b:
                        continue
                    if c % req_a == 0 or c % req_b == 0:
                        found = True
                        break
                if found:
                    return False
        return True

    # This is O(N^3).
    # To speed up, we can use numpy? No, standard library.
    # We can precompute a "multiples_in_S" dictionary for each req?
    # But S changes.
    
    # Let's try to optimize the inner loop by sorting S and using binary search?
    # No, divisibility is not monotonic.
    
    # Let's just run it and hope it's fast enough for N~128.
    # 128^3 / 6 (triples) = 350k triples.
    # Each triple check is fast.
    # 350k checks * 100 iterations = 35M checks.
    # Python does ~10M simple ops/sec. So ~3-4 seconds per 100 iterations.
    # This is feasible within 80s.
    
    # Phase 1: Cleanup
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        # Find all bad elements
        bad_elems = set()
        S_set = set(S)
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i+1, n):
                b = S[j]
                g = math.gcd(a, b)
                req_a = a // g
                req_b = b // g
                
                # Check for c
                for k in range(j+1, n):
                    c = S[k]
                    if c % req_a == 0 or c % req_b == 0:
                        bad_elems.add(a)
                        bad_elems.add(b)
                        bad_elems.add(c)
                        
        if bad_elems:
            # Remove one bad element.
            # Which one? The one involved in the most bad triples?
            # For now, remove the first one found or the smallest.
            # Let's remove the one with highest "badness".
            # Count how many bad triples each element is in.
            bad_counts = {x: 0 for x in bad_elems}
            for i in range(n):
                a = S[i]
                for j in range(i+1, n):
                    b = S[j]
                    g = math.gcd(a, b)
                    req_a = a // g
                    req_b = b // g
                    for k in range(j+1, n):
                        c = S[k]
                        if c % req_a == 0 or c % req_b == 0:
                            bad_counts[a] += 1
                            bad_counts[b] += 1
                            bad_counts[c] += 1
                            
            # Remove max
            worst = max(bad_counts, key=bad_counts.get)
            S.remove(worst)
            changed = True
            
    # Phase 2: Greedy Addition
    # Try to add elements from 128 down to 2.
    # Larger elements are safer.
    for x in range(128, 1, -1):
        if time.time() > deadline - 5:
            break
        if x in S:
            continue
            
        # Check if S U {x} is valid
        # Only need to check triples involving x.
        valid = True
        S_set = set(S)
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                # Triple (x, a, b)
                # Check x | a*b
                if (a * b) % x == 0:
                    valid = False
                    break
                # Check a | x*b
                g_ax = math.gcd(a, x)
                req = a // g_ax
                if (b % req) == 0:
                    valid = False
                    break
                # Check b | x*a
                g_bx = math.gcd(b, x)
                req = b // g_bx
                if (a % req) == 0:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            S.append(x)
            
    return S
