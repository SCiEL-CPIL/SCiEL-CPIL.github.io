import time
import random
from math import gcd

def construct():
    deadline = time.time() + 55.0  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute multiples for fast checking
    multiples_of = [[] for _ in range(N + 1)]
    for k in range(1, N + 1):
        for m in range(k, N + 1, k):
            if m >= 2:
                multiples_of[k].append(m)
    
    def can_add(a, in_set):
        """
        Check if a can be added to the set represented by in_set (boolean list).
        in_set[i] is True if i is in the set.
        For every distinct b, c in S: a does not divide b*c.
        This is equivalent to: for every b in S, (a/gcd(a,b)) does not divide any c in S where c != b.
        """
        for b in range(2, N + 1):
            if not in_set[b]:
                continue
            
            g = gcd(a, b)
            
            # Check: a | b*c => (a/g) | c for some c in S, c != b
            k1 = a // g
            if k1 == 1:
                # a divides b, so a | b*c for any c. But we need c != b.
                # If there's any other element in S, we have a problem.
                # Actually if a|b, then for any c, a|bc. So we need no other c in S.
                # But we're checking if we can ADD a, so S already has elements.
                # If a|b and there exists c in S with c != b, then a|bc.
                # So if len(S) >= 2 and a|b, we can't add a.
                # More precisely: if a|b, then for ANY c, a|bc.
                # So we need that there is NO c in S with c != b.
                # i.e., S must contain only b (and we're adding a).
                count = sum(in_set[2:])
                if count > 1:
                    return False
                # If count == 1, S = {b}, and a|b, then a|b*b? No, a|b means b=ka, so b*b = k*a*b, so a|b*b.
                # Wait, the condition is: for three DISTINCT members a,b,c, a does not divide b*c.
                # If S = {b} and we add a, then S' = {a, b}. We need to check triples from S'.
                # But S' only has 2 elements, so no triple of distinct members. So it's valid.
                # Actually the constraint is on the final set. If final set has < 3 elements, it's valid.
                # So if after adding a, |S'| < 3, it's automatically valid.
                # But we're checking can_add which assumes S is current and we're adding a.
                # The check should be: does adding a create any violation?
                # A violation is a triple (x,y,z) distinct in S'={S}∪{a} such that x|y*z.
                # Cases:
                # 1. x=a: need a | y*z for distinct y,z in S. This is what we're checking.
                # 2. x in S: need x | a*z or x | y*a for distinct y,z in S∪{a}.
                #    If x|a*z, that's (x, a, z) with x,z in S, a new. Or (x, z, a).
                #    If x|y*a, that's (x, y, a) with x,y in S, a new.
                # So we also need to check: for x in S, does x | a*c for some c in S, c != a?
                # But c must be distinct from x and a. Since a is new, c in S, c != x.
                # So: for x in S, for c in S, c != x: does x | a*c?
                # This is equivalent to: (x/gcd(x,a)) | c.
                
                # Let me redo the logic more carefully.
                pass
            
            for c in multiples_of[k1]:
                if c == b:
                    continue
                if in_set[c]:
                    return False
            
            # Check: b | a*c => (b/g) | c for some c in S, c != a (but a not in S yet)
            # Actually: for distinct x,y,z in S'={S}∪{a}, we need x ∤ y*z.
            # If x=b (in S), y=a (new), z=c (in S): need b ∤ a*c.
            # b | a*c iff (b/gcd(b,a)) | c.
            k2 = b // g
            for c in multiples_of[k2]:
                if c == b:  # c must be distinct from b, and c is in S
                    continue
                # c must also be distinct from a, but a is not in S yet, so c in S is fine
                # But we need c != a. Since a is being added and not in S, c in S is automatically != a.
                if in_set[c]:
                    return False
                    
        return True
    
    def greedy_construction(order):
        in_set = [False] * (N + 1)
        current_set = []
        for a in order:
            if can_add(a, in_set):
                in_set[a] = True
                current_set.append(a)
        return current_set, in_set
    
    def local_search(S, in_set, max_iterations=20000):
        """
        Local search to improve set S.
        Tries to add elements and perform swaps to increase size.
        """
        current_set = S[:]
        current_in_set = in_set[:]
        current_size = len(current_set)
        best_set = current_set[:]
        best_in_set = current_in_set[:]
        best_size = current_size
        
        non_members = [x for x in candidates if not current_in_set[x]]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
            
            if not non_members:
                break
            
            # Try to add a random non-member
            if random.random() < 0.5:
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                if can_add(x, current_in_set):
                    current_in_set[x] = True
                    current_set.append(x)
                    current_size += 1
                    non_members.pop(idx)
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                        best_in_set = current_in_set[:]
            else:
                # Try a swap: remove y from current_set, add x from non_members
                if len(current_set) > 0:
                    y_idx = random.randint(0, len(current_set) - 1)
                    y = current_set[y_idx]
                    idx = random.randint(0, len(non_members) - 1)
                    x = non_members[idx]
                    
                    # Temporarily remove y
                    current_in_set[y] = False
                    temp_set = current_set[:]
                    temp_set.pop(y_idx)
                    
                    if can_add(x, current_in_set):
                        # Swap accepted
                        current_in_set[x] = True
                        current_set = temp_set + [x]
                        non_members[idx] = y  # Replace x with y in non_members
                        
                        # After swap, try to add more elements
                        for _ in range(3):  # Try a few additions after swap
                            if not non_members:
                                break
                            add_idx = random.randint(0, len(non_members) - 1)
                            z = non_members[add_idx]
                            if can_add(z, current_in_set):
                                current_in_set[z] = True
                                current_set.append(z)
                                current_size += 1
                                non_members.pop(add_idx)
                                if current_size > best_size:
                                    best_size = current_size
                                    best_set = current_set[:]
                                    best_in_set = current_in_set[:]
                            else:
                                break
                        
                        # If no addition, size is same as before swap
                        if current_size < best_size:
                            # This shouldn't happen, but restore
                            pass
    
        return best_set, best_in_set
    
    best_set = []
    best_size = 0
    best_in_set = [False] * (N + 1)
    
    # Phase 1: Upper-half antichain strategy
    # Start with numbers in (128, 256] - these form an antichain
    upper_half = list(range(129, 257))
    lower_half = list(range(2, 129))
    
    # Order: upper half first, then lower half
    order1 = upper_half + lower_half
    S1, in_set1 = greedy_construction(order1)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1
        best_in_set = in_set1
    
    # Try reverse: lower half first
    order2 = lower_half + upper_half
    S2, in_set2 = greedy_construction(order2)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2
        best_in_set = in_set2
    
    # Try primes first
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    order3 = primes + composites
    S3, in_set3 = greedy_construction(order3)
    if len(S3) > best_size:
        best_size = len(S3)
        best_set = S3
        best_in_set = in_set3
    
    # Random orders
    rng = random.Random(42)
    for _ in range(20):
        if time.time() > deadline - 10.0:
            break
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        S_r, in_set_r = greedy_construction(shuffled)
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_set = S_r
            best_in_set = in_set_r
    
    # Phase 2: Local search on best set
    if time.time() < deadline - 3.0:
        improved_set, improved_in_set = local_search(best_set, best_in_set, max_iterations=30000)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            best_in_set = improved_in_set
    
    # Phase 3: More restarts with local search
    for _ in range(10):
        if time.time() > deadline - 2.0:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        S_r, in_set_r = greedy_construction(shuffled)
        if len(S_r) > best_size:
            best_size = len(S_r)
            best_set = S_r
            best_in_set = in_set_r
        elif len(S_r) >= best_size - 2 and time.time() < deadline - 1.0:
            # Close to best, try local search
            improved_set, improved_in_set = local_search(S_r, in_set_r, max_iterations=5000)
            if len(improved_set) > best_size:
                best_size = len(improved_set)
                best_set = improved_set
                best_in_set = improved_in_set
    
    # Final validation
    def is_valid(S):
        s_list = list(S)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = s_list[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    if not is_valid(best_set):
        # Fallback: return primes
        best_set = primes
    
    return list(best_set)
