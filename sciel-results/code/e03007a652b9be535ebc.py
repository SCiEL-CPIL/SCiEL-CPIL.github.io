import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute all triples
    all_triples = list(combinations(range(16), 3))
    
    best_size = 0
    best_set = set()
    
    # Randomized restarts
    # Each restart shuffles the order and greedily adds triples
    # We run until the deadline approaches
    while time.time() < deadline:
        random.shuffle(all_triples)
        current_set = set()
        
        # To check K(2,2,2) efficiently:
        # A K(2,2,2) is defined by 6 vertices {a, a', b, b', c, c'} 
        # such that all 8 triples {x,y,z} with x in {a,a'}, y in {b,b'}, z in {c,c'} are present.
        
        # Instead of checking all 8 for every new triple (which is expensive if done naively),
        # we can maintain a data structure or just check the condition directly.
        # Given the small size (560 triples), a direct check might be feasible if optimized,
        # but checking 8 specific triples for each candidate is O(1) set lookups.
        
        # For a new triple t = (u, v, w), it completes a K(2,2,2) if there exist
        # u', v', w' such that all 8 combinations of {u,u'}, {v,v'}, {w,w'} are in current_set ∪ {t}.
        # This is hard to check incrementally without precomputation.
        
        # Alternative: Since N=16 is small, we can just try to add and check.
        # How to check if adding t creates a K(2,2,2)?
        # The K(2,2,2) must include t.
        # Let t = {a, b, c}.
        # The other 5 vertices of the K(2,2,2) must be a', b', c', and the 6th vertex is determined?
        # No, K(2,2,2) has 6 vertices. t uses 3.
        # The K(2,2,2) is formed by pairs (A, A'), (B, B'), (C, C').
        # t picks one from each.
        # So if we add t={a,b,c}, we need to check if there exist a', b', c' 
        # such that all 8 triples from {a,a'}x{b,b'}x{c,c'} are in the graph.
        
        # This requires checking for every pair of vertices (a', b') if the other required triples exist.
        # Actually, for a fixed t={a,b,c}, the potential partners a', b', c' must be such that:
        # (a', b, c), (a, b', c), (a, b, c') are in the graph.
        # Then we check if (a', b', c), (a', b, c'), (a, b', c'), (a', b', c') are also in the graph.
        
        # This is still potentially expensive if we do it for every t.
        # However, 560 triples is small.
        # Let's implement a fast check.
        
        for t in all_triples:
            if time.time() > deadline:
                break
                
            if t in current_set:
                continue
                
            # Check if adding t creates a K(2,2,2)
            # t = (a, b, c)
            a, b, c = t
            
            # We need to find a', b', c' such that all 8 are present.
            # The 8 triples are:
            # (a,b,c) - t (being added)
            # (a',b,c), (a,b',c), (a,b,c')
            # (a',b',c), (a',b,c'), (a,b',c')
            # (a',b',c')
            
            # Candidate a' must form (a', b, c) with b,c.
            # Candidate b' must form (a, b', c) with a,c.
            # Candidate c' must form (a, b, c') with a,b.
            
            # Let's iterate over possible a', b', c'?
            # That's O(N^3) per triple, which is 16^3 * 560 ~ 2 million operations.
            # 2M operations per restart is very fast.
            
            conflict = False
            for a_prime in range(16):
                if a_prime == a: continue
                if (a_prime, b, c) not in current_set: continue
                
                for b_prime in range(16):
                    if b_prime == b or b_prime == a_prime: continue
                    if (a, b_prime, c) not in current_set: continue
                    # Also need (a_prime, b_prime, c) ? No, that's part of the 8.
                    # Let's be systematic.
                    # The set of 8 triples is:
                    # T = { (x, y, z) | x in {a, a'}, y in {b, b'}, z in {c, c'} }
                    # We know t=(a,b,c) is being added.
                    # We need the other 7 to be in current_set.
                    
                    # So for fixed a', b', we need to find c' such that:
                    # (a', b', c) in current_set
                    # (a, b', c) in current_set (checked above)
                    # (a', b, c) in current_set (checked above)
                    # (a', b, c') in current_set
                    # (a, b', c') in current_set
                    # (a, b, c') in current_set
                    # (a', b', c') in current_set
                    
                    # Let's refine the loops.
                    # For a given t=(a,b,c):
                    # 1. Find all a' such that (a', b, c) in current_set.
                    # 2. Find all b' such that (a, b', c) in current_set.
                    # 3. Find all c' such that (a, b, c') in current_set.
                    
                    # Then for each (a', b', c') in Cartesian product of these lists:
                    # Check if the remaining 4 triples are in current_set:
                    # (a', b', c), (a', b, c'), (a, b', c'), (a', b', c')
                    
                    # This is much faster.
                    
                    pass # Placeholder for logic below
            
            # Implementation of the fast check
            a_primes = []
            b_primes = []
            c_primes = []
            
            for x in range(16):
                if x == a: continue
                if (x, b, c) in current_set:
                    a_primes.append(x)
            
            for y in range(16):
                if y == b: continue
                if (a, y, c) in current_set:
                    b_primes.append(y)
            
            for z in range(16):
                if z == c: continue
                if (a, b, z) in current_set:
                    c_primes.append(z)
            
            for ap in a_primes:
                for bp in b_primes:
                    if ap == bp: continue
                    for cp in c_primes:
                        if cp == ap or cp == bp: continue
                        
                        # Check the other 4 triples
                        if (ap, bp, c) in current_set and \
                           (ap, b, cp) in current_set and \
                           (a, bp, cp) in current_set and \
                           (ap, bp, cp) in current_set:
                            conflict = True
                            break
                    if conflict: break
                if conflict: break
            
            if not conflict:
                current_set.add(t)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
        # If we found a very good solution, we might stop early?
        # The maximum is unknown, but C(16,3)=560.
        # If we get close to 560, we can stop.
        if best_size > 500:
            break

    return [list(t) for t in best_set]
