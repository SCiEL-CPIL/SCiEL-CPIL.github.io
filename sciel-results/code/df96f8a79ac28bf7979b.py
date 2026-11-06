import time
import random
from itertools import combinations

def construct():
        start_time = time.time()
        deadline = start_time + 80.0
        
        def fast_greedy(order):
            S = []
            K = set() # Sums of 3 distinct elements in S
            R = {}    # Residues of pairs in S\{a} mod a
            
            for x in order:
                # Check 1: x does not divide any sum in K
                valid = True
                for s in K:
                    if s % x == 0:
                        valid = False
                        break
                
                # Check 2: For each a in S, a does not divide x + u + v
                if valid:
                    for a in S:
                        # We need u, v in S \ {a} such that (x + u + v) % a == 0
                        # i.e., (u + v) % a == (-x) % a
                        target = (-x) % a
                        if target in R[a]:
                            valid = False
                            break
                
                if valid:
                    # Add x to S
                    old_S = S[:]
                    S.append(x)
                    
                    # Update K: add sums x + u + v for u, v in old_S, u != v
                    n_old = len(old_S)
                    for i in range(n_old):
                        for j in range(i + 1, n_old):
                            K.add(x + old_S[i] + old_S[j])
                    
                    # Update R for existing a
                    for a in old_S:
                        # Add residues (x + u) % a for u in old_S \ {a}
                        # Note: u must be distinct from a.
                        # Also u is from old_S.
                        # The pair is (x, u).
                        # Are x and u distinct? Yes, x is new.
                        # Is u distinct from a? We iterate u in old_S.
                        # If u == a, we skip?
                        # The condition for R[a] is pairs from S_new \ {a}.
                        # S_new = old_S U {x}.
                        # S_new \ {a} = (old_S \ {a}) U {x}.
                        # Pairs in S_new \ {a} involving x are (x, u) where u in old_S \ {a}.
                        # So we iterate u in old_S. If u == a, skip.
                        for u in old_S:
                            if u != a:
                                R[a].add((x + u) % a)
                    
                    # Initialize R[x]
                    # R[x] needs pairs from S_new \ {x} = old_S.
                    # So all pairs (u, v) in old_S, u != v.
                    R[x] = set()
                    for i in range(n_old):
                        for j in range(i + 1, n_old):
                            R[x].add((old_S[i] + old_S[j]) % x)
                
                if time.time() > deadline:
                    break
                    
            return S

        best_set = []
        best_size = 0
        
        # Orders
        orders = []
        orders.append(list(range(192, 0, -1)))
        orders.append(list(range(1, 193)))
        
        random.seed(42)
        # Run many random restarts
        # Since fast_greedy is fast, we can run many.
        for _ in range(200):
            if time.time() > deadline:
                break
            perm = list(range(1, 193))
            random.shuffle(perm)
            orders.append(perm)
            
        for order in orders:
            if time.time() > deadline:
                break
            S = fast_greedy(order)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
                
        return best_set
