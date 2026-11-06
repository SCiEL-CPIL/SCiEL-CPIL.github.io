import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, but we want to return well before 60s if possible,
    # or at least before 90s. Let's aim for 80s limit to be safe against overhead.
    deadline = start_time + 80.0
    
    def is_valid_addition(S_set, S_list, x):
        """Check if x can be added to S (which is a list, S_set is set of S)"""
        # 1. x must not divide sum of any 3 distinct others
        # S_list size is k. We need to check sums of 3 from S_list.
        # But we can use precomputed sums? 
        # For a single addition check, we can just iterate.
        # k is at most 48. C(48,3) = 17296.
        # If we do this for 192 candidates, that's 3.3M checks. Fast enough.
        
        for a, b, c in combinations(S_list, 3):
            if (a + b + c) % x == 0:
                return False
        
        # 2. For each y in S, y must not divide x + u + v for any u,v in S\{y}
        # y divides x + u + v  <=>  x + u + v == 0 mod y
        # u + v == -x mod y
        # We can check this by iterating u, v in S\{y}.
        # C(47, 2) = 1081.
        # 48 * 1081 = 51k checks.
        
        for y in S_list:
            target = (-x) % y
            # Check if any pair u, v in S_list \ {y} sums to target mod y
            # Optimization: Precompute pair sums mod y? 
            # Since this is called repeatedly, maybe not worth precomputing for every y every time.
            # Just iterate.
            found = False
            others = [z for z in S_list if z != y]
            for i in range(len(others)):
                for j in range(i+1, len(others)):
                    if (others[i] + others[j]) % y == target:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
        
        return True

    def fast_greedy(order):
        S = []
        S_set = set()
        
        # We will maintain K (sums of 3) and R (pair sums mod a) for speed?
        # The original fast_greedy did this. It was faster for the *building* phase.
        # Let's stick to the original fast_greedy logic but ensure it's not too slow.
        # The original logic was:
        # Check 1: x divides any sum in K?
        # Check 2: For each a in S, does x + u + v == 0 mod a? (using R[a])
        
        # Let's re-implement fast_greedy carefully.
        
        K = set() # Sums of 3 distinct elements in S
        R = {}    # R[a] = set of (u+v) % a for pairs u,v in S\{a}
        
        for x in order:
            if x in S_set:
                continue
            
            # Check 1: x does not divide any sum in K
            valid = True
            for s in K:
                if s % x == 0:
                    valid = False
                    break
            
            # Check 2: For each a in S, a does not divide x + u + v
            if valid:
                for a in S:
                    target = (-x) % a
                    if target in R[a]:
                        valid = False
                        break
            
            if valid:
                old_S = S[:]
                S.append(x)
                S_set.add(x)
                
                n_old = len(old_S)
                
                # Update K: add sums x + u + v for u, v in old_S, u != v
                for i in range(n_old):
                    for j in range(i + 1, n_old):
                        K.add(x + old_S[i] + old_S[j])
                
                # Update R for existing a
                for a in old_S:
                    for u in old_S:
                        if u != a:
                            R[a].add((x + u) % a)
                
                # Initialize R[x]
                R[x] = set()
                for i in range(n_old):
                    for j in range(i + 1, n_old):
                        R[x].add((old_S[i] + old_S[j]) % x)
                        
            if time.time() > deadline - 5:
                break
                
        return S

    def maximalize(S, deadline):
        """Try to add any element from 1..192 not in S to S if valid."""
        S_set = set(S)
        S_list = S[:]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Sort candidates? Maybe larger numbers are harder to add? 
            # Or smaller? 
            # Let's just iterate 1..192.
            for x in range(1, 193):
                if x in S_set:
                    continue
                if is_valid_addition(S_set, S_list, x):
                    S_list.append(x)
                    S_set.add(x)
                    changed = True
                    # If we added one, restart loop? Or continue?
                    # If we continue, we might add multiple.
                    # But adding one changes validity for others.
                    # So we should restart or just keep going?
                    # If we keep going, we check against the *updated* S_list.
                    # This is correct.
                    # However, if we add x, we should break the inner loop and restart the while loop?
                    # No, just updating S_list and S_set and continuing the for loop is fine, 
                    # but the condition `is_valid_addition` uses the current S_list.
                    # So it checks against the set including the newly added x.
                    # This is correct.
                    
            # If we go through all 1..192 and add nothing, changed is False.
            
        return S_list

    best_set = []
    best_size = 0
    
    random.seed(42)
    
    # Generate orders
    orders = []
    orders.append(list(range(192, 0, -1))) # Descending
    orders.append(list(range(1, 193)))     # Ascending
    
    # Random orders
    for _ in range(20):
        perm = list(range(1, 193))
        random.shuffle(perm)
        orders.append(perm)
        
    # Biased orders (prefer large numbers, as they are less likely to divide sums)
    for _ in range(20):
        weights = [i**2 for i in range(1, 193)]
        perm = random.choices(range(1, 193), weights=weights, k=192)
        seen = set()
        unique_perm = []
        for x in perm:
            if x not in seen:
                seen.add(x)
                unique_perm.append(x)
        orders.append(unique_perm)

    # Run greedy on each order
    for i, order in enumerate(orders):
        if time.time() > deadline - 10:
            break
            
        S = fast_greedy(order)
        
        # Maximalize the result
        if time.time() < deadline - 2:
            S = maximalize(S, deadline - 2)
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we found 48, we can stop early? 
        # The max possible is 48.
        if best_size == 48:
            break

    return best_set
