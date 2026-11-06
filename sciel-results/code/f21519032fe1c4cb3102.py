"""Optimized program for 'No element dividing a sum of three others' task.
Strategy: Greedy with O(N^2) incremental updates for conflict checks.
Maintains sums of triples and pair residues modulo each element.
"""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    def try_add(S, sums_set, pair_res, y):
        """
        Check if y can be added to S.
        Returns True if valid, False otherwise.
        """
        # Check 1: y does not divide any sum of 3 distinct elements in S
        # Sum of 3 distinct positive ints is > 0.
        # Max sum is 192+191+190 = 573.
        # So we check multiples of y up to 573.
        for k in range(1, 574 // y + 1):
            if k * y in sums_set:
                return False
        
        # Check 2: For each a in S, a does not divide sum of any 2 distinct elements in S\{a} plus y.
        # i.e., (b + c + y) % a != 0 for all distinct b, c in S\{a}.
        # Equivalent to: (-y % a) not in pair_res[a]
        for a in S:
            target = (-y) % a
            if target in pair_res[a]:
                return False
        
        return True

    def update_structures(S, sums_set, pair_res, y):
        """
        Add y to S and update sums_set and pair_res.
        Assumes y is valid to add.
        """
        old_S = S[:] # List of current elements
        S.append(y)
        
        # Update sums_set: Add y+b+c for all pairs b,c in old_S
        n_old = len(old_S)
        for i in range(n_old):
            b = old_S[i]
            for j in range(i+1, n_old):
                c = old_S[j]
                sums_set.add(y + b + c)
        
        # Update pair_res for existing a in old_S
        # New pairs for a are (y, b) where b in old_S and b != a
        for a in old_S:
            for b in old_S:
                if b == a:
                    continue
                r = (y + b) % a
                pair_res[a].add(r)
        
        # Initialize pair_res for y
        # Pairs are (b, c) in old_S, b != c
        res_y = set()
        for i in range(n_old):
            b = old_S[i]
            for j in range(i+1, n_old):
                c = old_S[j]
                res_y.add((b + c) % y)
        pair_res[y] = res_y

    def remove_element(S, sums_set, pair_res, y):
        """
        Remove y from S and update structures.
        This is complex to do incrementally. 
        For local search, it might be faster to rebuild from scratch if N is small, 
        or implement a reverse update.
        Given N <= 48, rebuilding is O(N^2) which is ~2300 ops.
        So we can just rebuild for swaps.
        """
        pass

    def build_structures(S):
        """
        Build sums_set and pair_res from scratch for a given set S.
        """
        sums_set = set()
        pair_res = {}
        n = len(S)
        
        # Sums of triples
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    sums_set.add(S[i] + S[j] + S[k])
                    
        # Pair residues
        for a in S:
            res_a = set()
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    res_a.add((b + c) % a)
            pair_res[a] = res_a
            
        return sums_set, pair_res

    best = []
    best_len = 0
    
    # Greedy Phase
    # Try multiple orderings
    orderings = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
    ]
    
    random.seed(42)
    for _ in range(20):
        order = list(range(1, 193))
        random.shuffle(order)
        orderings.append(order)
        
    for order in orderings:
        if time.time() > deadline - 10:
            break
            
        S = []
        sums_set = set()
        pair_res = {}
        
        for y in order:
            if len(S) >= 48:
                break
            if try_add(S, sums_set, pair_res, y):
                update_structures(S, sums_set, pair_res, y)
                
        if len(S) > best_len:
            best = S[:]
            best_len = len(S)
            if best_len >= 48:
                break

    # Local Search Phase
    # Use 1-out-1-in and 2-out-2-in swaps
    # Since rebuild is O(N^2) ~ 2300, we can afford many swaps.
    
    current = best[:]
    current_len = best_len
    
    if current_len < 48 and time.time() < deadline - 5:
        improved = True
        while improved and current_len < 48 and time.time() < deadline - 5:
            improved = False
            sums_set, pair_res = build_structures(current)
            
            # Try 1-out-1-in
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                # Remove a
                temp_S = [v for j, v in enumerate(current) if j != i]
                # Rebuild for temp_S? Or just check candidates against temp_S?
                # Checking against temp_S requires structures for temp_S.
                # Rebuilding is O(N^2). 
                # Let's just rebuild for each removal candidate if we do 1-out-1-in.
                # This is O(N) rebuilds * O(N^2) rebuild cost = O(N^3) per pass.
                # N=48, N^3 = 110k. Acceptable.
                
                temp_sums, temp_pair_res = build_structures(temp_S)
                
                # Try adding b
                for b in range(192, 0, -1):
                    if b in temp_S:
                        continue
                    if try_add(temp_S, temp_sums, temp_pair_res, b):
                        new_S = temp_S + [b]
                        # Since we swapped 1 for 1, size is same.
                        # We want to increase size, so 1-out-1-in doesn't help unless we can add more later.
                        # Actually, we want to find a set of size N+1.
                        # So we should try 1-out-2-in or just try to add to current.
                        # Let's stick to 1-out-2-in for size increase.
                        pass
            
            # Try 1-out-2-in
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                a = current[i]
                temp_S = [v for j, v in enumerate(current) if j != i]
                temp_sums, temp_pair_res = build_structures(temp_S)
                
                # Try adding b
                for b in range(192, 0, -1):
                    if b in temp_S:
                        continue
                    if not try_add(temp_S, temp_sums, temp_pair_res, b):
                        continue
                    
                    # If b added, try adding c to temp_S + [b]
                    temp_S2 = temp_S + [b]
                    # Update structures for temp_S2?
                    # We can just rebuild or update.
                    # Rebuild is safer.
                    temp_sums2, temp_pair_res2 = build_structures(temp_S2)
                    
                    for c in range(192, 0, -1):
                        if c in temp_S2:
                            continue
                        if try_add(temp_S2, temp_sums2, temp_pair_res2, c):
                            new_S = temp_S2 + [c]
                            if len(new_S) > current_len:
                                current = new_S
                                current_len = len(current)
                                best = current[:]
                                best_len = current_len
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    return best[:48] if len(best) >= 48 else best
