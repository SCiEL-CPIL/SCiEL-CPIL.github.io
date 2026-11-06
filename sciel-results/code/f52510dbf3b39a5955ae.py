import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 90s limit (or 60s eval limit, 55 is safe)
    
    n = 256
    best_A = []
    best_B = []
    best_size = 0
    
    def recompute(S):
        sums = set()
        diffs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
            for j in range(len(S_list)):
                if i != j:
                    diffs.add(S_list[i] - S_list[j])
        return sums, diffs

    def is_valid_state(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        # Check Sidon
        # Number of pairs with replacement is k*(k+1)/2
        if len(sums_A) != len(A) * (len(A) + 1) // 2:
            return False
        if len(sums_B) != len(B) * (len(B) + 1) // 2:
            return False
        # Check Disjoint Diffs
        if diffs_A & diffs_B:
            return False
        return True

    def local_search(A, B):
        A = set(A)
        B = set(B)
        
        # Initial recompute
        sums_A, diffs_A = recompute(A)
        sums_B, diffs_B = recompute(B)
        
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > time_limit:
                break
                
            # 1. Try Adds
            # Try all candidates not in A or B
            candidates = [x for x in range(1, n + 1) if x not in A and x not in B]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                # Try Add to A
                if len(A) < 64:
                    # Check Sidon for A
                    ok_sidon = True
                    if 2 * x in sums_A:
                        ok_sidon = False
                    else:
                        for s in A:
                            if x + s in sums_A:
                                ok_sidon = False
                                break
                    
                    if ok_sidon:
                        # Check Diff Disjointness: new diffs vs diffs_B
                        ok_diff = True
                        for s in A:
                            if (x - s) in diffs_B or (s - x) in diffs_B:
                                ok_diff = False
                                break
                        
                        if ok_diff:
                            # Execute Add to A
                            A.add(x)
                            for s in A:
                                if s != x:
                                    sums_A.add(x + s)
                                    diffs_A.add(x - s)
                                    diffs_A.add(s - x)
                            sums_A.add(2 * x)
                            improved = True
                            added = True
                            break
                
                # Try Add to B
                if not added and len(B) < 64:
                    # Check Sidon for B
                    ok_sidon = True
                    if 2 * x in sums_B:
                        ok_sidon = False
                    else:
                        for s in B:
                            if x + s in sums_B:
                                ok_sidon = False
                                break
                    
                    if ok_sidon:
                        # Check Diff Disjointness: new diffs vs diffs_A
                        ok_diff = True
                        for s in B:
                            if (x - s) in diffs_A or (s - x) in diffs_A:
                                ok_diff = False
                                break
                        
                        if ok_diff:
                            # Execute Add to B
                            B.add(x)
                            for s in B:
                                if s != x:
                                    sums_B.add(x + s)
                                    diffs_B.add(x - s)
                                    diffs_B.add(s - x)
                            sums_B.add(2 * x)
                            improved = True
                            added = True
                            break
            
            if improved:
                continue

            # 2. Try Moves
            # Move x from A to B
            moved = False
            for x in list(A):
                if len(B) >= 64:
                    continue
                
                # Check if x can be added to B
                # Sidon check for B
                ok_sidon = True
                if 2 * x in sums_B:
                    ok_sidon = False
                else:
                    for s in B:
                        if x + s in sums_B:
                            ok_sidon = False
                            break
                
                if not ok_sidon:
                    continue
                
                # Diff check: new diffs (x vs B) must not be in diffs_A'
                # Need diffs_A'
                A_temp = A - {x}
                _, diffs_A_temp = recompute(A_temp)
                
                ok_diff = True
                for s in B:
                    if (x - s) in diffs_A_temp or (s - x) in diffs_A_temp:
                        ok_diff = False
                        break
                
                if ok_diff:
                    # Execute Move
                    # Remove x from A
                    # Update sums_A: remove sums involving x
                    for s in A:
                        if s != x:
                            sums_A.remove(x + s)
                    sums_A.remove(2 * x)
                    # Update diffs_A: recompute is safer/faster for small sets?
                    # Or remove diffs involving x.
                    # Diffs are not unique, so we can't just remove.
                    # Recompute A after removal.
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    
                    # Add x to B
                    B.add(x)
                    for s in B:
                        if s != x:
                            sums_B.add(x + s)
                            diffs_B.add(x - s)
                            diffs_B.add(s - x)
                    sums_B.add(2 * x)
                    
                    improved = True
                    moved = True
                    break
            
            if improved:
                continue

            # Move x from B to A
            for x in list(B):
                if len(A) >= 64:
                    continue
                
                # Check if x can be added to A
                ok_sidon = True
                if 2 * x in sums_A:
                    ok_sidon = False
                else:
                    for s in A:
                        if x + s in sums_A:
                            ok_sidon = False
                            break
                
                if not ok_sidon:
                    continue
                
                B_temp = B - {x}
                _, diffs_B_temp = recompute(B_temp)
                
                ok_diff = True
                for s in A:
                    if (x - s) in diffs_B_temp or (s - x) in diffs_B_temp:
                        ok_diff = False
                        break
                
                if ok_diff:
                    # Execute Move
                    for s in B:
                        if s != x:
                            sums_B.remove(x + s)
                    sums_B.remove(2 * x)
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    
                    A.add(x)
                    for s in A:
                        if s != x:
                            sums_A.add(x + s)
                            diffs_A.add(x - s)
                            diffs_A.add(s - x)
                    sums_A.add(2 * x)
                    
                    improved = True
                    moved = True
                    break
            
            if improved:
                continue

            # 3. Try Swaps
            # Swap x in A with y not in A/B
            # This allows us to replace a "bad" element with a "better" one
            swapped = False
            # To avoid O(N^2) full sweep which might be slow, we do a limited random search
            # But we can make it smarter.
            # Let's try swapping each x in A with a few random y's
            for x in list(A):
                if swapped: break
                # Generate candidates y
                # Exclude A and B
                # We can just pick random y from 1..N
                for _ in range(10):
                    y = random.randint(1, n)
                    if y in A or y in B:
                        continue
                    
                    # Check if y can replace x in A
                    # A' = A \ {x}
                    # Check Sidon for A' + y
                    # We need sums_A_prime.
                    # sums_A_prime = sums_A - {x+s for s in A} - {2x}
                    # Since sums are unique, this is valid.
                    # But we don't want to modify sums_A yet.
                    # Let's simulate.
                    
                    # Sidon Check
                    ok_sidon = True
                    # 2y must not be in sums_A_prime
                    # sums_A_prime contains sums of pairs in A\{x}.
                    # 2y is a new sum.
                    # It must not be in sums_A_prime.
                    # Note: 2y might be in sums_A if y==x? No y!=x.
                    # If 2y is in sums_A, it must be from a pair in A\{x} or involving x.
                    # If involving x, it's removed in prime.
                    # So we need to check if 2y is in sums_A_prime.
                    
                    # Efficient way:
                    # Check if 2y is in sums_A.
                    # If yes, check if it was formed by x.
                    # If formed by x, it's not in sums_A_prime.
                    # If formed by others, it is in sums_A_prime -> Conflict.
                    
                    # This is getting complex.
                    # Given k<=64, let's just recompute sums_A_prime and diffs_A_prime for the check?
                    # No, that's O(k^2).
                    # Let's stick to the previous logic:
                    # We can compute sums_A_prime by removing x's contributions.
                    
                    # Actually, let's just recompute A_temp and check.
                    # It's O(k^2). If we do this for 10 y's per x, and 64 x's...
                    # 64 * 10 * 4000 = 2.5M ops. This is acceptable.
                    
                    A_temp = A - {x}
                    sums_A_temp, diffs_A_temp = recompute(A_temp)
                    
                    # Check Sidon for adding y to A_temp
                    if 2 * y in sums_A_temp:
                        continue
                    sidon_ok = True
                    for s in A_temp:
                        if y + s in sums_A_temp:
                            sidon_ok = False
                            break
                    if not sidon_ok:
                        continue
                        
                    # Check Diff Disjointness
                    # New diffs from y with A_temp must not be in diffs_B
                    diff_ok = True
                    for s in A_temp:
                        if (y - s) in diffs_B or (s - y) in diffs_B:
                            diff_ok = False
                            break
                    if not diff_ok:
                        continue
                        
                    # Execute Swap
                    # Remove x from A
                    A.remove(x)
                    sums_A, diffs_A = recompute(A)
                    # Add y to A
                    A.add(y)
                    for s in A:
                        if s != y:
                            sums_A.add(y + s)
                            diffs_A.add(y - s)
                            diffs_A.add(s - y)
                    sums_A.add(2 * y)
                    
                    improved = True
                    swapped = True
                    break
            
            if improved:
                continue

            # Swap x in B with y not in A/B
            for x in list(B):
                if swapped: break
                for _ in range(10):
                    y = random.randint(1, n)
                    if y in A or y in B:
                        continue
                    
                    B_temp = B - {x}
                    sums_B_temp, diffs_B_temp = recompute(B_temp)
                    
                    if 2 * y in sums_B_temp:
                        continue
                    sidon_ok = True
                    for s in B_temp:
                        if y + s in sums_B_temp:
                            sidon_ok = False
                            break
                    if not sidon_ok:
                        continue
                        
                    diff_ok = True
                    for s in B_temp:
                        if (y - s) in diffs_A or (s - y) in diffs_A:
                            diff_ok = False
                            break
                    if not diff_ok:
                        continue
                        
                    # Execute Swap
                    B.remove(x)
                    sums_B, diffs_B = recompute(B)
                    B.add(y)
                    for s in B:
                        if s != y:
                            sums_B.add(y + s)
                            diffs_B.add(y - s)
                            diffs_B.add(s - y)
                    sums_B.add(2 * y)
                    
                    improved = True
                    swapped = True
                    break
            
            if improved:
                continue

        return list(A), list(B)

    # Greedy Randomized Adaptive Search Procedure (GRASP)
    num_restarts = 50
    
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        A = set()
        B = set()
        sums_A = set()
        diffs_A = set()
        sums_B = set()
        diffs_B = set()
        
        # Randomized Greedy Construction
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
                
            # Try to add to A
            if len(A) < 64:
                # Quick check
                ok = True
                if 2*x in sums_A: ok=False
                else:
                    for s in A:
                        if x+s in sums_A: ok=False; break
                if ok:
                    okd = True
                    for s in A:
                        if (x-s) in diffs_B or (s-x) in diffs_B: okd=False; break
                    if okd:
                        A.add(x)
                        for s in A:
                            if s!=x:
                                sums_A.add(x+s)
                                diffs_A.add(x-s)
                                diffs_A.add(s-x)
                        sums_A.add(2*x)
                        continue
            
            # Try to add to B
            if len(B) < 64:
                ok = True
                if 2*x in sums_B: ok=False
                else:
                    for s in B:
                        if x+s in sums_B: ok=False; break
                if ok:
                    okd = True
                    for s in B:
                        if (x-s) in diffs_A or (s-x) in diffs_A: okd=False; break
                    if okd:
                        B.add(x)
                        for s in B:
                            if s!=x:
                                sums_B.add(x+s)
                                diffs_B.add(x-s)
                                diffs_B.add(s-x)
                        sums_B.add(2*x)
                        continue
            
        # Local Search
        A, B = local_search(A, B)
        
        # Validate
        if is_valid_state(A, B):
            size = len(A) + len(B)
            if size > best_size:
                best_size = size
                best_A = list(A)
                best_B = list(B)

    # Final check
    if not best_A or not best_B:
        return [[1], [2]]
        
    return [best_A, best_B]
