import time
import random

def construct():
    n = 256
    deadline = time.time() + 55.0  # Safety margin
    
    best_A = []
    best_B = []
    best_sum = 0
    
    def check_add(A, B, x, target):
        """
        Check if x can be added to target (A or B).
        Returns True if valid, and updates sets if valid? 
        Better: Return a tuple (valid, new_sums, new_diffs) or just check.
        For speed, we'll do a dry run check.
        """
        if target == 'A':
            sums = A_sums
            diffs_own = A_diffs
            diffs_other = B_diffs
            elements = A
        else:
            sums = B_sums
            diffs_own = B_diffs
            diffs_other = A_diffs
            elements = B
            
        # Check Sidon property: x + a must be unique, and x+x must be unique
        # Note: We maintain sums as a set of all pair sums (including doubles)
        # When adding x, we need to check if (x+a) is already in sums for any a in elements
        # AND if (x+x) is already in sums.
        # Also, we need to ensure that the new sums (x+a) don't collide with each other?
        # No, if x+a1 == x+a2, then a1==a2, which is impossible since elements are distinct.
        # So we just need to check collisions with existing sums.
        
        valid = True
        new_sums_to_add = set()
        new_diffs_to_add = set()
        
        # Check differences first? Or sums?
        # Difference check: |x - a| must not be in diffs_own (trivially true if we maintain correctly? 
        # No, diffs_own contains differences between existing elements. |x-a| is a new difference.
        # We need to check if |x-a| is in diffs_own? No, that would mean x-a = b-c.
        # Wait, Sidon property is about SUMS. Difference disjointness is about DIFFS.
        
        # 1. Check Difference Disjointness
        # The new differences introduced are |x - a| for all a in elements.
        # These must not be in diffs_other (the other set's diffs).
        # They also must not be in diffs_own? 
        # If |x-a| = |b-c| where b,c in elements, does that violate anything?
        # The Sidon property is on sums. The difference disjointness is between the two sets.
        # Within a set, differences can repeat? 
        # "Pair sums inside each set... must be unique." -> Sidon.
        # "Their nonzero difference sets must be disjoint." -> D_A \cap D_B = empty.
        # It does NOT say D_A has no duplicates or D_A is a Sidon set of differences.
        # However, if A is Sidon, does that imply something about differences?
        # If a+b = c+d, then a-c = d-b.
        # The constraint is just D_A \cap D_B = empty.
        
        # So, for adding x to target:
        # 1. For each a in elements:
        #    d = abs(x - a)
        #    if d in diffs_other: return False
        #    add d to new_diffs_to_add
        # 2. Check Sidon:
        #    for each a in elements:
        #       s = x + a
        #       if s in sums: return False
        #       add s to new_sums_to_add
        #    s2 = 2*x
        #    if s2 in sums: return False
        #    add s2 to new_sums_to_add
        
        # Optimization: 
        # If we find a conflict in diffs, we can break early.
        
        for a in elements:
            d = abs(x - a)
            if d in diffs_other:
                return False, None, None
            new_diffs_to_add.add(d)
            
            s = x + a
            if s in sums:
                return False, None, None
            new_sums_to_add.add(s)
            
        s2 = 2 * x
        if s2 in sums:
            return False, None, None
        new_sums_to_add.add(s2)
        
        return True, new_sums_to_add, new_diffs_to_add

    def greedy_construct(rng):
        nonlocal A_sums, B_sums, A_diffs, B_diffs
        
        A = []
        B = []
        A_sums = set()
        B_sums = set()
        A_diffs = set()
        B_diffs = set()
        
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            # Try A first
            valid, ns, nd = check_add(A, B, x, 'A')
            if valid:
                A.append(x)
                A_sums.update(ns)
                A_diffs.update(nd)
                continue
            
            # Try B
            valid, ns, nd = check_add(A, B, x, 'B')
            if valid:
                B.append(x)
                B_sums.update(ns)
                B_diffs.update(nd)
                continue
                
        return A, B

    def local_search(A, B, rng):
        nonlocal A_sums, B_sums, A_diffs, B_diffs
        
        # We need to rebuild sums/diffs if we modify A/B significantly, 
        # or maintain them. Let's rebuild for simplicity in the swap logic 
        # if the set is small (max ~20-30).
        
        def rebuild(A, B):
            A_sums = set()
            B_sums = set()
            A_diffs = set()
            B_diffs = set()
            for i in range(len(A)):
                for j in range(i, len(A)):
                    A_sums.add(A[i] + A[j])
            for i in range(len(A)):
                for j in range(len(A)):
                    if i != j:
                        A_diffs.add(abs(A[i] - A[j]))
            for i in range(len(B)):
                for j in range(i, len(B)):
                    B_sums.add(B[i] + B[j])
            for i in range(len(B)):
                for j in range(len(B)):
                    if i != j:
                        B_diffs.add(abs(B[i] - B[j]))
                        
        rebuild(A, B)
        
        improved = True
        steps = 0
        max_steps = 1000 # Limit local search time per restart
        
        while improved and steps < max_steps and time.time() < deadline:
            improved = False
            steps += 1
            
            # 1. Try to add any remaining element
            remaining = set(range(1, n + 1)) - set(A) - set(B)
            if remaining:
                # Shuffle remaining for randomness
                rem_list = list(remaining)
                rng.shuffle(rem_list)
                for x in rem_list:
                    valid, ns, nd = check_add(A, B, x, 'A')
                    if valid:
                        A.append(x)
                        A_sums.update(ns)
                        A_diffs.update(nd)
                        improved = True
                        break
                    valid, ns, nd = check_add(A, B, x, 'B')
                    if valid:
                        B.append(x)
                        B_sums.update(ns)
                        B_diffs.update(nd)
                        improved = True
                        break
            
            if improved:
                continue
                
            # 2. If no additions, try swaps to "unstick"
            # Swap an element in A with a candidate from remaining
            if remaining and A:
                a_idx = rng.randrange(len(A))
                a = A[a_idx]
                rem_list = list(remaining)
                rng.shuffle(rem_list)
                for x in rem_list:
                    # Try A' = A - {a} + {x}
                    A_test = A[:]
                    A_test[a_idx] = x
                    
                    # Check if A_test is Sidon and D_A_test disjoint D_B
                    # Fast check:
                    ok = True
                    # Check Sidon for A_test
                    sums_test = set()
                    for i in range(len(A_test)):
                        for j in range(i, len(A_test)):
                            s = A_test[i] + A_test[j]
                            if s in sums_test:
                                ok = False
                                break
                            sums_test.add(s)
                        if not ok: break
                    if ok:
                        # Check Diffs
                        diffs_test = set()
                        for i in range(len(A_test)):
                            for j in range(len(A_test)):
                                if i != j:
                                    d = abs(A_test[i] - A_test[j])
                                    diffs_test.add(d)
                        if diffs_test.isdisjoint(B_diffs):
                            # Accept swap
                            A[a_idx] = x
                            # Rebuild A_sums and A_diffs
                            rebuild(A, B)
                            improved = True
                            break
                            
            # Swap an element in B with a candidate from remaining
            if improved:
                continue
                
            if remaining and B:
                b_idx = rng.randrange(len(B))
                b = B[b_idx]
                rem_list = list(remaining)
                rng.shuffle(rem_list)
                for x in rem_list:
                    B_test = B[:]
                    B_test[b_idx] = x
                    
                    ok = True
                    sums_test = set()
                    for i in range(len(B_test)):
                        for j in range(i, len(B_test)):
                            s = B_test[i] + B_test[j]
                            if s in sums_test:
                                ok = False
                                break
                            sums_test.add(s)
                        if not ok: break
                    if ok:
                        diffs_test = set()
                        for i in range(len(B_test)):
                            for j in range(len(B_test)):
                                if i != j:
                                    d = abs(B_test[i] - B_test[j])
                                    diffs_test.add(d)
                        if diffs_test.isdisjoint(A_diffs):
                            B[b_idx] = x
                            rebuild(A, B)
                            improved = True
                            break

        return A, B

    # Main Loop
    restart = 0
    while time.time() < deadline:
        rng = random.Random(restart)
        A, B = greedy_construct(rng)
        
        # Local search
        A, B = local_search(A, B, rng)
        
        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]
            
        if best_sum >= 32: # Theoretical max is likely lower, but 32 is a good target
            break
            
        restart += 1

    return [best_A, best_B]

