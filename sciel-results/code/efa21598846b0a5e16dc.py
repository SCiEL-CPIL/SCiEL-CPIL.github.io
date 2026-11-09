import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 256

    def elapsed():
        return time.time() - start_time

    def is_valid_solution(A, B):
        if not A or not B:
            return False
        if len(A) > 64 or len(B) > 64:
            return False
        
        # Check Sidon for A
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                s = A[i] + A[j]
                if s in sums_A:
                    return False
                sums_A.add(s)
        
        # Check Sidon for B
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                s = B[i] + B[j]
                if s in sums_B:
                    return False
                sums_B.add(s)
        
        # Check disjoint differences
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def build_conflict_graph(candidates):
        """
        Build a conflict graph where vertices are (element, set_assignment) pairs.
        Edges represent conflicts: sum collisions within a set, or difference overlaps between sets.
        """
        # We'll model this as: for each element x in [1,n], we can assign it to A, B, or neither.
        # Conflicts:
        # 1. If x,y both in A: x+y must be unique (including x=x)
        # 2. If x,y both in B: x+y must be unique (including x=x)
        # 3. If x,y in A and u,v in B: |x-y| != |u-v| for all pairs
        
        # This is complex. Instead, use a simpler approach:
        # Greedily build A and B with a different strategy.
        pass

    def greedy_mis_heuristic(candidates, K, seed):
        """
        Use a greedy coloring-like heuristic to build A and B.
        A gets elements with small differences (<= K), B gets large differences (> K).
        Use random tie-breaking to escape local minima.
        """
        random.seed(seed)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Sort candidates with some randomization
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        for x in shuffled:
            # Try adding to A
            can_add_A = True
            for s in A:
                if (x + s) in sums_A:
                    can_add_A = False
                    break
            if can_add_A and (2*x) in sums_A:
                can_add_A = False
            if can_add_A:
                for s in A:
                    d = abs(x - s)
                    if d > K:
                        can_add_A = False
                        break
                    if d in diffs_B:
                        can_add_A = False
                        break
            
            # Try adding to B
            can_add_B = True
            for s in B:
                if (x + s) in sums_B:
                    can_add_B = False
                    break
            if can_add_B and (2*x) in sums_B:
                can_add_B = False
            if can_add_B:
                for s in B:
                    d = abs(x - s)
                    if d <= K:
                        can_add_B = False
                        break
                    if d in diffs_A:
                        can_add_B = False
                        break
            
            # Choose with random tie-breaking
            if can_add_A and can_add_B:
                if random.random() < 0.5:
                    # Add to A
                    A.append(x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        diffs_A.add(abs(x - s))
                    sums_A.add(2*x)
                else:
                    # Add to B
                    B.append(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(abs(x - s))
                    sums_B.add(2*x)
            elif can_add_A:
                A.append(x)
                for s in A[:-1]:
                    sums_A.add(x + s)
                    diffs_A.add(abs(x - s))
                sums_A.add(2*x)
            elif can_add_B:
                B.append(x)
                for s in B[:-1]:
                    sums_B.add(x + s)
                    diffs_B.add(abs(x - s))
                sums_B.add(2*x)
            
            if len(A) >= 64 and len(B) >= 64:
                break
        
        return A, B

    def polish_sets(A, B, K):
        """
        Try to improve A and B by swapping or adding elements.
        """
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
        
        in_A = set(A)
        in_B = set(B)
        all_elements = list(range(1, 257))
        
        improved = True
        while improved and elapsed() < deadline - 2.0:
            improved = False
            
            # Try adding new elements
            for x in all_elements:
                if x in in_A or x in in_B:
                    continue
                
                # Try adding to A
                can_add_A = True
                for s in A:
                    if (x + s) in sums_A:
                        can_add_A = False
                        break
                if can_add_A and (2*x) in sums_A:
                    can_add_A = False
                if can_add_A:
                    for s in A:
                        d = abs(x - s)
                        if d > K or d in diffs_B:
                            can_add_A = False
                            break
                
                if can_add_A and len(A) < 64:
                    A.append(x)
                    in_A.add(x)
                    for s in A[:-1]:
                        sums_A.add(x + s)
                        diffs_A.add(abs(x - s))
                    sums_A.add(2*x)
                    improved = True
                    continue
                
                # Try adding to B
                can_add_B = True
                for s in B:
                    if (x + s) in sums_B:
                        can_add_B = False
                        break
                if can_add_B and (2*x) in sums_B:
                    can_add_B = False
                if can_add_B:
                    for s in B:
                        d = abs(x - s)
                        if d <= K or d in diffs_A:
                            can_add_B = False
                            break
                
                if can_add_B and len(B) < 64:
                    B.append(x)
                    in_B.add(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(abs(x - s))
                    sums_B.add(2*x)
                    improved = True
            
            # Try swapping elements between A and B
            for i in range(len(A)):
                if elapsed() > deadline - 2.0:
                    break
                x = A[i]
                # Temporarily remove from A
                A_new = A[:i] + A[i+1:]
                in_A.discard(x)
                
                # Recompute sums and diffs for A
                sums_A_new = set()
                diffs_A_new = set()
                for p in range(len(A_new)):
                    for q in range(p, len(A_new)):
                        sums_A_new.add(A_new[p] + A_new[q])
                for p in range(len(A_new)):
                    for q in range(p+1, len(A_new)):
                        diffs_A_new.add(abs(A_new[q] - A_new[p]))
                
                # Try adding x to B
                can_add_B = True
                for s in B:
                    if (x + s) in sums_B:
                        can_add_B = False
                        break
                if can_add_B and (2*x) in sums_B:
                    can_add_B = False
                if can_add_B:
                    for s in B:
                        d = abs(x - s)
                        if d <= K or d in diffs_A_new:
                            can_add_B = False
                            break
                
                if can_add_B and len(B) < 64:
                    # Perform swap
                    A = A_new
                    B.append(x)
                    in_B.add(x)
                    for s in B[:-1]:
                        sums_B.add(x + s)
                        diffs_B.add(abs(x - s))
                    sums_B.add(2*x)
                    # Update A's data structures
                    sums_A = sums_A_new
                    diffs_A = diffs_A_new
                    improved = True
                else:
                    # Restore
                    A = A[:i] + [x] + A[i:]
                    in_A.add(x)
                    sums_A = set()
                    diffs_A = set()
                    for p in range(len(A)):
                        for q in range(p, len(A)):
                            sums_A.add(A[p] + A[q])
                    for p in range(len(A)):
                        for q in range(p+1, len(A)):
                            diffs_A.add(abs(A[q] - A[p]))
        
        return A, B

    best_score = 0
    best_sets = ([1], [2])
    
    # Try different K values
    for K in range(80, 180, 4):
        if elapsed() > deadline - 5.0:
            break
        
        candidates = list(range(1, n + 1))
        
        # Multiple restarts with different seeds
        for restart in range(30):
            if elapsed() > deadline - 5.0:
                break
            
            A, B = greedy_mis_heuristic(candidates, K, seed=restart * 1000 + K)
            
            if A and B:
                # Polish
                A, B = polish_sets(A, B, K)
                
                if is_valid_solution(A, B):
                    score = len(A) + len(B)
                    if score > best_score:
                        best_score = score
                        best_sets = (A[:], B[:])
    
    # Fallback
    if best_score == 0:
        best_sets = ([1], [2])

    return list(best_sets)
