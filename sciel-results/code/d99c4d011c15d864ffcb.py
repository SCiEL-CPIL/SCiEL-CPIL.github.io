"""Direct construction of two Sidon sets with disjoint difference sets."""
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
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
                d = A[j] - A[i]
                diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = B[j] - B[i]
                diffs_B.add(d)
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def can_add_to_set(x, S, sums_S, diffs_S, diffs_other):
        """Check if x can be added to set S."""
        for s in S:
            if (x + s) in sums_S:
                return False
        if (2 * x) in sums_S:
            return False
        for s in S:
            d = abs(x - s)
            if d != 0 and d in diffs_other:
                return False
        return True

    def add_to_set(x, S, sums_S, diffs_S):
        """Add x to set S and update auxiliary structures."""
        S.append(x)
        for s in S[:-1]:
            sums_S.add(x + s)
            d = abs(x - s)
            if d != 0:
                diffs_S.add(d)
        sums_S.add(2 * x)

    def local_search_polish(A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to insert elements that were skipped during greedy."""
        all_elements = set(range(1, n + 1))
        used = set(A) | set(B)
        skipped = sorted(all_elements - used)
        
        improved = True
        while improved and elapsed() < deadline - 5.0:
            improved = False
            for x in skipped:
                if elapsed() > deadline - 5.0:
                    break
                if x in used:
                    continue
                # Try adding to A
                if len(A) < 64 and can_add_to_set(x, A, sums_A, diffs_A, diffs_B):
                    add_to_set(x, A, sums_A, diffs_A)
                    used.add(x)
                    improved = True
                    break
                # Try adding to B
                if len(B) < 64 and can_add_to_set(x, B, sums_B, diffs_B, diffs_A):
                    add_to_set(x, B, sums_B, diffs_B)
                    used.add(x)
                    improved = True
                    break
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B

    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(1000): # Max restarts, but will break on time
        if elapsed() > deadline - 5.0:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Shuffle candidate elements
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
                
            # Determine priority: try to add to the smaller set first
            # If sizes are equal, pick randomly
            if len(A) < len(B):
                order = [0, 1]
            elif len(B) < len(A):
                order = [1, 0]
            else:
                order = random.sample([0, 1], 2)
            
            added = False
            for side in order:
                if side == 0:
                    if len(A) >= 64:
                        continue
                    if can_add_to_set(x, A, sums_A, diffs_A, diffs_B):
                        add_to_set(x, A, sums_A, diffs_A)
                        added = True
                        break
                else:
                    if len(B) >= 64:
                        continue
                    if can_add_to_set(x, B, sums_B, diffs_B, diffs_A):
                        add_to_set(x, B, sums_B, diffs_B)
                        added = True
                        break
            
        # Local search polish: try to add skipped elements
        if A and B:
            A, B, sums_A, sums_B, diffs_A, diffs_B = local_search_polish(
                A, B, sums_A, sums_B, diffs_A, diffs_B
            )
            
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
