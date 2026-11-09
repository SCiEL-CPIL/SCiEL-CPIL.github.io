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
    
    def can_add_to(A, sums, diffs_other, x):
        """Check if x can be added to set A."""
        if not A and x == 0:
            return False
        # Check Sidon: x+a for all a in A must not be in sums, and 2x must not be in sums
        for a in A:
            if (x + a) in sums:
                return False
        if (2 * x) in sums:
            return False
        # Check disjoint diffs: |x-a| for all a in A must not be in diffs_other
        for a in A:
            d = abs(x - a)
            if d != 0 and d in diffs_other:
                return False
        return True
    
    def add_to_set(A, sums, diffs, x):
        """Add x to set A and update sums and diffs."""
        for a in A:
            sums.add(x + a)
            d = abs(x - a)
            if d != 0:
                diffs.add(d)
        sums.add(2 * x)
        A.append(x)
    
    def polish(A, B, sums_A, sums_B, diffs_A, diffs_B, time_limit):
        """Try to insert skipped elements into the solution."""
        all_used = set(A) | set(B)
        candidates = [x for x in range(1, n + 1) if x not in all_used]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > time_limit:
                break
            # Try adding to A
            if len(A) < 64 and can_add_to(A, sums_A, diffs_B, x):
                add_to_set(A, sums_A, diffs_A, x)
            elif len(B) < 64 and can_add_to(B, sums_B, diffs_A, x):
                add_to_set(B, sums_B, diffs_B, x)
        
        return A, B, sums_A, sums_B, diffs_A, diffs_B
    
    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(2000): # Max restarts, but will break on time
        if elapsed() > deadline - 3.0:
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
            if elapsed() > deadline - 3.0:
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
                    if can_add_to(A, sums_A, diffs_B, x):
                        add_to_set(A, sums_A, diffs_A, x)
                        added = True
                        break
                else:
                    if len(B) >= 64:
                        continue
                    if can_add_to(B, sums_B, diffs_A, x):
                        add_to_set(B, sums_B, diffs_B, x)
                        added = True
                        break
            
        if A and B:
            # Polish pass: try to insert skipped elements
            polish_time_limit = min(deadline - 2.0, time.time() + 0.5)
            A, B, sums_A, sums_B, diffs_A, diffs_B = polish(
                A, B, sums_A, sums_B, diffs_A, diffs_B, polish_time_limit
            )
            
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
