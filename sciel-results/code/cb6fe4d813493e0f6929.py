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

    def try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
        """Try to insert x into A or B, returning True if successful."""
        # Try A first if smaller, else B first
        if len(A) < len(B):
            order = [0, 1]
        elif len(B) < len(A):
            order = [1, 0]
        else:
            order = random.sample([0, 1], 2)
        
        for side in order:
            if side == 0:
                if len(A) >= 64:
                    continue
                valid_sidon = True
                for a in A:
                    if (x + a) in sums_A:
                        valid_sidon = False
                        break
                if valid_sidon and (2*x) not in sums_A:
                    valid_diffs = True
                    for a in A:
                        d = abs(x - a)
                        if d != 0 and d in diffs_B:
                            valid_diffs = False
                            break
                    if valid_diffs:
                        A.append(x)
                        for a in A[:-1]:
                            sums_A.add(x + a)
                            d = abs(x - a)
                            if d != 0:
                                diffs_A.add(d)
                        sums_A.add(2*x)
                        return True
            else:
                if len(B) >= 64:
                    continue
                valid_sidon = True
                for b in B:
                    if (x + b) in sums_B:
                        valid_sidon = False
                        break
                if valid_sidon and (2*x) not in sums_B:
                    valid_diffs = True
                    for b in B:
                        d = abs(x - b)
                        if d != 0 and d in diffs_A:
                            valid_diffs = False
                            break
                    if valid_diffs:
                        B.append(x)
                        for b in B[:-1]:
                            sums_B.add(x + b)
                            d = abs(x - b)
                            if d != 0:
                                diffs_B.add(d)
                        sums_B.add(2*x)
                        return True
        return False

    def polish(A, B):
        """Local search: try inserting skipped elements into the solution."""
        sums_A = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        sums_B = set()
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        diffs_A = set()
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                diffs_A.add(abs(A[j] - A[i]))
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                diffs_B.add(abs(B[j] - B[i]))
        
        in_A = set(A)
        in_B = set(B)
        used = in_A | in_B
        remaining = [x for x in range(1, n+1) if x not in used]
        
        # Try multiple passes of insertion
        for pass_num in range(3):
            if elapsed() > deadline - 3.0:
                break
            improved = False
            for x in remaining[:]:
                if elapsed() > deadline - 3.0:
                    break
                if try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B):
                    remaining.remove(x)
                    improved = True
            if not improved:
                break
        
        return A, B

    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(2000):
        if elapsed() > deadline - 5.0:
            break
            
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        # Candidate ordering: mix of random and low-first strategies
        if restart % 3 == 0:
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        elif restart % 3 == 1:
            # Low-first with some randomization
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random() * x)
        else:
            # Pure random
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
            
            try_insert(x, A, B, sums_A, sums_B, diffs_A, diffs_B)
        
        if A and B:
            # Polish pass
            A, B = polish(A, B)
            
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
