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
                d = abs(A[j] - A[i])
                diffs_A.add(d)
        diffs_B = set()
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = abs(B[j] - B[i])
                diffs_B.add(d)
                
        if diffs_A & diffs_B:
            return False
            
        return True

    def polish(A, B):
        """Local search: try inserting skipped elements into the solution."""
        # Rebuild masks
        M_A = 0
        M_B = 0
        for a in A:
            M_A |= (1 << a)
        for b in B:
            M_B |= (1 << b)
            
        sum_mask_A = 0
        sum_mask_B = 0
        M_DA = 0
        M_DB = 0
        
        # Compute sums and diffs for A
        for i in range(len(A)):
            for j in range(i, len(A)):
                sum_mask_A |= (1 << (A[i] + A[j]))
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                d = abs(A[j] - A[i])
                M_DA |= (1 << d)
                
        # Compute sums and diffs for B
        for i in range(len(B)):
            for j in range(i, len(B)):
                sum_mask_B |= (1 << (B[i] + B[j]))
        for i in range(len(B)):
            for j in range(i+1, len(B)):
                d = abs(B[j] - B[i])
                M_DB |= (1 << d)
        
        in_A = set(A)
        in_B = set(B)
        used = in_A | in_B
        remaining = [x for x in range(1, n+1) if x not in used]
        
        # Try multiple passes of insertion
        for pass_num in range(5):
            if elapsed() > deadline - 3.0:
                break
            improved = False
            for x in remaining[:]:
                if elapsed() > deadline - 3.0:
                    break
                
                # Try insert into A
                success = False
                if len(A) < 64:
                    # Sum check
                    if ((M_A << x) & sum_mask_A) == 0 and ((sum_mask_A >> (2*x)) & 1) == 0:
                        # Diff check vs B
                        if ((M_A >> x) & M_DB) == 0 and ((M_DB << x) & M_A) == 0:
                            # Insert
                            A.append(x)
                            M_A |= (1 << x)
                            for a in A[:-1]:
                                sum_mask_A |= (1 << (x + a))
                                d = abs(x - a)
                                M_DA |= (1 << d)
                            sum_mask_A |= (1 << (2*x))
                            success = True
                            
                # Try insert into B
                if not success and len(B) < 64:
                    # Sum check
                    if ((M_B << x) & sum_mask_B) == 0 and ((sum_mask_B >> (2*x)) & 1) == 0:
                        # Diff check vs A
                        if ((M_B >> x) & M_DA) == 0 and ((M_DA << x) & M_B) == 0:
                            # Insert
                            B.append(x)
                            M_B |= (1 << x)
                            for b in B[:-1]:
                                sum_mask_B |= (1 << (x + b))
                                d = abs(x - b)
                                M_DB |= (1 << d)
                            sum_mask_B |= (1 << (2*x))
                            success = True
                            
                if success:
                    remaining.remove(x)
                    improved = True
                    
            if not improved:
                break
        
        return A, B

    best_score = 0
    best_sets = ([1], [2]) # Fallback
    
    # Randomized Greedy with restarts
    for restart in range(5000):
        if elapsed() > deadline - 5.0:
            break
            
        A = []
        B = []
        M_A = 0
        M_B = 0
        sum_mask_A = 0
        sum_mask_B = 0
        M_DA = 0
        M_DB = 0
        
        # Candidate ordering
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if elapsed() > deadline - 5.0:
                break
            
            # Try A
            inserted = False
            if len(A) < 64:
                if ((M_A << x) & sum_mask_A) == 0 and ((sum_mask_A >> (2*x)) & 1) == 0:
                    if ((M_A >> x) & M_DB) == 0 and ((M_DB << x) & M_A) == 0:
                        A.append(x)
                        M_A |= (1 << x)
                        for a in A[:-1]:
                            sum_mask_A |= (1 << (x + a))
                            d = abs(x - a)
                            M_DA |= (1 << d)
                        sum_mask_A |= (1 << (2*x))
                        inserted = True
            # Try B
            if not inserted and len(B) < 64:
                if ((M_B << x) & sum_mask_B) == 0 and ((sum_mask_B >> (2*x)) & 1) == 0:
                    if ((M_B >> x) & M_DA) == 0 and ((M_DA << x) & M_B) == 0:
                        B.append(x)
                        M_B |= (1 << x)
                        for b in B[:-1]:
                            sum_mask_B |= (1 << (x + b))
                            d = abs(x - b)
                            M_DB |= (1 << d)
                        sum_mask_B |= (1 << (2*x))
                        inserted = True
                        
        if A and B:
            # Polish pass
            A, B = polish(A, B)
            
            if is_valid_solution(A, B):
                score = len(A) + len(B)
                if score > best_score:
                    best_score = score
                    best_sets = (A[:], B[:])
    
    return list(best_sets)
