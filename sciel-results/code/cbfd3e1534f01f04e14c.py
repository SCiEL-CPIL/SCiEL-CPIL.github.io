import random
import math
import time

def construct():
    deadline = time.time() + 75  # safety margin before 90s limit
    
    def score(A, B):
        """Return number of distinct pairwise-coprime sums, or 0 if invalid."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        # Check pairwise coprime
        lst = sorted(sums)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) != 1:
                    return 0
        return len(lst)
    
    def valid(A, B):
        """Check if A and B are valid (3 distinct each, in [1,128])."""
        if len(A) != 3 or len(B) != 3:
            return False
        if len(set(A)) != 3 or len(set(B)) != 3:
            return False
        if any(x < 1 or x > 128 for x in A):
            return False
        if any(x < 1 or x > 128 for x in B):
            return False
        return True
    
    def local_search(A, B, max_iter=2000):
        """Improve A, B via local search."""
        best_A = list(A)
        best_B = list(B)
        best_score = score(best_A, best_B)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try changing one element of A or B
            improved = False
            
            # Try changing A[i]
            for i in range(3):
                old = best_A[i]
                for new_val in range(1, 129):
                    if new_val == old:
                        continue
                    new_A = list(best_A)
                    new_A[i] = new_val
                    if len(set(new_A)) != 3:
                        continue
                    s = score(new_A, best_B)
                    if s > best_score:
                        best_A = new_A
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # Try changing B[i]
            for i in range(3):
                old = best_B[i]
                for new_val in range(1, 129):
                    if new_val == old:
                        continue
                    new_B = list(best_B)
                    new_B[i] = new_val
                    if len(set(new_B)) != 3:
                        continue
                    s = score(best_A, new_B)
                    if s > best_score:
                        best_B = new_B
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return best_A, best_B, best_score
    
    def random_candidate():
        """Generate a random valid candidate."""
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        return A, B
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)
    
    # Phase 1: Local search from seed
    best_A, best_B, best_score = local_search(best_A, best_B, max_iter=5000)
    
    # Phase 2: Random restarts
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        
        A, B = random_candidate()
        s = score(A, B)
        if s > best_score:
            best_A, best_B, best_score = A, B, s
        
        # Local search from this candidate
        if s >= 3:  # Only pursue if there's some promise
            A2, B2, s2 = local_search(A, B, max_iter=1000)
            if s2 > best_score:
                best_A, best_B, best_score = A2, B2, s2
        
        if best_score >= 9:
            break
    
    return best_A, best_B
