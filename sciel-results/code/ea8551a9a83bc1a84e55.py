import time
import random
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 256
    MAX_SIZE = 64
    
    best_score = 0
    best_A = [1]
    best_B = [2]
    
    # Precompute all possible differences for quick lookup
    # diffs[x][y] = abs(x-y) for x != y, 0 for x==y
    
    def valid_sidon(set_list):
        """Check if a list forms a valid Sidon set."""
        sums = set()
        n = len(set_list)
        for i in range(n):
            for j in range(i, n):
                s = set_list[i] + set_list[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def valid_disjoint_diffs(A, B):
        """Check if nonzero difference sets of A and B are disjoint."""
        diffs_A = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    d = abs(A[i] - A[j])
                    if d > 0:
                        diffs_A.add(d)
        
        diffs_B = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    d = abs(B[i] - B[j])
                    if d > 0:
                        diffs_B.add(d)
        
        return len(diffs_A & diffs_B) == 0
    
    def is_valid_solution(A, B):
        """Full validity check."""
        if not A or not B:
            return False
        if len(A) > MAX_SIZE or len(B) > MAX_SIZE:
            return False
        if not valid_sidon(A):
            return False
        if not valid_sidon(B):
            return False
        if not valid_disjoint_diffs(A, B):
            return False
        return True
    
    def get_diffs(S):
        """Get nonzero difference set of S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    d = abs(S[i] - S[j])
                    if d > 0:
                        diffs.add(d)
        return diffs
    
    def get_sums(S):
        """Get pair sums of S (including a+a)."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums
    
    # Conflict graph approach
    # For each element x in [1, 256], determine which set it can go to
    # and what conflicts it creates
    
    def can_add_to_set(x, S, diffs_S, sums_S, diffs_other):
        """Check if x can be added to set S without violating constraints."""
        # Check Sidon condition for S
        for a in S:
            d = abs(x - a)
            if d in diffs_S:
                return False
            if d in diffs_other:
                return False
        
        # Check if adding x creates duplicate differences within S
        # This happens if |x - a1| == |x - a2| for a1 != a2
        # Which means 2x == a1 + a2
        if 2 * x in sums_S:
            return False
            
        return True
    
    def add_element(x, S, diffs_S, sums_S):
        """Add x to set S, updating diffs and sums."""
        S.append(x)
        sums_S.add(2 * x)
        for a in S:
            if a != x:
                d = abs(x - a)
                diffs_S.add(d)
                sums_S.add(x + a)
    
    def greedy_build(seed_val):
        """Build a solution using randomized greedy with local search."""
        random.seed(seed_val)
        
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        sums_A = set()
        sums_B = set()
        
        # Candidate elements
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        # Greedy phase: try to add each candidate
        for x in candidates:
            valid_A = can_add_to_set(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add_to_set(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                # Choose the set with more room
                if len(A) < len(B):
                    add_element(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_element(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_element(x, A, diffs_A, sums_A)
                    else:
                        add_element(x, B, diffs_B, sums_B)
            elif valid_A:
                add_element(x, A, diffs_A, sums_A)
            elif valid_B:
                add_element(x, B, diffs_B, sums_B)
        
        # Local search phase: try to swap elements to improve
        # We'll try removing an element from one set and adding it to the other,
        # or removing and trying a different element
        
        improved = True
        max_iterations = 50
        iteration = 0
        
        while improved and iteration < max_iterations and time.time() < deadline:
            improved = False
            iteration += 1
            
            # Try swapping: remove an element from A, try to add to B or vice versa
            for _ in range(10):
                if time.time() >= deadline:
                    break
                
                # Try moving an element from A to B
                if len(A) > 0:
                    idx = random.randint(0, len(A) - 1)
                    x = A[idx]
                    
                    # Temporarily remove x from A
                    A.pop(idx)
                    # Rebuild diffs and sums for A
                    diffs_A_new = get_diffs(A)
                    sums_A_new = get_sums(A)
                    
                    # Try to add x to B
                    if can_add_to_set(x, B, diffs_B, sums_B, diffs_A_new):
                        A_final = A.copy()
                        A_final.append(x)
                        B_final = B.copy()
                        
                        # Check if this improves
                        score_old = len(A) + len(B)
                        score_new = len(A_final) + len(B_final)
                        
                        if score_new > score_old:
                            A = A_final
                            B = B_final
                            diffs_A = get_diffs(A)
                            sums_A = get_sums(A)
                            diffs_B = get_diffs(B)
                            sums_B = get_sums(B)
                            improved = True
                            continue
                    
                    # Try adding a different element to A
                    # Find an element not in A or B that can be added to A
                    remaining = [y for y in range(1, N + 1) if y not in A and y not in B]
                    random.shuffle(remaining)
                    
                    for y in remaining[:5]:  # Try up to 5 candidates
                        if can_add_to_set(y, A, diffs_A_new, sums_A_new, diffs_B):
                            A_new = A.copy()
                            A_new.append(y)
                            A_new.sort()
                            
                            score_old = len(A) + len(B)
                            score_new = len(A_new) + len(B)
                            
                            if score_new > score_old:
                                A = A_new
                                diffs_A = get_diffs(A)
                                sums_A = get_sums(A)
                                improved = True
                                break
                    
                    if not improved:
                        # Restore x to A
                        A.append(x)
                        A.sort()
                        diffs_A = get_diffs(A)
                        sums_A = get_sums(A)
                
                # Try moving an element from B to A
                if len(B) > 0:
                    idx = random.randint(0, len(B) - 1)
                    x = B[idx]
                    
                    # Temporarily remove x from B
                    B.pop(idx)
                    diffs_B_new = get_diffs(B)
                    sums_B_new = get_sums(B)
                    
                    # Try to add x to A
                    if can_add_to_set(x, A, diffs_A, sums_A, diffs_B_new):
                        B_final = B.copy()
                        B_final.append(x)
                        
                        score_old = len(A) + len(B)
                        score_new = len(A) + len(B_final)
                        
                        if score_new > score_old:
                            B = B_final
                            diffs_A = get_diffs(A)
                            sums_A = get_sums(A)
                            diffs_B = get_diffs(B)
                            sums_B = get_sums(B)
                            improved = True
                            continue
                    
                    # Try adding a different element to B
                    remaining = [y for y in range(1, N + 1) if y not in A and y not in B]
                    random.shuffle(remaining)
                    
                    for y in remaining[:5]:
                        if can_add_to_set(y, B, diffs_B_new, sums_B_new, diffs_A):
                            B_new = B.copy()
                            B_new.append(y)
                            B_new.sort()
                            
                            score_old = len(A) + len(B)
                            score_new = len(A) + len(B_new)
                            
                            if score_new > score_old:
                                B = B_new
                                diffs_B = get_diffs(B)
                                sums_B = get_sums(B)
                                improved = True
                                break
                    
                    if not improved:
                        # Restore x to B
                        B.append(x)
                        B.sort()
                        diffs_B = get_diffs(B)
                        sums_B = get_sums(B)
        
        # Second greedy pass with remaining elements
        remaining = [x for x in range(1, N + 1) if x not in A and x not in B]
        random.shuffle(remaining)
        
        for x in remaining:
            valid_A = can_add_to_set(x, A, diffs_A, sums_A, diffs_B)
            valid_B = can_add_to_set(x, B, diffs_B, sums_B, diffs_A)
            
            if valid_A and valid_B:
                if len(A) < len(B):
                    add_element(x, A, diffs_A, sums_A)
                elif len(B) < len(A):
                    add_element(x, B, diffs_B, sums_B)
                else:
                    if random.random() < 0.5:
                        add_element(x, A, diffs_A, sums_A)
                    else:
                        add_element(x, B, diffs_B, sums_B)
            elif valid_A:
                add_element(x, A, diffs_A, sums_A)
            elif valid_B:
                add_element(x, B, diffs_B, sums_B)
        
        A.sort()
        B.sort()
        
        # Final validity check
        if is_valid_solution(A, B):
            return A, B
        else:
            return [], []
    
    # Multiple restarts with different seeds
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        seed_val = attempts * 7919 + 12345
        
        A, B = greedy_build(seed_val)
        
        if A and B:
            score = len(A) + len(B)
            if score > best_score:
                best_score = score
                best_A = A.copy()
                best_B = B.copy()
                
                # Early stop if we hit a very high score
                if best_score >= 50:
                    break
    
    # If we didn't find anything, use a simple fallback
    if not best_A or not best_B:
        best_A = [1]
        best_B = [2]
    
    return [best_A, best_B]
