"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    n = 256
    max_size = 64
    
    def _check_sidon(s):
        """Check if set s is a Sidon set (all pair sums unique)."""
        sums = set()
        for i, a in enumerate(s):
            for b in s[i:]:
                sm = a + b
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def _get_diffs(s):
        """Get all nonzero differences from set s."""
        diffs = set()
        for a in s:
            for b in s:
                if a != b:
                    d = b - a
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def _validate_pair(A, B):
        """Validate that A and B are both Sidon sets and their nonzero diffs are disjoint."""
        if not A or not B:
            return False
        if len(A) > max_size or len(B) > max_size:
            return False
        if not _check_sidon(A):
            return False
        if not _check_sidon(B):
            return False
        diffA = _get_diffs(A)
        diffB = _get_diffs(B)
        if diffA & diffB:
            return False
        return True
    
    def _greedy_build(order, start_set_A, start_set_B, deadline):
        """Greedy build: iterate through order, add to A or B if valid."""
        A = list(start_set_A)
        B = list(start_set_B)
        diffsA = _get_diffs(A) if A else set()
        diffsB = _get_diffs(B) if B else set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Try adding to A
            trialA = A + [x]
            if len(trialA) <= max_size and _check_sidon(trialA):
                new_diffsA = _get_diffs(trialA)
                if not (new_diffsA & diffsB):
                    A = trialA
                    diffsA = new_diffsA
                    continue
            
            # Try adding to B
            trialB = B + [x]
            if len(trialB) <= max_size and _check_sidon(trialB):
                new_diffsB = _get_diffs(trialB)
                if not (diffsA & new_diffsB):
                    B = trialB
                    diffsB = new_diffsB
                    continue
        
        return A, B
    
    def _local_search(A, B, deadline, max_iter=200):
        """Try to improve by swapping elements or adding/removing."""
        A = list(A)
        B = list(B)
        best_score = len(A) + len(B)
        best_A, best_B = list(A), list(B)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a random element from 1..n to A or B
            for _ in range(5):
                if time.time() > deadline:
                    break
                x = random.randint(1, n)
                
                # Try adding to A
                if x not in A and len(A) < max_size:
                    trialA = A + [x]
                    if _check_sidon(trialA):
                        new_diffsA = _get_diffs(trialA)
                        diffsB = _get_diffs(B)
                        if not (new_diffsA & diffsB):
                            A = trialA
                            best_score = len(A) + len(B)
                            best_A, best_B = list(A), list(B)
                            improved = True
                            break
                
                # Try adding to B
                if x not in B and len(B) < max_size:
                    trialB = B + [x]
                    if _check_sidon(trialB):
                        new_diffsB = _get_diffs(trialB)
                        diffsA = _get_diffs(A)
                        if not (diffsA & new_diffsB):
                            B = trialB
                            best_score = len(B) + len(A)
                            best_A, best_B = list(A), list(B)
                            improved = True
                            break
            
            # Try swapping an element out of A for a new one
            if len(A) > 0:
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(A) - 1)
                    old = A[idx]
                    x = random.randint(1, n)
                    if x == old or x in A:
                        continue
                    trialA = [a for i, a in enumerate(A) if i != idx] + [x]
                    trialA.sort()
                    if _check_sidon(trialA):
                        new_diffsA = _get_diffs(trialA)
                        diffsB = _get_diffs(B)
                        if not (new_diffsA & diffsB):
                            A = trialA
                            best_score = len(A) + len(B)
                            best_A, best_B = list(A), list(B)
                            improved = True
                            break
            
            # Try swapping an element out of B for a new one
            if len(B) > 0:
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(B) - 1)
                    old = B[idx]
                    x = random.randint(1, n)
                    if x == old or x in B:
                        continue
                    trialB = [b for i, b in enumerate(B) if i != idx] + [x]
                    trialB.sort()
                    if _check_sidon(trialB):
                        new_diffsB = _get_diffs(trialB)
                        diffsA = _get_diffs(A)
                        if not (diffsA & new_diffsB):
                            B = trialB
                            best_score = len(B) + len(A)
                            best_A, best_B = list(A), list(B)
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_A, best_B, best_score
    
    # Multiple restarts with different random orderings
    best_A = []
    best_B = []
    best_total = 0
    
    # First, try a deterministic greedy from scratch with natural ordering
    A0, B0 = _greedy_build(list(range(1, n + 1)), [], [], deadline)
    total0 = len(A0) + len(B0)
    if total0 > best_total:
        best_A, best_B = A0, B0
        best_total = total0
    
    # Then try randomized restarts
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A, B = _greedy_build(order, [], [], deadline)
        total = len(A) + len(B)
        
        # Local search to improve
        A_ls, B_ls, total_ls = _local_search(A, B, deadline, max_iter=100)
        if total_ls > total:
            A, B, total = A_ls, B_ls, total_ls
        
        if total > best_total:
            best_A, best_B = A, B
            best_total = total
    
    # Final validation
    if not _validate_pair(best_A, best_B):
        # Fallback to simple valid sets
        best_A = [1, 2]
        best_B = [3, 4]
    
    return [sorted(best_A), sorted(best_B)]
# EVOLVE-BLOCK-END
