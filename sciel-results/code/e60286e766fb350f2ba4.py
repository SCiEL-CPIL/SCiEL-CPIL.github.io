# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

def _check_sidon(s):
    """Check if set s is a Sidon set (all pair sums a+b with a<=b are unique)."""
    if len(s) <= 1:
        return True
    sums = []
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.append(s[i] + s[j])
    return len(sums) == len(set(sums))

def _diff_set(s):
    """Compute nonzero difference set of s."""
    diffs = set()
    for i in range(len(s)):
        for j in range(len(s)):
            if i != j:
                diffs.add(s[i] - s[j])
    return diffs

def _can_add(s, x, other_diffs):
    """Check if x can be added to Sidon set s without violating Sidon or difference constraints."""
    # Check Sidon: all pair sums including x+x must be unique
    for a in s:
        # Check a+x doesn't equal any existing sum
        pass
    # More efficient: check new sums involving x
    new_sums = [x + x] + [x + a for a in s]
    # Existing sums
    existing_sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            existing_sums.add(s[i] + s[j])
    for ns in new_sums:
        if ns in existing_sums:
            return False
    # Check difference disjointness
    new_diffs = set()
    for a in s:
        new_diffs.add(x - a)
        new_diffs.add(a - x)
    if new_diffs & other_diffs:
        return False
    return True

def _remove_from_sets(A, B, x):
    """Remove x from A and B if present."""
    if x in A:
        A.remove(x)
    if x in B:
        B.remove(x)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    best_A = []
    best_B = []
    best_score = 0
    
    # Try multiple random orderings
    rng = random.Random(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random ordering of elements 1..n
        elements = list(range(1, n + 1))
        rng.shuffle(elements)
        
        A = []
        B = []
        diffA = set()
        diffB = set()
        
        for x in elements:
            if time.time() > deadline:
                break
            
            # Try adding to A
            can_A = _can_add(A, x, diffB)
            # Try adding to B
            can_B = _can_add(B, x, diffA)
            
            if can_A and can_B:
                # Add to the smaller set to balance
                if len(A) <= len(B):
                    A.append(x)
                    # Update diffA
                    for a in A[:-1]:
                        diffA.add(x - a)
                        diffA.add(a - x)
                    diffA.add(0)  # x-x, but we only care about nonzero
                    diffA.discard(0)
                else:
                    B.append(x)
                    for b in B[:-1]:
                        diffB.add(x - b)
                        diffB.add(b - x)
                    diffB.discard(0)
            elif can_A:
                A.append(x)
                for a in A[:-1]:
                    diffA.add(x - a)
                    diffA.add(a - x)
                diffA.discard(0)
            elif can_B:
                B.append(x)
                for b in B[:-1]:
                    diffB.add(x - b)
                    diffB.add(b - x)
                diffB.discard(0)
        
        # Local search: try swaps to improve
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score
        
        # Local search with swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            A_set = set(A)
            B_set = set(B)
            dA = _diff_set(A)
            dB = _diff_set(B)
            
            # Try moving an element from A to B or vice versa
            for x in A[:]:
                if time.time() > deadline:
                    break
                # Try removing x from A and adding to B
                A2 = [a for a in A if a != x]
                B2 = B + [x]
                if _check_sidon(A2) and _check_sidon(B2):
                    dA2 = _diff_set(A2)
                    dB2 = _diff_set(B2)
                    if not (dA2 & dB2):
                        if len(A2) + len(B2) > len(A) + len(B):
                            A = A2
                            B = B2
                            improved = True
                            break
                        elif len(A2) + len(B2) == len(A) + len(B) and len(A2) < len(A):
                            # Balance: move from larger to smaller
                            if len(A) > len(B):
                                A = A2
                                B = B2
                                improved = True
                                break
            
            if improved:
                continue
            
            # Try moving from B to A
            for x in B[:]:
                if time.time() > deadline:
                    break
                A2 = A + [x]
                B2 = [b for b in B if b != x]
                if _check_sidon(A2) and _check_sidon(B2):
                    dA2 = _diff_set(A2)
                    dB2 = _diff_set(B2)
                    if not (dA2 & dB2):
                        if len(A2) + len(B2) > len(A) + len(B):
                            A = A2
                            B = B2
                            improved = True
                            break
                        elif len(A2) + len(B2) == len(A) + len(B) and len(B) > len(A):
                            B = B2
                            A = A2
                            improved = True
                            break
            
            if improved:
                continue
            
            # Try swapping: remove x from A, add y to A; or cross-set swap
            # Try adding an unassigned element
            assigned = set(A) | set(B)
            dA = _diff_set(A)
            dB = _diff_set(B)
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x in assigned:
                    continue
                if _can_add(A, x, dB):
                    A.append(x)
                    improved = True
                    break
                if _can_add(B, x, dA):
                    B.append(x)
                    improved = True
                    break
        
        # Update best
        score = len(A) + len(B)
        if score > best_score:
            best_A = A[:]
            best_B = B[:]
            best_score = score
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
