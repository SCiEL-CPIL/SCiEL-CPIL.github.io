"""Improved construction for the 'no a divides b+c' problem on [1,256]."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations_with_replacement

def _is_valid(S):
    """Check if a sorted list S is a valid set."""
    S = sorted(S)
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                s = b + c
                if s % a == 0:
                    return False
    return True

def _count_violations(S):
    """Count the number of violations in a sorted list S."""
    S = sorted(S)
    n = len(S)
    count = 0
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            for k in range(j, n):
                c = S[k]
                s = b + c
                if s % a == 0:
                    count += 1
    return count

def _local_search(S, deadline):
    """Try to improve a valid set by swapping or adding/removing elements."""
    S = sorted(S)
    best = S[:]
    best_score = len(S)
    
    # Try to add elements
    candidates = list(range(1, 257))
    for x in candidates:
        if x in best:
            continue
        new_S = sorted(best + [x])
        if _is_valid(new_S):
            best = new_S
            best_score = len(best)
    
    # Try to swap elements
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            for x in range(1, 257):
                if x in best:
                    continue
                new_S = best[:]
                new_S[i] = x
                new_S = sorted(new_S)
                if _is_valid(new_S) and len(new_S) > best_score:
                    best = new_S
                    best_score = len(new_S)
                    improved = True
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    # Start with the upper tail [171, 256]
    S = list(range(171, 257))
    best = S[:]
    best_size = len(S)
    
    # Try multi-start with different upper tails
    for start in range(128, 172):
        if time.time() > deadline:
            break
        S = list(range(start, 257))
        if len(S) < best_size:
            continue
        if not _is_valid(S):
            continue
        best = S[:]
        best_size = len(S)
    
    # Try to improve the best set with local search
    if time.time() < deadline:
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Try adding smaller elements to the best set
    if time.time() < deadline:
        candidates = list(range(1, min(best[0], 257)))
        candidates.sort(key=lambda x: -x)  # Try larger candidates first
        for x in candidates:
            if time.time() > deadline:
                break
            if x in best:
                continue
            new_S = sorted(best + [x])
            if _is_valid(new_S):
                best = new_S
                best_size = len(best)
    
    # Final local search pass
    if time.time() < deadline:
        improved = _local_search(best, deadline)
        if len(improved) > best_size:
            best = improved
    
    return best
# EVOLVE-BLOCK-END
