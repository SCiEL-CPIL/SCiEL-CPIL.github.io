"""Improved program: greedy + local search to maximize set size without a forbidden triple."""
import time
import random
from itertools import combinations

N = 128

def _violations_count(S):
    """Count number of forbidden triples (a,b,c) in set S."""
    S_set = set(S)
    cnt = 0
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i+1, n):
            b = S[j]
            for k in range(j+1, n):
                c = S[k]
                if (a+b) in S_set and (a+c) in S_set and (b+c) in S_set:
                    cnt += 1
    return cnt

def _violations_count_fast(S_list):
    """Faster violation count using set lookup."""
    S_set = set(S_list)
    cnt = 0
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i+1, n):
            b = S_list[j]
            if (a+b) not in S_set:
                continue
            for k in range(j+1, n):
                c = S_list[k]
                if (a+c) in S_set and (b+c) in S_set:
                    cnt += 1
    return cnt

def _greedy_construct():
    """Greedy: add elements in random order if they don't create a violation."""
    S = set()
    elems = list(range(1, N+1))
    random.shuffle(elems)
    for x in elems:
        # Check if adding x creates any forbidden triple
        ok = True
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if (a+b) in S and (a+x) in S and (b+x) in S:
                    ok = False
                    break
                if (a+x) in S and (a+b) in S and (b+x) in S:
                    # a+b, a+x, b+x all in S (with x being new, a+x and b+x are x-related)
                    # Wait, x is not yet in S, so a+x and b+x are not in S unless they equal existing elements
                    pass
            if not ok:
                break
        if ok:
            S.add(x)
    return list(S)

def _greedy_construct_v2():
    """Greedy with proper check: adding x is valid iff no triple (a,b,c) in S∪{x} is forbidden."""
    S = set()
    elems = list(range(1, N+1))
    random.shuffle(elems)
    for x in elems:
        # A triple (a,b,c) with a,b,c distinct from S∪{x} is forbidden iff
        # all of a,b,c,a+b,a+c,b+c are in S∪{x}.
        # Since x is new, the only way x participates is if x is one of {a,b,c} or one of {a+b,a+c,b+c}.
        # Case 1: x is one of a,b,c. WLOG x=a. Then we need b,c in S, and b+c in S∪{x}, x+b in S∪{x}, x+c in S∪{x}.
        #   Since x is not in S, x+b and x+c must be in S (not equal to x since b,c>0). And b+c must be in S or b+c==x.
        #   Actually b+c==x is possible. And x+b = x+c only if b=c which is impossible.
        #   So: exists b,c in S, b≠c, such that (x+b in S) and (x+c in S) and (b+c in S or b+c==x).
        # Case 2: x is one of the sums, e.g., x = a+b. Then a,c in S, b in S, a+c in S, b+c in S.
        #   So: exists a,b,c in S, a≠b≠c, such that a+b==x, a+c in S, b+c in S.
        ok = True
        S_list = list(S)
        n = len(S_list)
        # Case 1: x is a member
        for i in range(n):
            b = S_list[i]
            if (x + b) not in S:
                continue
            for j in range(i+1, n):
                c = S_list[j]
                if (x + c) not in S:
                    continue
                if (b + c) in S or (b + c) == x:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            # Case 2: x is a sum
            for i in range(n):
                a = S_list[i]
                if a >= x:
                    continue
                b = x - a
                if b not in S or b == a:
                    continue
                for j in range(n):
                    c = S_list[j]
                    if c == a or c == b:
                        continue
                    if (a + c) in S and (b + c) in S:
                        ok = False
                        break
                if not ok:
                    break
        if ok:
            S.add(x)
    return list(S)

def _local_search(S_list, max_iters=200000, deadline=None):
    """Local search: try removing one element and adding another to improve size."""
    S = set(S_list)
    best = len(S)
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Try to add an element
        added = False
        for x in range(1, N+1):
            if x in S:
                continue
            # Check if adding x is valid
            S_list = list(S)
            n = len(S_list)
            ok = True
            # Case 1: x is a member
            for i in range(n):
                b = S_list[i]
                if (x + b) not in S:
                    continue
                for j in range(i+1, n):
                    c = S_list[j]
                    if (x + c) not in S:
                        continue
                    if (b + c) in S or (b + c) == x:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                # Case 2: x is a sum
                for i in range(n):
                    a = S_list[i]
                    if a >= x:
                        continue
                    b = x - a
                    if b not in S or b == a:
                        continue
                    for j in range(n):
                        c = S_list[j]
                        if c == a or c == b:
                            continue
                        if (a + c) in S and (b + c) in S:
                            ok = False
                            break
                    if not ok:
                        break
            if ok:
                S.add(x)
                added = True
                break
        if added:
            continue
        # Try remove one, add another (swap) to maintain or increase size
        # Only useful if we can increase size, but size can't increase if no add works.
        # So local search is just greedy add. For improvement, try different orderings.
        break
    return list(S)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    best_size = 0
    best_set = set()
    
    # Multiple random greedy tries
    for trial in range(500):
        if time.time() > deadline:
            break
        S_list = _greedy_construct_v2()
        S = set(S_list)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Local search on best
    if best_size > 0 and time.time() < deadline:
        S_list = _local_search(list(best_set), deadline=deadline)
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_set = set(S_list)
    
    # Final: try more random restarts if time permits
    for trial in range(2000):
        if time.time() > deadline:
            break
        S_list = _greedy_construct_v2()
        S = set(S_list)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    return list(best_set)
