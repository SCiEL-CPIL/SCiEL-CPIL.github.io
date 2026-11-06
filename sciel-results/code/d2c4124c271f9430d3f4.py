import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill
    n = 256

    def _valid(S):
        """Check the property for sorted set S. Returns True if valid."""
        lst = sorted(S)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                for k in range(j, m):
                    c = lst[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(a, S):
        """Check if element a can be added to set S without violating property.
        a must be smaller than all elements in S."""
        elements = sorted(S)
        m = len(elements)
        for j in range(m):
            b = elements[j]
            if b <= a:
                continue
            for k in range(j, m):
                c = elements[k]
                if (b + c) % a == 0:
                    return False
        return True

    # Start with proven valid base: [171, 256]
    # For a >= 171, 2a >= 342 > 256+256=512? No, but 3*171=513 > 2*256=512
    # So for any a,b,c in [171,256] with a<b<=c: b+c < 2*256=512 < 3*171=513
    # And b+c > 2*171=342 > 171=a, so b+c is not a multiple of a (only possible multiple would be 2a, but 2a < b+c < 3a)
    S = set(range(171, n + 1))

    # Descending greedy: try to add elements from 170 down to 1
    for a in range(170, 0, -1):
        if time.time() > deadline - 10:
            break
        if a not in S and can_add(a, S):
            S.add(a)

    best_S = set(S)
    best_size = len(S)

    # Local search: try removing one element and adding multiple smaller ones
    improved = True
    iters = 0
    max_iters = 300
    while improved and iters < max_iters and time.time() < deadline - 5:
        improved = False
        iters += 1
        S_list = sorted(S)
        # Try removing each element (prioritize larger ones as they may block more)
        for a in reversed(S_list):
            if time.time() > deadline - 5:
                break
            S_test = S - {a}
            # Try to add as many smaller elements as possible
            for b in range(1, a):
                if b not in S_test and can_add(b, S_test):
                    S_test.add(b)
            if len(S_test) > best_size:
                S = S_test
                best_S = set(S)
                best_size = len(S)
                improved = True
                break

    # Final greedy pass on remaining elements
    for a in range(1, n + 1):
        if time.time() > deadline - 2:
            break
        if a not in S and can_add(a, S):
            S.add(a)

    if len(S) > best_size:
        best_S = set(S)

    # Final verification with fallback
    if time.time() < deadline - 1:
        if not _valid(best_S):
            best_S = set(range(171, n + 1))

    return list(best_S)
