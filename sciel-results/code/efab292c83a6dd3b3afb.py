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
        a must be smaller than all elements in S (or we handle general case)."""
        # For all b, c in S with a < b <= c: a must not divide (b+c)
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

    def greedy_descending(S_base, upper_limit):
        """Starting from S_base, greedily add elements from upper_limit down to 1."""
        S = set(S_base)
        for a in range(upper_limit, 0, -1):
            if time.time() > deadline - 2:
                break
            if a not in S and can_add(a, S):
                S.add(a)
        return S

    def local_search(S, max_iters=200):
        """Try to improve S by removing one element and adding multiple smaller ones."""
        S = set(S)
        best_size = len(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 3:
            improved = False
            iters += 1
            S_list = sorted(S)
            # Try removing each element (prioritize larger ones as they block more)
            for a in reversed(S_list):
                if time.time() > deadline - 3:
                    break
                S_test = S - {a}
                # Try to add as many smaller elements as possible
                added = 0
                for b in range(1, a + 1):
                    if b not in S_test and can_add(b, S_test):
                        S_test.add(b)
                        added += 1
                if len(S_test) > best_size:
                    S = S_test
                    best_size = len(S)
                    improved = True
                    break
        return S

    # Strategy 1: Current approach - base [171,256], descending greedy
    S1 = set(range(171, n + 1))
    S1 = greedy_descending(S1, 170)

    # Strategy 2: Wider base [160,256], descending greedy
    if time.time() < deadline - 5:
        S2 = set(range(160, n + 1))
        S2 = greedy_descending(S2, 159)
    else:
        S2 = S1

    # Strategy 3: Even wider base [144,256]
    if time.time() < deadline - 5:
        S3 = set(range(144, n + 1))
        S3 = greedy_descending(S3, 143)
    else:
        S3 = S1

    # Strategy 4: Greedy from below (add smallest first)
    if time.time() < deadline - 5:
        S4 = set()
        for a in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if can_add(a, S4):
                S4.add(a)
    else:
        S4 = S1

    # Strategy 5: Greedy from above (add largest first)
    if time.time() < deadline - 5:
        S5 = set()
        for a in range(n, 0, -1):
            if time.time() > deadline - 3:
                break
            if can_add(a, S5):
                S5.add(a)
    else:
        S5 = S1

    # Pick the best candidate so far
    candidates = [S1, S2, S3, S4, S5]
    best_S = max(candidates, key=len)
    best_size = len(best_S)

    # Apply local search to the best candidate
    if time.time() < deadline - 5:
        best_S = local_search(best_S, max_iters=300)
        best_size = len(best_S)

    # Also try local search on other good candidates if time permits
    for S_cand in candidates:
        if time.time() > deadline - 4:
            break
        if len(S_cand) >= best_size - 5:
            S_improved = local_search(set(S_cand), max_iters=100)
            if len(S_improved) > best_size:
                best_S = S_improved
                best_size = len(best_S)

    # Final greedy pass to add any remaining elements
    if time.time() < deadline - 2:
        S_final = set(best_S)
        for a in range(1, n + 1):
            if time.time() > deadline - 2:
                break
            if a not in S_final and can_add(a, S_final):
                S_final.add(a)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)

    # Final verification
    if not _valid(best_S):
        # Fallback to safe base
        best_S = set(range(171, n + 1))

    return sorted(best_S)
