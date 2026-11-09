import time

def _check_validity(A):
    """Check if set A satisfies cardinality-separated subset sums."""
    n = len(A)
    if n == 0:
        return True
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0] = {0}
    for a in A:
        for k in range(n - 1, -1, -1):
            for s in list(sums_by_k[k]):
                sums_by_k[k + 1].add(s + a)
    seen = {}
    for k in range(n + 1):
        for s in sums_by_k[k]:
            if s in seen:
                return False
            seen[s] = k
    return True

def _sufficient_condition(A):
    """Check sufficient condition: min sum of (k+1) smallest > max sum of k largest for all k."""
    n = len(A)
    if n == 0:
        return True
    A_sorted = sorted(A)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + A_sorted[i]
    total = prefix[n]
    for k in range(n):
        min_sum_k1 = prefix[k + 1]
        max_sum_k = total - prefix[n - k]
        if min_sum_k1 <= max_sum_k:
            return False
    return True

def _check_validity_fast(A):
    """Faster check: use sufficient condition first, then exact DP."""
    n = len(A)
    if n == 0:
        return True
    if _sufficient_condition(A):
        return True
    return _check_validity(A)

def construct():
    start_time = time.time()
    deadline = 55.0
    
    # Strategy: greedy with candidate ordering that maximizes separation
    # Try multiple greedy orders and pick the best
    
    best_A = []
    
    # Order 1: ascending
    A1 = []
    for candidate in range(1, 257):
        if time.time() - start_time > deadline:
            break
        new_A = A1 + [candidate]
        if _check_validity_fast(new_A):
            A1 = new_A
    if len(A1) > len(best_A):
        best_A = A1[:]
    
    # Order 2: descending (larger elements first for better separation)
    if time.time() - start_time < deadline - 5:
        A2 = []
        for candidate in range(256, 0, -1):
            if time.time() - start_time > deadline:
                break
            new_A = A2 + [candidate]
            if _check_validity_fast(new_A):
                A2 = new_A
        if len(A2) > len(best_A):
            best_A = A2[:]
    
    # Order 3: powers of 2 first, then fill in
    if time.time() - start_time < deadline - 5:
        A3 = []
        used = set()
        # Start with powers of 2
        for p in [1, 2, 4, 8, 16, 32, 64, 128]:
            if p <= 256:
                A3.append(p)
                used.add(p)
        # Then try to add others in ascending order
        for candidate in range(1, 257):
            if time.time() - start_time > deadline:
                break
            if candidate in used:
                continue
            new_A = A3 + [candidate]
            if _check_validity_fast(new_A):
                A3 = new_A
                used.add(candidate)
        if len(A3) > len(best_A):
            best_A = A3[:]
    
    # Local search: try to swap elements to fit more
    if time.time() - start_time < deadline - 3:
        A = best_A[:]
        improved = True
        while improved and time.time() - start_time < deadline - 2:
            improved = False
            n = len(A)
            for i in range(n):
                if time.time() - start_time > deadline - 1:
                    break
                for candidate in range(1, 257):
                    if candidate in A:
                        continue
                    # Try replacing A[i] with candidate
                    new_A = A[:i] + [candidate] + A[i+1:]
                    if _check_validity_fast(new_A):
                        # Check if this helps us add more later
                        # For now, just accept if valid (may help future additions)
                        # Actually, this doesn't increase size. Skip pure swaps.
                        pass
            # Try to add one more element
            for candidate in range(1, 257):
                if time.time() - start_time > deadline - 1:
                    break
                if candidate in A:
                    continue
                new_A = A + [candidate]
                if _check_validity_fast(new_A):
                    A = new_A
                    improved = True
                    break
        if len(A) > len(best_A):
            best_A = A[:]
    
    # Try a completely different approach: use the sufficient condition to guide construction
    if time.time() - start_time < deadline - 5:
        # Build set where each new element is large enough to maintain separation
        A4 = []
        for candidate in range(1, 257):
            if time.time() - start_time > deadline:
                break
            new_A = A4 + [candidate]
            if _sufficient_condition(new_A):
                A4 = new_A
        if len(A4) > len(best_A):
            best_A = A4[:]
    
    return [int(x) for x in best_A]
