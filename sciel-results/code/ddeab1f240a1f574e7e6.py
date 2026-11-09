import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s

    # Precompute: for each pair (a, b) with a < b, which c >= b cause a | (b+c)?
    # Actually, let's precompute conflicts differently.
    # For the constraint: for all a < b <= c in S, a must NOT divide b+c.
    # We can precompute for each (a, b) with a < b, the set of c >= b such that a | (b+c).
    # But that's expensive. Instead, let's precompute for each a, the residues.
    
    # Better approach: precompute conflict[a][b] = set of c >= b where a | (b+c)
    # This is too large. Let's use a different strategy.
    
    # For can_add(S, x): we need to check if adding x creates any violation.
    # The violations involve x in one of three roles. Let me optimize each case.
    
    def valid(S):
        """Check if set S is valid: for all a<b<=c in S, a does not divide b+c."""
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(i+1, len(L)):
                b = L[j]
                for k in range(j, len(L)):
                    c = L[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add(S, x):
        """Check if adding x to S keeps it valid. S is a set."""
        L = sorted(S)
        # Case 1: x is smallest: x < b <= c, x | (b+c)
        # For each pair (b,c) in S with b >= x, c >= b, check if x | (b+c)
        for i in range(len(L)):
            b = L[i]
            if b < x:
                continue
            for j in range(i, len(L)):
                c = L[j]
                if (b + c) % x == 0:
                    return False
        # Case 2: x is middle: a < x <= c, a | (x+c)
        for a in L:
            if a >= x:
                break
            for c in L:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        # Case 3: x is largest: a < b <= x, a | (b+x)
        for i in range(len(L)):
            a = L[i]
            if a >= x:
                break
            for j in range(i+1, len(L)):
                b = L[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        return True

    def conflict_degree(S, x):
        """Count how many conflicts x would create with S (for heuristic)."""
        L = sorted(S)
        count = 0
        # Case 1: x smallest
        for i in range(len(L)):
            b = L[i]
            if b < x:
                continue
            for j in range(i, len(L)):
                c = L[j]
                if (b + c) % x == 0:
                    count += 1
        # Case 2: x middle
        for a in L:
            if a >= x:
                break
            for c in L:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    count += 1
        # Case 3: x largest
        for i in range(len(L)):
            a = L[i]
            if a >= x:
                break
            for j in range(i+1, len(L)):
                b = L[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    count += 1
        return count

    def greedy_from_tail(tail_start, order, use_degree=False):
        """Greedy: start with [tail_start, 256], add elements in given order."""
        S = set(range(tail_start, 257))
        for candidate in order:
            if candidate in S:
                continue
            if can_add(S, candidate):
                S.add(candidate)
        return S

    def greedy_weighted(S, candidates, max_time):
        """Greedy that picks the candidate with lowest conflict degree first."""
        S = set(S)
        remaining = [c for c in candidates if c not in S]
        end = start_time + max_time
        while remaining and time.time() < end:
            # Find candidate with lowest conflict degree that can be added
            best_cand = None
            best_deg = float('inf')
            for c in remaining:
                if can_add(S, c):
                    deg = conflict_degree(S, c)
                    if deg < best_deg:
                        best_deg = deg
                        best_cand = c
            if best_cand is None:
                break
            S.add(best_cand)
            remaining.remove(best_cand)
        return S

    def local_search(S, max_time):
        """Try to improve S by removing one element and adding one or more."""
        S = set(S)
        best = S
        end = start_time + max_time
        while time.time() < end:
            improved = False
            L = sorted(S)
            for i in range(len(L)):
                if time.time() > end:
                    break
                a = L[i]
                S2 = S - {a}
                # Try adding elements not in S2, prioritizing those with low conflict
                added_any = False
                for candidate in range(1, 257):
                    if time.time() > end:
                        break
                    if candidate in S2:
                        continue
                    if can_add(S2, candidate):
                        S2.add(candidate)
                        added_any = True
                if added_any:
                    # Try to add more
                    for candidate2 in range(1, 257):
                        if time.time() > end:
                            break
                        if candidate2 in S2:
                            continue
                        if can_add(S2, candidate2):
                            S2.add(candidate2)
                    if len(S2) > len(best):
                        best = S2
                        S = set(S2)
                        improved = True
                        break
            if not improved:
                break
        return best

    def local_search_2swap(S, max_time):
        """Try to improve S by removing two elements and adding two."""
        S = set(S)
        best = S
        end = start_time + max_time
        while time.time() < end:
            improved = False
            L = sorted(S)
            for i in range(len(L)):
                if time.time() > end:
                    break
                for j in range(i+1, len(L)):
                    if time.time() > end:
                        break
                    a1, a2 = L[i], L[j]
                    S2 = S - {a1, a2}
                    # Try adding elements
                    for candidate in range(1, 257):
                        if time.time() > end:
                            break
                        if candidate in S2:
                            continue
                        if can_add(S2, candidate):
                            S2.add(candidate)
                            # Try to add one more
                            for candidate2 in range(candidate+1, 257):
                                if time.time() > end:
                                    break
                                if candidate2 in S2:
                                    continue
                                if can_add(S2, candidate2):
                                    S3 = S2 | {candidate2}
                                    if len(S3) > len(best):
                                        best = S3
                                        S = set(S3)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        return best

    # Try multiple tail starts and orders
    best = set(range(171, 257))
    best_size = len(best)

    tail_starts = [171, 175, 180, 185, 190, 195, 200, 210, 220, 165, 160, 155]
    
    # Basic orders
    orders = [
        list(range(1, 171)),
        list(range(170, 0, -1)),
    ]
    
    # Add some randomized orders
    rng = random.Random(42)
    for _ in range(8):
        o = list(range(1, 171))
        rng.shuffle(o)
        orders.append(o)
    
    # Add orders that prioritize numbers with fewer divisors/multiples
    for ts in tail_starts:
        if time.time() > deadline - 35:
            break
        for order in orders:
            if time.time() > deadline - 25:
                break
            S = greedy_from_tail(ts, order)
            if len(S) > best_size:
                best = S
                best_size = len(S)
    
    # Try weighted greedy from the best tail
    if time.time() < deadline - 20:
        tail_start = min(best)
        candidates = list(range(1, tail_start))
        S = greedy_weighted(best, candidates, max_time=deadline - 22)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search on the best
    if time.time() < deadline - 15:
        best = local_search(best, max_time=deadline - 17)
        best_size = len(best)
    
    # 2-swap local search
    if time.time() < deadline - 8:
        best = local_search_2swap(best, max_time=deadline - 10)
        best_size = len(best)
    
    # Final verification
    if not valid(best):
        best = set(range(171, 257))

    return sorted(best)
