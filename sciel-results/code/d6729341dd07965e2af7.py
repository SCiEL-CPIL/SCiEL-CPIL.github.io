import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    n = 256

    def can_add(x, S, S_sorted=None):
        """Check if adding x to S preserves validity. S_sorted is optional pre-sorted list."""
        if x in S:
            return True
        if S_sorted is None:
            S_sorted = sorted(S)
        
        # Case 1: x is the smallest: x < b <= c, check x does not divide b+c
        for b in S_sorted:
            if b <= x:
                continue
            for c in S_sorted:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is the middle: a < x <= c, check a does not divide x+c
        for a in S_sorted:
            if a >= x:
                break
            for c in S_sorted:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is the largest: a < b <= x, check a does not divide b+x
        for a in S_sorted:
            if a >= x:
                break
            for b in S_sorted:
                if b > x:
                    break
                if a < b:
                    if (b + x) % a == 0:
                        return False
        
        return True

    def greedy_construct(order):
        """Greedy construction with given element ordering."""
        S = set()
        S_sorted = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S, S_sorted):
                S.add(x)
                # Maintain sorted list
                import bisect
                bisect.insort(S_sorted, x)
        return S, S_sorted

    # Multiple greedy strategies with different orderings
    candidates = []

    # Strategy 1: Descending (larger numbers first - fewer conflicts)
    S1, _ = greedy_construct(range(256, 0, -1))
    candidates.append(S1)

    # Strategy 2: Ascending (smaller numbers first - might allow more large numbers later)
    if time.time() < deadline - 5:
        S2, _ = greedy_construct(range(1, 257))
        candidates.append(S2)

    # Strategy 3: Random orderings
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            order = list(range(1, 257))
            rng.shuffle(order)
            Sr, _ = greedy_construct(order)
            candidates.append(Sr)

    # Strategy 4: Weighted - prioritize numbers with fewer divisors/multiples
    if time.time() < deadline - 10:
        # Numbers that are prime or near-prime have fewer divisibility issues
        def score(x):
            # Prefer larger numbers, and numbers with few divisors
            div_count = sum(1 for i in range(1, x+1) if x % i == 0)
            return (x, -div_count)
        order = sorted(range(1, 257), key=score, reverse=True)
        S4, _ = greedy_construct(order)
        candidates.append(S4)

    # Pick the best candidate
    best_S = max(candidates, key=len)
    best_size = len(best_S)

    # Local search phase: try to add any missing element
    if time.time() < deadline - 5:
        current = set(best_S)
        current_sorted = sorted(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in range(1, 257):
                if time.time() > deadline - 5:
                    break
                if x in current:
                    continue
                if can_add(x, current, current_sorted):
                    current.add(x)
                    current_sorted.append(x)
                    current_sorted.sort()
                    improved = True
                    if len(current) > best_size:
                        best_S = set(current)
                        best_size = len(current)

    # Local search phase: try 1-remove + 1-add swaps
    if time.time() < deadline - 5:
        current = set(best_S)
        current_sorted = sorted(current)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            # Try removing each element and see if we can add two new ones
            for rem in sorted(current):
                if time.time() > deadline - 5:
                    break
                trial = set(current)
                trial.discard(rem)
                trial_sorted = [x for x in current_sorted if x != rem]
                added_count = 0
                for add in range(256, 0, -1):
                    if add in trial:
                        continue
                    if can_add(add, trial, trial_sorted):
                        trial.add(add)
                        trial_sorted.append(add)
                        trial_sorted.sort()
                        added_count += 1
                        if added_count >= 1:
                            break
                if added_count > 0 and len(trial) > len(current):
                    current = trial
                    current_sorted = trial_sorted
                    if len(current) > best_size:
                        best_S = set(current)
                        best_size = len(current)
                    break  # Restart the outer loop with new current

    # Final: try one more round of additions
    if time.time() < deadline - 2:
        current = set(best_S)
        current_sorted = sorted(current)
        for x in range(1, 257):
            if time.time() > deadline - 2:
                break
            if x not in current and can_add(x, current, current_sorted):
                current.add(x)
                current_sorted.append(x)
                current_sorted.sort()
        if len(current) > len(best_S):
            best_S = current

    if not best_S:
        return [256]
    
    return list(best_S)
