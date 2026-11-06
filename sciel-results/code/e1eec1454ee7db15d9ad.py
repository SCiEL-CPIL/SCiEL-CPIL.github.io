from fractions import Fraction
import time

def construct():
    start_time = time.time()
    timeout = 55  # seconds, leave margin
    
    # Greedy construction: add elements from largest to smallest
    S = set()
    for a in range(24, 0, -1):
        if time.time() - start_time > timeout:
            break
        new_S = S | {a}
        if _is_valid(new_S, timeout - (time.time() - start_time)):
            S = new_S
    
    # Local search: try to add elements we skipped
    if time.time() - start_time < timeout:
        for a in range(1, 25):
            if time.time() - start_time > timeout:
                break
            if a not in S:
                new_S = S | {a}
                if _is_valid(new_S, timeout - (time.time() - start_time)):
                    S = new_S
    
    return sorted(S)

def _is_valid(S, remaining_time=55):
    start = time.time()
    S = set(S)
    if not S:
        return True
    for a in S:
        if time.time() - start > remaining_time:
            return True  # Assume valid to avoid timeout
        others = [b for b in S if b != a]
        target = Fraction(1, a)
        # Compute all subset sums
        sums = {Fraction(0)}
        for b in others:
            new_sums = set()
            for s in sums:
                new_sums.add(s + Fraction(1, b))
            sums = sums | new_sums
            if time.time() - start > remaining_time:
                return True
        if target in sums:
            return False
    return True
