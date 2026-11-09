import time
from math import gcd, lcm

def construct():
    start_time = time.time()
    deadline = 55.0

    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return set(divs)

    # Start with divisors of 480 (2^5·3·5) = 24 elements
    # All pairwise lcms divide 480 ≤ 512
    S = get_divisors(480)
    best_S = set(S)
    best_size = len(S)

    # Try to add more elements via greedy addition
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 513):
            if x in best_S:
                continue
            valid = True
            for y in best_S:
                if lcm(x, y) > 512:
                    valid = False
                    break
            if valid:
                best_S.add(x)
                best_size += 1
                added = True

    # Now try swap operations: remove one element, add two
    # This can increase size by 1
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_copy = set(best_S)
        for x in list(S_copy):
            if time.time() > deadline:
                break
            # Try removing x
            S_trial = S_copy - {x}
            # Find all elements that can be added to S_trial
            candidates = []
            for y in range(1, 513):
                if y in S_trial:
                    continue
                valid = True
                for z in S_trial:
                    if lcm(y, z) > 512:
                        valid = False
                        break
                if valid:
                    candidates.append(y)
            
            # Try adding pairs from candidates
            if len(candidates) >= 2:
                # Check if we can add at least 2 candidates that are mutually compatible
                found_pair = False
                for i in range(len(candidates)):
                    if found_pair:
                        break
                    for j in range(i + 1, len(candidates)):
                        if lcm(candidates[i], candidates[j]) <= 512:
                            # We can add both
                            new_S = S_trial | {candidates[i], candidates[j]}
                            if len(new_S) > len(best_S):
                                best_S = new_S
                                best_size = len(best_S)
                                improved = True
                                found_pair = True
                                break

    # Final greedy pass
    added = True
    while added and time.time() < deadline:
        added = False
        for x in range(1, 513):
            if x in best_S:
                continue
            valid = True
            for y in best_S:
                if lcm(x, y) > 512:
                    valid = False
                    break
            if valid:
                best_S.add(x)
                best_size += 1
                added = True

    return list(best_S)
