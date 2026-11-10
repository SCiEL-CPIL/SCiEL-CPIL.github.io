"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    def validate(elements):
        """Check if a set of elements satisfies the constraint."""
        if not elements:
            return True
        if len(set(elements)) != len(elements):
            return False
        for x in elements:
            if not isinstance(x, int) or x < 1 or x > 256:
                return False
        # Build subset sums indexed by cardinality
        # sums[k] = set of achievable sums with cardinality k
        sums = [set() for _ in range(len(elements) + 1)]
        sums[0].add(0)
        for val in elements:
            # Iterate backwards to avoid using val multiple times
            for k in range(len(sums) - 2, -1, -1):
                if sums[k]:
                    for s in sums[k]:
                        sums[k + 1].add(s + val)
        # Check: for any two different cardinalities, their sum sets must be disjoint
        for i in range(len(sums)):
            for j in range(i + 1, len(sums)):
                if sums[i] & sums[j]:
                    return False
        return True

    def greedy_construct(order, deadline):
        """Greedy: add each element if it doesn't violate the constraint."""
        elements = []
        # sums[k] = set of achievable sums with cardinality k
        sums = [set() for _ in range(len(order) + 1)]
        sums[0].add(0)
        
        for val in order:
            if time.time() > deadline:
                break
            # Check if adding val violates constraint
            # New sums would be: for each k, sums[k] + val -> added to sums[k+1]
            # We need to check if any new sum conflicts with existing sums of different cardinality
            valid = True
            new_sums = {}
            for k in range(len(sums) - 1, -1, -1):
                if sums[k]:
                    for s in sums[k]:
                        ns = s + val
                        nk = k + 1
                        if nk not in new_sums:
                            new_sums[nk] = set()
                        new_sums[nk].add(ns)
            
            # Check conflicts
            for nk, nset in new_sums.items():
                # Check against existing sums of different cardinality
                for k in range(len(sums)):
                    if k != nk and sums[k]:
                        if nset & sums[k]:
                            valid = False
                            break
                if not valid:
                    break
                # Check within new_sums of different cardinality
                for nk2, nset2 in new_sums.items():
                    if nk2 != nk and nset2:
                        if nset & nset2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            # Merge new sums
            for nk, nset in new_sums.items():
                sums[nk].update(nset)
            elements.append(val)
        
        return elements

    # Baseline: small consecutive range
    baseline = list(range(128, 136))
    best = baseline
    best_size = len(best)

    # Greedy from smallest to largest
    result = greedy_construct(range(1, 257), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(best)

    # Greedy from largest to smallest
    result2 = greedy_construct(range(256, 0, -1), deadline)
    if len(result2) > best_size:
        best = result2
        best_size = len(result2)

    # Greedy from middle outward
    mid = list(range(129, 257)) + list(range(1, 129))
    result3 = greedy_construct(mid, deadline)
    if len(result3) > best_size:
        best = result3
        best_size = len(result3)

    # Try random orders
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        result_r = greedy_construct(order, deadline)
        if len(result_r) > best_size:
            best = result_r
            best_size = len(result_r)

    # Final validation
    if not validate(best):
        # Fallback to known valid small set
        best = list(range(128, 136))

    # Ensure all elements are distinct ints in 1..256
    best = [int(x) for x in best]
    return best
# EVOLVE-BLOCK-END
