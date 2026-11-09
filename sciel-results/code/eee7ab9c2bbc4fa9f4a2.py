import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 12
    
    # Greedy minimal extension: build the set one element at a time,
    # always choosing the smallest possible positive integer that keeps
    # all subset sums distinct.
    #
    # We maintain the set of all achievable subset sums.
    # Initially, sums = {0} (empty subset).
    # At each step, we try candidate values 1, 2, 3, ... and pick the first
    # one that, when added to the current set, produces a new set of subset
    # sums with no collisions (i.e., |new_sums| == 2 * |old_sums|).
    
    sums = {0}
    chosen = []
    
    for _ in range(n):
        found = False
        # Upper bound: in the worst case we might need to go up to 2^12,
        # but in practice for distinct subset sums the greedy stays much lower.
        # We'll iterate upward until we find a valid value.
        candidate = 1
        while not found:
            # Check if adding 'candidate' keeps all subset sums distinct
            # New sums = old_sums ∪ {s + candidate for s in old_sums}
            # They are distinct iff (old_sums + candidate) is disjoint from old_sums
            new_sums = set()
            valid = True
            for s in sums:
                ns = s + candidate
                if ns in new_sums or ns in sums:
                    # Collision: either two different subsets produce the same sum,
                    # or the new sum collides with an existing one
                    # Actually we need to check: the set {s + candidate : s in sums} 
                    # must be disjoint from sums, AND all s+candidate must be distinct
                    # (which they are since sums elements are distinct and we add same candidate)
                    # So we just need: (sums + candidate) ∩ sums = ∅
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                # All good, accept this candidate
                sums.update(new_sums)
                chosen.append(candidate)
                found = True
            else:
                candidate += 1
                # Time check
                if time.time() > deadline:
                    # Fallback: use powers of 2 for remaining
                    for j in range(len(chosen), n):
                        chosen.append(1 << j)
                    return chosen
    
    return chosen

# Test / verify
if __name__ == "__main__":
    result = construct()
    print(f"Result: {result}")
    print(f"Max: {max(result)}")
    
    # Verify distinct subset sums
    n = len(result)
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        assert s not in sums, f"Collision at mask {mask}"
        sums.add(s)
    print(f"All {len(sums)} subset sums are distinct.")
