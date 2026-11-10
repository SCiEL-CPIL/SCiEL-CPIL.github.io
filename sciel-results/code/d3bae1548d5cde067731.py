import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts
    # conflicts[i] is a set of j such that i and j conflict
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # Initial set: all odds
    S = set(range(1, N + 1, 2))
    
    # List of evens not in S
    evens = [i for i in range(2, N + 1, 2) if i not in S]
    
    # Greedy addition of non-conflicting evens
    # We repeatedly scan and add any even that has no conflict with current S
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for e in list(evens):
            if time.time() > deadline:
                break
            if not conflicts[e].intersection(S):
                S.add(e)
                evens.remove(e)
                changed = True
    
    # Local Search: try swaps to improve cardinality
    # 1-for-1 swaps: remove one element, try to add one that was previously blocked
    # Also try multi-add after removals
    
    def try_add_greedy(S, evens, conflicts, deadline):
        """Greedily add any evens that don't conflict with S."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for e in list(evens):
                if time.time() > deadline:
                    break
                if not conflicts[e].intersection(S):
                    S.add(e)
                    evens.remove(e)
                    changed = True
    
    # 1-for-1 swap local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        for s_elem in S_list:
            if time.time() > deadline:
                break
            # Try removing s_elem and see if we can add more
            S_test = S - {s_elem}
            # Find evens that were blocked only by s_elem
            candidates = []
            for e in evens:
                # e conflicts with S_test?
                if not conflicts[e].intersection(S_test):
                    # e would be addable
                    candidates.append(e)
            # Try adding all candidates greedily
            if candidates:
                S_test2 = S_test.copy()
                evens_test = list(evens)
                # Add candidates in order of fewest conflicts (heuristic)
                candidates.sort(key=lambda e: len(conflicts[e] & S_test))
                added = 0
                for e in candidates:
                    if time.time() > deadline:
                        break
                    if not conflicts[e].intersection(S_test2):
                        S_test2.add(e)
                        if e in evens_test:
                            evens_test.remove(e)
                        added += 1
                # Net gain = added - 1 (we removed one)
                if added > 1:
                    S = S_test2
                    evens = evens_test
                    improved = True
                    break
                elif added == 1:
                    # No net gain, but might enable further improvements
                    # Skip for now, just keep original
                    pass
    
    # After swaps, do another greedy pass
    if time.time() < deadline:
        try_add_greedy(S, evens, conflicts, deadline)
    
    # Try 1-for-2 or 1-for-3 swaps more aggressively
    # Remove one element, add multiple
    for _ in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        S_list = list(S)
        found_swap = False
        for s_elem in S_list:
            if time.time() > deadline:
                break
            S_test = S - {s_elem}
            # Find all evens that become addable
            candidates = []
            for e in evens:
                if not conflicts[e].intersection(S_test):
                    candidates.append(e)
            if len(candidates) >= 2:
                # Try to add as many as possible
                S_test2 = S_test.copy()
                evens_test = list(evens)
                # Sort by conflict count with S_test
                candidates.sort(key=lambda e: len(conflicts[e] & S_test))
                added = 0
                for e in candidates:
                    if time.time() > deadline:
                        break
                    if not conflicts[e].intersection(S_test2):
                        S_test2.add(e)
                        if e in evens_test:
                            evens_test.remove(e)
                        added += 1
                if added > 1:
                    S = S_test2
                    evens = evens_test
                    found_swap = True
                    break
        if not found_swap:
            break
        # After a swap, do greedy
        if time.time() < deadline:
            try_add_greedy(S, evens, conflicts, deadline)
    
    return list(S)
# EVOLVE-BLOCK-END
