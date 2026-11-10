import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    n = 128
    full_range = list(range(1, n + 1))
    
    # Known good base: 80 elements
    base = list(range(65, 129)) + list(range(17, 33))
    best = list(base)
    best_size = len(best)
    
    def _is_valid(S):
        S_set = set(S)
        for a, b, c in combinations(S, 3):
            if (a + b) in S_set and (a + c) in S_set and (b + c) in S_set:
                return False
        return True
    
    def _can_add_fast(S_set, x):
        """O(|S|) check: adding x creates forbidden triple iff there exist a,b in S
        such that a+b, a+x, b+x are all in S_set or equal to x.
        Since x is new, a+x==x means a==0 (impossible), so a+x and b+x must be in S_set.
        Also a+b must be in S_set (or a+b==x, but x not in S_set yet).
        So: exists a,b in S with a+b in S_set, a+x in S_set, b+x in S_set.
        """
        # For each a in S, check if a+x in S_set
        # If so, for each b in S (b != a), check if b+x in S_set AND a+b in S_set
        # Optimize: precompute which a have a+x in S_set
        candidates_a = [a for a in S_set if (a + x) in S_set]
        if not candidates_a:
            return True
        for a in candidates_a:
            for b in S_set:
                if b == a:
                    continue
                if (b + x) in S_set and (a + b) in S_set:
                    return False
        return True
    
    def _greedy_construct(order, S_set, deadline):
        """Given initial set S_set and an order of remaining elements, greedily add."""
        S = set(S_set)
        for x in order:
            if time.time() > deadline:
                break
            if x in S:
                continue
            if _can_add_fast(S, x):
                S.add(x)
        return S
    
    def _local_search(S, deadline, max_iters=500):
        """Randomized local search: try to add random elements, or swap (remove one, add one+)."""
        S = set(S)
        S_list = list(S)
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            # Try adding a random element from 1..128 not in S
            remaining = [x for x in range(1, n + 1) if x not in S]
            if not remaining:
                break
            # Try several random candidates
            tried = 0
            for _t in range(10):
                if time.time() > deadline:
                    break
                x = random.choice(remaining)
                if _can_add_fast(S, x):
                    S.add(x)
                    remaining.remove(x)
                    tried += 1
                    break
            if tried == 0:
                # Try a swap: remove a random element, see if we can add more
                if len(S) > 5:
                    remove_x = random.choice(S_list)
                    S.remove(remove_x)
                    S_list.remove(remove_x)
                    # Try to add elements
                    added = 0
                    new_remaining = [x for x in range(1, n + 1) if x not in S]
                    for _t in range(20):
                        if time.time() > deadline:
                            break
                        if not new_remaining:
                            break
                        x = random.choice(new_remaining)
                        if _can_add_fast(S, x):
                            S.add(x)
                            new_remaining.remove(x)
                            S_list.append(x)
                            added += 1
                    if added <= 1:
                        # Net loss or no gain, revert
                        S.add(remove_x)
                        S_list.append(remove_x)
                        for x in list(S_list):
                            if x != remove_x and x not in S:
                                S_list.remove(x)
        return S
    
    # Multi-start greedy with different orderings
    rng = random.Random(42)
    num_starts = 0
    while time.time() < deadline - 15:
        num_starts += 1
        # Different ordering strategies
        strat = num_starts % 5
        if strat == 0:
            # Random shuffle
            order = full_range[:]
            rng.shuffle(order)
        elif strat == 1:
            # Prefer high numbers (sums likely > 128)
            order = list(range(n, 0, -1))
            rng.shuffle(order[:40])  # shuffle low end
        elif strat == 2:
            # Prefer numbers whose sums with common elements exceed 128
            # i.e., prefer large numbers
            order = sorted(full_range, key=lambda x: -x)
            rng.shuffle(order[30:])
        elif strat == 3:
            # Start from base set, add remaining in random order
            remaining = [x for x in full_range if x not in set(base)]
            rng.shuffle(remaining)
            S = _greedy_construct(remaining, set(base), deadline)
            if len(S) > best_size:
                best = list(S)
                best_size = len(S)
            continue
        else:
            # Prefer middle-range numbers (33-64) mixed with high
            mid = list(range(33, 65))
            high = list(range(65, 129))
            low = list(range(1, 33))
            rng.shuffle(mid)
            rng.shuffle(low)
            order = mid + high[:30] + low + high[30:]
            rng.shuffle(order[60:])
        
        S = _greedy_construct(order, set(), deadline)
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Local search on the best found so far
    if time.time() < deadline - 5:
        S = _local_search(set(best), deadline, max_iters=200)
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Also try local search from base
    if time.time() < deadline - 3:
        S = _local_search(set(base), deadline, max_iters=300)
        if len(S) > best_size:
            best = list(S)
            best_size = len(S)
    
    # Final verification
    if not _is_valid(best):
        return list(range(65, 129)) + list(range(17, 33))
    
    return [int(x) for x in best]
