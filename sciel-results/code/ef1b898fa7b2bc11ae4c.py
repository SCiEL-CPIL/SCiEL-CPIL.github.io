"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for a in S:
            for b in S:
                if a + b in s:
                    return False
        return True

    def greedy_add(S, rng, max_iter=2000000):
        """Try to add elements to S while maintaining sum-free property."""
        current = set(S)
        for _ in range(max_iter):
            if time.time() > deadline:
                return list(current)
            # Try to add each element not in current
            added = False
            for x in S:
                if x in current:
                    continue
                # Check if adding x would violate sum-free
                # We need: for all a, b in current, a + b != x
                # and for all a in current, a + x != any element in current
                # and x + x not in current
                ok = True
                if x + x in current:
                    ok = False
                if ok:
                    for a in current:
                        if a + x in current:
                            ok = False
                            break
                if ok:
                    for a in current:
                        for b in current:
                            if a + b == x:
                                ok = False
                                break
                        if not ok:
                            break
                if ok:
                    current.add(x)
                    added = True
                    break
            if not added:
                break
        return list(current)

    def local_search(S, rng, max_iters=50000):
        """Local search: try removing one element and adding another."""
        current = list(S)
        current_set = set(current)
        best = list(current)
        best_size = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing a random element
            if len(current) == 0:
                break
            idx = rng.randint(0, len(current) - 1)
            removed = current.pop(idx)
            current_set.discard(removed)
            
            # Try to add back something or another element
            # Try adding the removed element back if it helps (it won't, so skip)
            # Instead, try adding other elements
            for x in S:
                if x in current_set:
                    continue
                # Check if adding x to current_set would violate
                ok = True
                if x + x in current_set:
                    ok = False
                if ok:
                    for a in current_set:
                        if a + x in current_set:
                            ok = False
                            break
                if ok:
                    for a in current_set:
                        for b in current_set:
                            if a + b == x:
                                ok = False
                                break
                        if not ok:
                            break
                if ok and len(current) + 1 > best_size:
                    current.append(x)
                    current_set.add(x)
                    best = list(current)
                    best_size = len(best)
                    improved = True
                    break
            if not improved:
                # Restore removed
                current.append(removed)
                current_set.add(removed)
        
        return best

    def solve_one(S, seed):
        rng = Random(seed + 1000)
        # Start with odd numbers
        result = [x for x in S if x % 2]
        
        # Greedy addition
        result = greedy_add(result, rng)
        
        # Local search
        result = local_search(result, rng)
        
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(solve_one(S, seed))
    
    return results
# EVOLVE-BLOCK-END
