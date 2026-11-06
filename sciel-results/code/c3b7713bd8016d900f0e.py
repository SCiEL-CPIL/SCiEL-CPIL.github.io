import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s kill

    universe = list(range(1, 23))
    
    # Precompute scaled reciprocals using a common denominator
    BASE = 232792560
    scaled = {i: BASE // i for i in range(1, 23)}

    best_set = []
    best_size = 0

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_set = list(S)
            best_size = len(S)

    def is_dissociated(S):
        """
        Fast check: all subset sums of reciprocals of S are distinct.
        Uses a set with early exit on collision.
        """
        if not S:
            return True
        n = len(S)
        seen = {0}
        sums = [0]
        for x in S:
            val = scaled[x]
            current_len = len(sums)
            for i in range(current_len):
                s = sums[i] + val
                if s in seen:
                    return False
                seen.add(s)
                sums.append(s)
        return True

    def greedy_build(order):
        """Build a dissociated set by greedily adding elements in given order."""
        S = []
        for x in order:
            new_S = S + [x]
            if is_dissociated(new_S):
                S = new_S
        return S

    def local_improve(S):
        """Try to add elements, then do swaps to escape local optima."""
        S = list(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding elements
            rem = [x for x in universe if x not in S]
            for x in rem:
                if time.time() > deadline:
                    break
                new_S = S + [x]
                if is_dissociated(new_S):
                    S = new_S
                    update_best(S)
                    improved = True
                    break
            if improved:
                continue
            # Try single swaps
            rem = [x for x in universe if x not in S]
            for a in S:
                if time.time() > deadline:
                    break
                for b in rem:
                    if time.time() > deadline:
                        break
                    new_S = [x for x in S if x != a] + [b]
                    if is_dissociated(new_S):
                        S = new_S
                        update_best(S)
                        improved = True
                        break
                if improved:
                    break
        return S

    # Strategy 1: Deterministic greedy (largest first)
    current_S = greedy_build(sorted(universe, reverse=True))
    update_best(current_S)
    current_S = local_improve(current_S)

    # Strategy 2: Multiple randomized restarts with different heuristics
    rng = random.Random(42)
    num_restarts = 200
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Vary the ordering strategy
        strategy = r % 4
        if strategy == 0:
            # Pure random
            perm = universe.copy()
            rng.shuffle(perm)
        elif strategy == 1:
            # Random with bias toward larger numbers
            perm = sorted(universe, key=lambda x: -x + rng.random() * 5)
        elif strategy == 2:
            # Random with bias toward smaller numbers (sometimes helps)
            perm = sorted(universe, key=lambda x: x + rng.random() * 5)
        else:
            # Weighted random: larger numbers more likely to be picked earlier
            weights = [x * x for x in universe]
            perm = []
            pool = universe.copy()
            while pool:
                total_w = sum(weights[i] for i in range(len(pool)))
                r_val = rng.random() * total_w
                cum = 0
                for idx, x in enumerate(pool):
                    cum += weights[idx]
                    if r_val <= cum:
                        perm.append(x)
                        pool.pop(idx)
                        break
    
        current_S = greedy_build(perm)
        update_best(current_S)
        current_S = local_improve(current_S)

    # Strategy 3: Focused improvement on the best set found so far
    if best_set:
        current_S = local_improve(list(best_set))
        update_best(current_S)

    # Strategy 4: Try to reach size 15+ by targeted search
    # If we haven't found size 15, try more aggressive local search
    if best_size < 15 and time.time() < deadline:
        # Try all pairs of swaps from the best set
        S = list(best_set)
        for _ in range(200):
            if time.time() > deadline:
                break
            rem = [x for x in universe if x not in S]
            found = False
            for a in S:
                if time.time() > deadline:
                    break
                for b in rem:
                    if time.time() > deadline:
                        break
                    new_S = [x for x in S if x != a] + [b]
                    if is_dissociated(new_S):
                        S = new_S
                        update_best(S)
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try double swaps: remove two, add two
                for i, a in enumerate(S):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(S)):
                        if time.time() > deadline:
                            break
                        b = S[j]
                        rem = [x for x in universe if x not in S]
                        for c in rem:
                            if time.time() > deadline:
                                break
                            for d in rem:
                                if d == c:
                                    continue
                                if time.time() > deadline:
                                    break
                                new_S = [x for x in S if x != a and x != b] + [c, d]
                                if is_dissociated(new_S):
                                    S = new_S
                                    update_best(S)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            if not found:
                break

    if not best_set:
        best_set = [22]

    return best_set

if __name__ == "__main__":
    print(construct())
