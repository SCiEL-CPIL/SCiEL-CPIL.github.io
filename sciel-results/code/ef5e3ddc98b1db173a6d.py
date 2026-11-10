import time
import random

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s wall-clock kill

    # ---- Fast bitset validity check ----
    # sums[k] is a bitset (int) of subset sums of cardinality k.
    # Validity: sum of popcounts of sums[k] equals popcount of union of sums[k].
    
    def is_valid(sums, max_k):
        union = 0
        total_count = 0
        for k in range(max_k + 1):
            s = sums[k]
            if s:
                union |= s
                total_count += s.bit_count()
        return union.bit_count() == total_count

    def try_greedy(order):
        """Greedy: add x if it keeps all cardinality sum-sets disjoint."""
        sums = [0] * 61  # bitset per cardinality
        sums[0] = 1 << 0  # empty subset sum 0
        chosen = []
        max_k = 0

        for x in order:
            if time.time() > deadline:
                break
            
            # Compute new sums
            new_sums = [0] * 61
            for k in range(max_k + 1, 0, -1):
                new_sums[k] = sums[k] | ((sums[k-1] << x))
            new_sums[0] = sums[0]
            new_max_k = max_k + 1

            # Check disjointness using fast popcount method
            if is_valid(new_sums, new_max_k):
                sums = new_sums
                max_k = new_max_k
                chosen.append(x)

        return chosen

    # ---- Initial: ascending and descending ----
    best = try_greedy(range(1, 257))
    if time.time() < deadline - 3:
        r = try_greedy(range(256, 0, -1))
        if len(r) > len(best):
            best = r

    # ---- Randomized orderings ----
    rng = random.Random(42)
    best_len = len(best)
    best_set = set(best)

    for trial in range(500): # Increased trials due to speedup
        if time.time() > deadline - 2:
            break

        # Build a randomized order
        mode = trial % 4
        if mode == 0:
            order = list(range(1, 257))
            rng.shuffle(order)
        elif mode == 1:
            # Bias toward large numbers first
            order = list(range(256, 0, -1))
            rng.shuffle(order)
        elif mode == 2:
            # Interleave small and large
            small = list(range(1, 129))
            large = list(range(256, 127, -1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = []
            si, li = 0, 0
            while si < len(small) or li < len(large):
                if si < len(small):
                    order.append(small[si]); si += 1
                if li < len(large):
                    order.append(large[li]); li += 1
        else:
            # Weighted: prefer numbers far from 128
            order = list(range(1, 257))
            order.sort(key=lambda x: -abs(x - 128))
            rng.shuffle(order)

        r = try_greedy(order)
        if len(r) > best_len:
            best_len = len(r)
            best = r
            best_set = set(best)

    # ---- Polish: try to insert unused integers into the best set ----
    if time.time() < deadline - 1:
        unused = [x for x in range(1, 257) if x not in best_set]
        unused.sort()
        for x in unused:
            if time.time() > deadline - 0.5:
                break
            test_set = best + [x]
            sums = [0] * 61
            sums[0] = 1 << 0
            for v in test_set:
                new_sums = [0] * 61
                for k in range(len(test_set), 0, -1):
                    new_sums[k] = sums[k] | ((sums[k-1] << v))
                new_sums[0] = sums[0]
                sums = new_sums
            
            if is_valid(sums, len(test_set)):
                best = test_set
                best_set = set(best)

    # ---- Return as list of Python ints ----
    return list(best)
