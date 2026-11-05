import time
import random
from itertools import combinations

def _is_sidon_fast(arr):
    n = len(arr)
    seen = set()
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            s = ai + arr[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(arr):
    n = len(arr)
    for k in range(n, 0, -1):
        for subset in combinations(arr, k):
            if _is_sidon_fast(subset):
                return k
    return 1

def _collision_count(arr):
    n = len(arr)
    sums = []
    for i in range(n):
        ai = arr[i]
        for j in range(i, n):
            sums.append(ai + arr[j])
    from collections import Counter
    c = Counter(sums)
    total = 0
    for v in c.values():
        if v > 1:
            total += v - 1
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Start with a known bad set (consecutive integers)
    best_set = list(range(12))
    best_size = _max_sidon_size(best_set)

    random.seed(42)

    candidates = []

    # 1. Small range sets (most collisions)
    for _ in range(500):
        span = random.randint(12, 50)
        lo = random.randint(0, max(0, 128 - span))
        hi = min(lo + span, 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))

    # 2. APs (arithmetic progressions) - very structured
    for step in range(2, 40):
        max_start = 128 - 11 * step
        if max_start < 0:
            continue
        for start in range(0, max_start + 1, max(1, step // 3)):
            arr = [start + i * step for i in range(12)]
            if max(arr) <= 128:
                candidates.append(arr)

    # 3. Quadratic-like sequences
    for a in range(1, 15):
        for b in range(0, 20):
            arr = [a * i * i + b * i for i in range(12)]
            if max(arr) <= 128:
                candidates.append(sorted(arr))

    # 4. Clustered: multiple groups
    for _ in range(300):
        num_clusters = random.randint(2, 4)
        arr = []
        cluster_size = 12 // num_clusters
        remaining = 12 - cluster_size * num_clusters
        for c in range(num_clusters):
            size = cluster_size + (1 if c < remaining else 0)
            lo = random.randint(0, 120)
            hi = min(lo + random.randint(5, 15), 128)
            if hi - lo + 1 >= size:
                cluster = random.sample(range(lo, hi + 1), size)
                arr.extend(cluster)
        if len(arr) == 12 and len(set(arr)) == 12:
            candidates.append(sorted(arr))

    # 5. Random full range
    for _ in range(300):
        candidates.append(sorted(random.sample(range(129), 12)))

    # 6. Very tight clusters (nearly all in small range)
    for _ in range(200):
        lo = random.randint(0, 100)
        hi = min(lo + random.randint(12, 25), 128)
        if hi - lo + 1 >= 12:
            candidates.append(sorted(random.sample(range(lo, hi + 1), 12)))

    # Deduplicate
    seen_sets = set()
    unique_candidates = []
    for c in candidates:
        t = tuple(c)
        if t not in seen_sets:
            seen_sets.add(t)
            unique_candidates.append(c)

    # Evaluate candidates - prioritize by collision count (higher = likely smaller Sidon)
    unique_candidates.sort(key=lambda x: -_collision_count(x))

    for arr in unique_candidates:
        if time.time() > deadline - 15.0:
            break
        size = _max_sidon_size(arr)
        if size < best_size:
            best_size = size
            best_set = arr[:]
            if best_size <= 4:
                break

    # Local search: try to increase collisions and decrease Sidon size
    if best_set and time.time() < deadline - 10.0:
        current = best_set[:]
        current_size = best_size
        current_collisions = _collision_count(current)
        stall = 0
        max_stall = 150

        while stall < max_stall and time.time() < deadline - 5.0:
            stall += 1
            improved = False
            n = len(current)

            # Single replacement to increase collisions
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                old = current[i]
                for _ in range(50):
                    new_val = random.randint(0, 128)
                    if new_val == old or new_val in current:
                        continue
                    current[i] = new_val
                    current.sort()
                    nc = _collision_count(current)
                    if nc > current_collisions:
                        ns = _max_sidon_size(current)
                        if ns < current_size:
                            current_size = ns
                            current_collisions = nc
                            improved = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = current[:]
                                stall = 0
                                if best_size <= 3:
                                    return best_set
                        else:
                            current[i] = old
                            current.sort()
                    else:
                        current[i] = old
                        current.sort()

            # Pairwise swap to increase collisions
            if not improved:
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        current[i], current[j] = current[j], current[i]
                        current.sort()
                        nc = _collision_count(current)
                        if nc > current_collisions:
                            ns = _max_sidon_size(current)
                            if ns < current_size:
                                current_size = ns
                                current_collisions = nc
                                improved = True
                                if current_size < best_size:
                                    best_size = current_size
                                    best_set = current[:]
                                    stall = 0
                                    if best_size <= 3:
                                        return best_set
                            else:
                                current[i], current[j] = current[j], current[i]
                                current.sort()
                        else:
                            current[i], current[j] = current[j], current[i]
                            current.sort()

        if current_size < best_size:
            best_size = current_size
            best_set = current[:]

    return best_set
