import time
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit

    NEED = set(range(1, 31))
    MAXN = 32

    def is_valid(a):
        """Check if a is a valid Sidon set containing 0, with all differences 1..30 present."""
        if 0 not in a:
            return False
        a = sorted(set(a))
        if len(a) > MAXN:
            return False
        diffs = set()
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                d = a[j] - ai
                if d in diffs:
                    return False
                diffs.add(d)
        return NEED.issubset(diffs)

    def get_diffs(a):
        a = sorted(set(a))
        diffs = set()
        for i in range(len(a)):
            ai = a[i]
            for j in range(i + 1, len(a)):
                diffs.add(a[j] - ai)
        return diffs

    def greedy_build(skip_prob=0.0, max_val=5000):
        """Greedy construction: add smallest x that maintains Sidon property."""
        A = [0]
        diffs = set()
        for x in range(1, max_val):
            new_diffs = diffs.copy()
            ok = True
            for a in A:
                d = x - a
                if d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            A.append(x)
            diffs = new_diffs
            if NEED.issubset(diffs):
                break
        return A

    def try_reduce_max(a, time_limit=1.0):
        """Try to reduce the maximum element by moving elements to smaller positions."""
        a = sorted(set(a))
        if not is_valid(a):
            return a

        cur = list(a)
        cur_max = max(cur)
        best = list(cur)
        best_max = cur_max

        end_time = time.time() + time_limit
        while time.time() < end_time:
            # Try to move the largest element to a smaller position
            max_idx = len(cur) - 1
            max_val = cur[max_idx]
            found = False

            # Try moving largest element down
            for new_val in range(max_val - 1, 0, -1):
                if new_val in cur:
                    continue
                test = list(cur)
                test[max_idx] = new_val
                test = sorted(set(test))
                if is_valid(test):
                    cur = test
                    cur_max = new_val
                    if cur_max < best_max:
                        best = list(cur)
                        best_max = cur_max
                    found = True
                    break

            if not found:
                # Try moving other elements to free up space
                for idx in range(len(cur) - 2, 0, -1):
                    found = False
                    for new_val in range(cur[idx] - 1, 0, -1):
                        if new_val in cur:
                            continue
                        test = list(cur)
                        test[idx] = new_val
                        test = sorted(set(test))
                        if is_valid(test):
                            cur = test
                            cur_max = max(cur)
                            if cur_max < best_max:
                                best = list(cur)
                                best_max = cur_max
                            found = True
                            break
                    if found:
                        break

            if not found:
                # Try a random perturbation
                idx = random.randint(1, len(cur) - 1)
                delta = random.randint(-10, 10)
                new_val = cur[idx] + delta
                if new_val <= 0 or new_val in cur:
                    continue
                test = list(cur)
                test[idx] = new_val
                test = sorted(set(test))
                if is_valid(test):
                    cur = test
                    cur_max = max(cur)
                    if cur_max < best_max:
                        best = list(cur)
                        best_max = cur_max

        return best

    def local_search(a, time_limit=2.0):
        """More aggressive local search: try all single-element moves."""
        a = sorted(set(a))
        if not is_valid(a):
            return a

        cur = list(a)
        cur_max = max(cur)
        best = list(cur)
        best_max = cur_max

        end_time = time.time() + time_limit
        while time.time() < end_time:
            improved = False
            for idx in range(len(cur)):
                if idx == 0:
                    continue  # Keep 0 fixed
                old_val = cur[idx]
                # Try moving this element to a smaller value
                for new_val in range(old_val - 1, 0, -1):
                    if new_val in cur:
                        continue
                    test = list(cur)
                    test[idx] = new_val
                    test = sorted(set(test))
                    if is_valid(test):
                        new_max = max(test)
                        if new_max < best_max:
                            cur = test
                            best = list(cur)
                            best_max = new_max
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try random perturbations
                for _ in range(10):
                    idx = random.randint(1, len(cur) - 1)
                    delta = random.randint(-15, 15)
                    new_val = cur[idx] + delta
                    if new_val <= 0 or new_val in cur:
                        continue
                    test = list(cur)
                    test[idx] = new_val
                    test = sorted(set(test))
                    if is_valid(test):
                        new_max = max(test)
                        if new_max < best_max:
                            cur = test
                            best = list(cur)
                            best_max = new_max
                            improved = True
                            break
                if not improved:
                    break

        return best

    # Phase 1: Greedy construction with different skip probabilities
    best_a = greedy_build(skip_prob=0.0, max_val=5000)
    best_max = max(best_a)

    for skip_prob in [0.05, 0.1, 0.15, 0.2]:
        if time.time() > deadline - 10:
            break
        a = greedy_build(skip_prob=skip_prob, max_val=5000)
        if is_valid(a) and max(a) < best_max:
            best_a = a
            best_max = max(a)

    # Phase 2: Local search to reduce max
    if time.time() < deadline - 5:
        best_a = local_search(best_a, time_limit=min(5.0, deadline - time.time() - 5))
        best_max = max(best_a)

    # Phase 3: Try more aggressive local search with random restarts
    for restart in range(10):
        if time.time() > deadline - 2:
            break

        # Start from a perturbed version of best_a
        if restart == 0:
            start_a = list(best_a)
        else:
            # Create a perturbed starting point
            start_a = list(best_a)
            for _ in range(3):
                idx = random.randint(1, len(start_a) - 1)
                delta = random.randint(-20, 20)
                new_val = start_a[idx] + delta
                if new_val > 0 and new_val not in start_a:
                    start_a[idx] = new_val
                    start_a = sorted(set(start_a))

        if not is_valid(start_a):
            continue

        result = local_search(start_a, time_limit=min(3.0, max(0.5, deadline - time.time() - 2)))
        if is_valid(result) and max(result) < best_max:
            best_a = result
            best_max = max(result)

    # Final validation
    if is_valid(best_a):
        return best_a

    # Fallback: return a known valid set
    a = greedy_build(skip_prob=0.0, max_val=10000)
    return a
