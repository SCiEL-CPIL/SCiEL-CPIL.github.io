"""Rewritten program: conflict-graph greedy with random tie-breaking and polish."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 55  # safety margin before 60s kill

    def greedy_random(seed=None):
        if seed is not None:
            random.seed(seed)
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        for x in candidates:
            # Try to add x to A
            new_diffs_A = set()
            valid = True
            for a in A:
                d = abs(x - a)
                if d == 0:
                    continue
                new_diffs_A.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                new_sums = set()
                for a in A:
                    s = x + a
                    if s in sums_A:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_A:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    for a in A:
                        all_diffs.add(abs(x - a))
                    sums_A |= new_sums
                    diffs_A |= new_diffs_A
                    A.append(x)
                    continue
            # Try to add x to B
            new_diffs_B = set()
            valid = True
            for b in B:
                d = abs(x - b)
                if d == 0:
                    continue
                new_diffs_B.add(d)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                new_sums = set()
                for b in B:
                    s = x + b
                    if s in sums_B:
                        valid = False
                        break
                    new_sums.add(s)
                if valid:
                    s = 2 * x
                    if s in sums_B:
                        valid = False
                    else:
                        new_sums.add(s)
                if valid:
                    for b in B:
                        all_diffs.add(abs(x - b))
                    sums_B |= new_sums
                    diffs_B |= new_diffs_B
                    B.append(x)

        return A, B

    def polish(A, B, time_limit=2.0):
        """Try to insert skipped elements into existing sets."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        for a in A:
            for a2 in A:
                sums_A.add(a + a2)
            for a2 in A:
                if a != a2:
                    diffs_A.add(abs(a - a2))

        for b in B:
            for b2 in B:
                sums_B.add(b + b2)
            for b2 in B:
                if b != b2:
                    diffs_B.add(abs(b - b2))

        all_diffs = diffs_A | diffs_B

        start = time.time()
        used = set(A) | set(B)
        remaining = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(remaining)

        for x in remaining:
            if time.time() - start > time_limit:
                break
            # Try A
            valid = True
            for a in A:
                d = abs(x - a)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                for a in A:
                    if x + a in sums_A or 2 * x in sums_A:
                        valid = False
                        break
            if valid and len(A) < max_size:
                for a in A:
                    all_diffs.add(abs(x - a))
                    sums_A.add(x + a)
                sums_A.add(2 * x)
                A.append(x)
                continue
            # Try B
            valid = True
            for b in B:
                d = abs(x - b)
                if d in all_diffs:
                    valid = False
                    break
            if valid:
                for b in B:
                    if x + b in sums_B or 2 * x in sums_B:
                        valid = False
                        break
            if valid and len(B) < max_size:
                for b in B:
                    all_diffs.add(abs(x - b))
                    sums_B.add(x + b)
                sums_B.add(2 * x)
                B.append(x)

        return A, B

    def greedy_with_coloring(seed=None):
        """Build conflict graph implicitly; use random ordering as coloring heuristic."""
        if seed is not None:
            random.seed(seed)
        
        # Random priority ordering for tie-breaking
        priorities = list(range(1, n + 1))
        random.shuffle(priorities)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        all_diffs = set()
        
        for x in priorities:
            # Try A first, then B
            valid_A = True
            for a in A:
                d = abs(x - a)
                if d in all_diffs:
                    valid_A = False
                    break
            if valid_A:
                for a in A:
                    if x + a in sums_A:
                        valid_A = False
                        break
                if valid_A and 2 * x in sums_A:
                    valid_A = False
            if valid_A and len(A) < max_size:
                for a in A:
                    all_diffs.add(abs(x - a))
                    sums_A.add(x + a)
                sums_A.add(2 * x)
                A.append(x)
                continue
            
            valid_B = True
            for b in B:
                d = abs(x - b)
                if d in all_diffs:
                    valid_B = False
                    break
            if valid_B:
                for b in B:
                    if x + b in sums_B:
                        valid_B = False
                        break
                if valid_B and 2 * x in sums_B:
                    valid_B = False
            if valid_B and len(B) < max_size:
                for b in B:
                    all_diffs.add(abs(x - b))
                    sums_B.add(x + b)
                sums_B.add(2 * x)
                B.append(x)
        
        return A, B

    best_A = []
    best_B = []
    best_size = 0
    seed_counter = 0

    # Phase 1: multiple greedy restarts
    while time.time() < deadline - 10:
        A, B = greedy_random(seed=seed_counter)
        A, B = polish(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Phase 2: coloring-based greedy on remaining time
    while time.time() < deadline - 2:
        A, B = greedy_with_coloring(seed=seed_counter)
        A, B = polish(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Final polish on best solution
    if best_size > 0:
        best_A, best_B = polish(best_A, best_B, time_limit=2.0)

    return [best_A, best_B]
