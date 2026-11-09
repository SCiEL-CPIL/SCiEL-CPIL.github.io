"""Rewritten program: conflict-graph greedy with random tie-breaking, polish, and ILS."""
import time
import random

def construct():
    n = 256
    max_size = 64
    deadline = time.time() + 58  # safety margin before 60s kill

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
            # Try the smaller set first to balance
            try_A_first = len(A) <= len(B)
            if try_A_first:
                order = [(A, sums_A, diffs_A), (B, sums_B, diffs_B)]
            else:
                order = [(B, sums_B, diffs_B), (A, sums_A, diffs_A)]
            
            added = False
            for target_set, target_sums, target_diffs in order:
                if len(target_set) >= max_size:
                    continue
                new_diffs = set()
                valid = True
                for elem in target_set:
                    d = abs(x - elem)
                    if d == 0:
                        continue
                    new_diffs.add(d)
                    if d in all_diffs:
                        valid = False
                        break
                if valid:
                    new_sums = set()
                    for elem in target_set:
                        s = x + elem
                        if s in target_sums:
                            valid = False
                            break
                        new_sums.add(s)
                    if valid:
                        s = 2 * x
                        if s in target_sums:
                            valid = False
                        else:
                            new_sums.add(s)
                    if valid:
                        for elem in target_set:
                            all_diffs.add(abs(x - elem))
                        target_sums |= new_sums
                        target_diffs |= new_diffs
                        target_set.append(x)
                        added = True
                        break

        return A, B

    def polish(A, B, time_limit=2.0):
        """Try to insert skipped elements into existing sets."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        all_diffs = set()

        for i in range(len(A)):
            for j in range(len(A)):
                sums_A.add(A[i] + A[j])
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(abs(A[i] - A[j]))

        for i in range(len(B)):
            for j in range(len(B)):
                sums_B.add(B[i] + B[j])
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(abs(B[i] - B[j]))

        all_diffs = diffs_A | diffs_B

        start = time.time()
        used = set(A) | set(B)
        remaining = [x for x in range(1, n + 1) if x not in used]
        random.shuffle(remaining)

        for x in remaining:
            if time.time() - start > time_limit:
                break
            # Try the smaller set first
            try_A_first = len(A) <= len(B)
            sets_to_try = [(A, sums_A, diffs_A), (B, sums_B, diffs_B)] if try_A_first else [(B, sums_B, diffs_B), (A, sums_A, diffs_A)]
            
            added = False
            for target_set, target_sums, target_diffs in sets_to_try:
                if len(target_set) >= max_size:
                    continue
                valid = True
                for elem in target_set:
                    d = abs(x - elem)
                    if d in all_diffs:
                        valid = False
                        break
                if valid:
                    if 2 * x in target_sums:
                        valid = False
                    else:
                        for elem in target_set:
                            if x + elem in target_sums:
                                valid = False
                                break
                if valid:
                    for elem in target_set:
                        all_diffs.add(abs(x - elem))
                        target_sums.add(x + elem)
                    target_sums.add(2 * x)
                    target_set.append(x)
                    added = True
                    break

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

    def ils_polish(A, B, time_limit=3.0):
        """Iterated local search: try removing random elements and re-polishing."""
        start = time.time()
        current_size = len(A) + len(B)
        improved = True
        
        while improved and time.time() - start < time_limit:
            improved = False
            # Try removing random elements to free up space
            for _ in range(5):
                if time.time() - start > time_limit:
                    break
                trial_A = A[:]
                trial_B = B[:]
                # Remove 1-3 random elements
                num_remove = random.randint(1, 3)
                for _ in range(num_remove):
                    if len(trial_A) > 1 and random.random() < 0.5:
                        trial_A.pop(random.randint(0, len(trial_A) - 1))
                    elif len(trial_B) > 1:
                        trial_B.pop(random.randint(0, len(trial_B) - 1))
                    else:
                        break
                trial_A, trial_B = polish(trial_A, trial_B, time_limit=0.5)
                trial_size = len(trial_A) + len(trial_B)
                if trial_size > current_size:
                    A = trial_A
                    B = trial_B
                    current_size = trial_size
                    improved = True
                    break
        
        return A, B

    # Phase 1: multiple greedy restarts with polish
    while time.time() < deadline - 15:
        A, B = greedy_random(seed=seed_counter)
        A, B = polish(A, B, time_limit=1.0)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Phase 2: ILS on the best solution found
    if best_size > 0 and time.time() < deadline - 5:
        il_A, il_B = ils_polish(best_A[:], best_B[:], time_limit=min(5.0, deadline - time.time() - 3))
        il_size = len(il_A) + len(il_B)
        if il_size > best_size:
            best_size = il_size
            best_A = il_A
            best_B = il_B

    # Phase 3: More greedy restarts if time remains
    while time.time() < deadline - 2:
        A, B = greedy_random(seed=seed_counter)
        A, B = polish(A, B, time_limit=0.8)
        size = len(A) + len(B)
        if size > best_size:
            best_size = size
            best_A = A[:]
            best_B = B[:]
        seed_counter += 1

    # Final polish on best solution
    if best_size > 0 and time.time() < deadline - 0.5:
        remaining_time = max(0.5, deadline - time.time() - 0.2)
        best_A, best_B = polish(best_A, best_B, time_limit=remaining_time)

    return [best_A, best_B]
