import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    n = 256
    max_size = 64

    # Precompute all pairs (i,j) with i<=j for sum constraints, and (i,j) i!=j for diff
    # State: x[k] in {0,1,2} for k in 0..n-1 (element k+1)
    # 0 = not in either set, 1 = in A, 2 = in B

    # Precompute pair indices
    sum_pairs = []  # (i, j) with i <= j, for sum constraints
    for i in range(n):
        for j in range(i, n):
            sum_pairs.append((i, j))

    diff_pairs = []  # (i, j) with i != j, for difference constraints
    for i in range(n):
        for j in range(n):
            if i != j:
                diff_pairs.append((i, j))

    # For each sum value s, list of (i,j) pairs that sum to s
    # s ranges from 2 to 512
    sum_to_pairs = {}
    for i in range(n):
        for j in range(i, n):
            s = (i + 1) + (j + 1)
            if s not in sum_to_pairs:
                sum_to_pairs[s] = []
            sum_to_pairs[s].append((i, j))

    # For each nonzero difference d, list of (i,j) pairs where (j+1)-(i+1) = d
    # d ranges from -(n-1) to (n-1), excluding 0
    diff_to_pairs = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d = (j + 1) - (i + 1)
                if d not in diff_to_pairs:
                    diff_to_pairs[d] = []
                diff_to_pairs[d].append((i, j))

    def compute_violations(x):
        """Count hard constraint violations. Returns (sum_viol, diff_viol, total)."""
        # Sum violations: for each set (A or B), count pairs with same sum
        # For set A (state=1): for each sum value, if more than one pair is active, that's a violation
        sum_viol = 0
        for s, pairs in sum_to_pairs.items():
            # Count active pairs in A
            count_A = 0
            for (i, j) in pairs:
                if x[i] == 1 and x[j] == 1:
                    count_A += 1
            if count_A > 1:
                sum_viol += count_A - 1
            # Count active pairs in B
            count_B = 0
            for (i, j) in pairs:
                if x[i] == 2 and x[j] == 2:
                    count_B += 1
            if count_B > 1:
                sum_viol += count_B - 1

        # Difference violations: for each nonzero d, if both A and B have pairs with that diff
        diff_viol = 0
        for d, pairs in diff_to_pairs.items():
            has_A = False
            has_B = False
            for (i, j) in pairs:
                if x[i] == 1 and x[j] == 1:
                    has_A = True
                    break
            if has_A:
                for (i, j) in pairs:
                    if x[i] == 2 and x[j] == 2:
                        has_B = True
                        break
            if has_A and has_B:
                # Count how many A-pairs and B-pairs contribute
                count_A = 0
                count_B = 0
                for (i, j) in pairs:
                    if x[i] == 1 and x[j] == 1:
                        count_A += 1
                    if x[i] == 2 and x[j] == 2:
                        count_B += 1
                diff_viol += min(count_A, count_B)

        total = sum_viol + diff_viol
        return sum_viol, diff_viol, total

    def compute_score(x):
        """Score = |A| + |B| - penalty for violations."""
        count_A = sum(1 for v in x if v == 1)
        count_B = sum(1 for v in x if v == 2)
        _, _, total_viol = compute_violations(x)
        return count_A + count_B - 10 * total_viol

    def random_initial(x):
        """Generate a random initial state with some density."""
        for k in range(n):
            r = random.random()
            if r < 0.15:
                x[k] = 1
            elif r < 0.30:
                x[k] = 2
            else:
                x[k] = 0

    def walk_sat(x, max_iters, deadline):
        """WalkSAT-style local search on the Max-SAT formulation."""
        _, _, current_viol = compute_violations(x)
        current_score = compute_score(x)
        best_x = x[:]
        best_score = current_score
        best_viol = current_viol

        # Precompute for each variable k, which constraints it participates in
        # This is complex, so we'll use a simpler approach: flip random variables

        for it in range(max_iters):
            if time.time() > deadline:
                break

            # Choose a variable to flip
            if current_viol > 0:
                # Focus on violated constraints: pick a random violated pair, flip one of its elements
                r = random.random()
                if r < 0.5:
                    # Pick a random sum violation
                    violated_sums = []
                    for s, pairs in sum_to_pairs.items():
                        count_A = 0
                        for (i, j) in pairs:
                            if x[i] == 1 and x[j] == 1:
                                count_A += 1
                        if count_A > 1:
                            violated_sums.append(s)
                            break
                        count_B = 0
                        for (i, j) in pairs:
                            if x[i] == 2 and x[j] == 2:
                                count_B += 1
                        if count_B > 1:
                            violated_sums.append(s)
                            break
                    if violated_sums:
                        s = random.choice(violated_sums)
                        pairs = sum_to_pairs[s]
                        active_pairs = [(i, j) for (i, j) in pairs if (x[i] == 1 and x[j] == 1) or (x[i] == 2 and x[j] == 2)]
                        if active_pairs:
                            (i, j) = random.choice(active_pairs)
                            k = random.choice([i, j])
                            old_val = x[k]
                            # Flip to a different value
                            new_val = random.choice([v for v in range(3) if v != old_val])
                            x[k] = new_val
                    else:
                        # Try diff violation
                        violated_diffs = []
                        for d, pairs in diff_to_pairs.items():
                            has_A = False
                            has_B = False
                            for (i, j) in pairs:
                                if x[i] == 1 and x[j] == 1:
                                    has_A = True
                                    break
                            if has_A:
                                for (i, j) in pairs:
                                    if x[i] == 2 and x[j] == 2:
                                        has_B = True
                                        break
                            if has_A and has_B:
                                violated_diffs.append(d)
                                break
                        if violated_diffs:
                            d = random.choice(violated_diffs)
                            pairs = diff_to_pairs[d]
                            active_A = [(i, j) for (i, j) in pairs if x[i] == 1 and x[j] == 1]
                            active_B = [(i, j) for (i, j) in pairs if x[i] == 2 and x[j] == 2]
                            if active_A and active_B:
                                if random.random() < 0.5:
                                    (i, j) = random.choice(active_A)
                                else:
                                    (i, j) = random.choice(active_B)
                                k = random.choice([i, j])
                                old_val = x[k]
                                new_val = random.choice([v for v in range(3) if v != old_val])
                                x[k] = new_val
                        else:
                            k = random.randint(0, n - 1)
                            old_val = x[k]
                            new_val = random.choice([v for v in range(3) if v != old_val])
                            x[k] = new_val
                else:
                    # Pick a random diff violation
                    violated_diffs = []
                    for d, pairs in diff_to_pairs.items():
                        has_A = False
                        has_B = False
                        for (i, j) in pairs:
                            if x[i] == 1 and x[j] == 1:
                                has_A = True
                                break
                        if has_A:
                            for (i, j) in pairs:
                                if x[i] == 2 and x[j] == 2:
                                    has_B = True
                                    break
                        if has_A and has_B:
                            violated_diffs.append(d)
                            break
                    if violated_diffs:
                        d = random.choice(violated_diffs)
                        pairs = diff_to_pairs[d]
                        active_A = [(i, j) for (i, j) in pairs if x[i] == 1 and x[j] == 1]
                        active_B = [(i, j) for (i, j) in pairs if x[i] == 2 and x[j] == 2]
                        if active_A and active_B:
                            if random.random() < 0.5:
                                (i, j) = random.choice(active_A)
                            else:
                                (i, j) = random.choice(active_B)
                            k = random.choice([i, j])
                            old_val = x[k]
                            new_val = random.choice([v for v in range(3) if v != old_val])
                            x[k] = new_val
                        else:
                            k = random.randint(0, n - 1)
                            old_val = x[k]
                            new_val = random.choice([v for v in range(3) if v != old_val])
                            x[k] = new_val
                    else:
                        k = random.randint(0, n - 1)
                        old_val = x[k]
                        new_val = random.choice([v for v in range(3) if v != old_val])
                        x[k] = new_val
            else:
                # No violations: try to increase cardinality
                # Pick a random variable that is 0 and try to set it to 1 or 2
                zero_vars = [k for k in range(n) if x[k] == 0]
                if zero_vars:
                    k = random.choice(zero_vars)
                    x[k] = random.choice([1, 2])
                else:
                    # All variables assigned, try flipping
                    k = random.randint(0, n - 1)
                    old_val = x[k]
                    new_val = random.choice([v for v in range(3) if v != old_val])
                    x[k] = new_val

            # Evaluate
            _, _, new_viol = compute_violations(x)
            new_score = compute_score(x)

            if new_score > best_score or (new_score == best_score and new_viol < best_viol):
                best_x = x[:]
                best_score = new_score
                best_viol = new_viol

            # Accept move if it doesn't increase violations too much (simulated annealing style)
            if new_viol <= current_viol:
                current_viol = new_viol
            else:
                # Probabilistic acceptance
                delta = new_viol - current_viol
                if random.random() < math.exp(-delta / 5.0):
                    current_viol = new_viol
                else:
                    # Revert
                    # We need to revert x[k] but we don't have old_val stored easily
                    # Just keep going (the best_x tracks the best)
                    pass

        return best_x, best_score, best_viol

    best_A = []
    best_B = []
    best_total = 0

    num_restarts = 50
    iters_per_restart = 5000

    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break

        x = [0] * n
        random_initial(x)

        best_x, best_score, best_viol = walk_sat(x, iters_per_restart, deadline - 1)

        if best_viol == 0:
            # Valid solution
            A = [k + 1 for k in range(n) if best_x[k] == 1]
            B = [k + 1 for k in range(n) if best_x[k] == 2]

            # Enforce max_size
            if len(A) > max_size:
                random.shuffle(A)
                A = A[:max_size]
            if len(B) > max_size:
                random.shuffle(B)
                B = B[:max_size]

            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A = A
                best_B = B
        else:
            # Still track best even with violations for potential use
            A = [k + 1 for k in range(n) if best_x[k] == 1]
            B = [k + 1 for k in range(n) if best_x[k] == 2]
            if len(A) > max_size:
                random.shuffle(A)
                A = A[:max_size]
            if len(B) > max_size:
                random.shuffle(B)
                B = B[:max_size]
            # Only accept if no violations
            if best_viol == 0:
                total = len(A) + len(B)
                if total > best_total:
                    best_total = total
                    best_A = A
                    best_B = B

    # If no valid solution found, use a simple fallback
    if not best_A:
        # Try a simple greedy
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x_val in range(1, n + 1):
            if time.time() > deadline - 1:
                break
            # Try A
            if len(A) < max_size:
                ok = True
                if x_val + x_val in sums_A:
                    ok = False
                for a in A:
                    if x_val + a in sums_A:
                        ok = False
                        break
                    if x_val - a in diffs_B:
                        ok = False
                        break
                    if a - x_val in diffs_B:
                        ok = False
                        break
                if ok:
                    A.append(x_val)
                    for a in A[:-1]:
                        sums_A.add(x_val + a)
                        diffs_A.add(x_val - a)
                        diffs_A.add(a - x_val)
                    sums_A.add(x_val + x_val)
                    continue
            # Try B
            if len(B) < max_size:
                ok = True
                if x_val + x_val in sums_B:
                    ok = False
                for b in B:
                    if x_val + b in sums_B:
                        ok = False
                        break
                    if x_val - b in diffs_A:
                        ok = False
                        break
                    if b - x_val in diffs_A:
                        ok = False
                        break
                if ok:
                    B.append(x_val)
                    for b in B[:-1]:
                        sums_B.add(x_val + b)
                        diffs_B.add(x_val - b)
                        diffs_B.add(b - x_val)
                    sums_B.add(x_val + x_val)

        best_A = A if A else [1]
        best_B = B if B else [2]

    return [best_A, best_B]
