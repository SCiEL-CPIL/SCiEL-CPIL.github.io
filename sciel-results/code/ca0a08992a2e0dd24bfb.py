import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s

    N = 51975

    # Generate all divisors of N greater than 1
    divisors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            if i > 1:
                divisors.append(i)
            other = N // i
            if other != i and other > 1:
                divisors.append(other)
    divisors.sort()

    def compute_uncovered(classes):
        """Compute number of uncovered residues given list of [m, r] classes."""
        covered = bytearray(N)
        for m, r in classes:
            for pos in range(r, N, m):
                covered[pos] = 1
        return sum(1 for x in covered if x == 0)

    def greedy_init(moduli_list, covered=None):
        """Greedy: for each modulus in order, pick residue covering most uncovered."""
        if covered is None:
            covered = bytearray(N)
        classes = []
        for m in moduli_list:
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = 0
                for pos in range(r, N, m):
                    if covered[pos] == 0:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            for pos in range(best_r, N, m):
                covered[pos] = 1
        uncovered = sum(1 for x in covered if x == 0)
        return classes, uncovered, covered

    def local_search(classes, score, time_limit):
        """Local search: try changing one residue at a time, keep best."""
        classes = [c[:] for c in classes]
        score = score
        covered = bytearray(N)
        for m, r in classes:
            for pos in range(r, N, m):
                covered[pos] = 1

        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(len(classes)):
                if (time.time() - start_time) > time_limit:
                    break
                m, r = classes[i]
                # Try all possible residues for this modulus
                best_r = r
                best_score = score

                # Count how many positions this class covers
                class_positions = list(range(r, N, m))

                for new_r in range(m):
                    if new_r == r:
                        continue
                    new_positions = list(range(new_r, N, m))

                    # Compute change in uncovered count
                    # Positions that were covered only by this class become uncovered
                    # Positions that are now covered but weren't before become covered
                    gain = 0
                    loss = 0
                    for pos in new_positions:
                        if covered[pos] == 0:
                            gain += 1
                    for pos in class_positions:
                        # Check if any other class covers this position
                        still_covered = False
                        for j, (mm, rr) in enumerate(classes):
                            if j == i:
                                continue
                            if (pos - rr) % mm == 0:
                                still_covered = True
                                break
                        if not still_covered:
                            loss += 1

                    new_score = score - gain + loss
                    if new_score < best_score:
                        best_score = new_score
                        best_r = new_r

                if best_r != r:
                    # Update covered array
                    for pos in range(r, N, m):
                        covered[pos] = 0
                    for pos in range(best_r, N, m):
                        covered[pos] = 1
                    classes[i] = [m, best_r]
                    score = best_score
                    improved = True
                # else: no change, covered stays the same

        return classes, score

    # Strategy 1: Greedy with all divisors in increasing order
    best_classes, best_score, _ = greedy_init(divisors)

    # Run local search
    if (time.time() - start_time) < deadline - 10:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 5)

    # Strategy 2: Try different orderings and subsets
    random.seed(42)
    remaining = deadline - (time.time() - start_time)

    if remaining > 15:
        # Try greedy with shuffled order
        shuffled = divisors[:]
        random.shuffle(shuffled)
        classes2, score2, _ = greedy_init(shuffled)
        if score2 < best_score:
            best_classes, best_score = classes2, score2
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3)

    if (time.time() - start_time) < deadline - 15:
        # Try with larger moduli first
        divs_desc = sorted(divisors, reverse=True)
        classes3, score3, _ = greedy_init(divs_desc)
        if score3 < best_score:
            best_classes, best_score = classes3, score3
            if (time.time() - start_time) < deadline - 5:
                best_classes, best_score = local_search(best_classes, best_score, deadline - 3)

    # Strategy 3: Random perturbations + local search (simulated annealing-like)
    if (time.time() - start_time) < deadline - 10:
        current_classes = [c[:] for c in best_classes]
        current_score = best_score
        temperature = max(1, best_score // 5)

        for iteration in range(200):
            if (time.time() - start_time) > deadline - 5:
                break

            # Perturb: change one or two residues
            num_changes = random.randint(1, 2)
            old_changes = []
            for _ in range(num_changes):
                idx = random.randint(0, len(current_classes) - 1)
                m, r = current_classes[idx]
                old_r = r
                new_r = random.randint(0, m - 1)
                if new_r != old_r:
                    old_changes.append((idx, m, old_r, new_r))

            if not old_changes:
                continue

            # Apply changes
            for idx, m, old_r, new_r in old_changes:
                current_classes[idx] = [m, new_r]

            # Compute new score
            covered = bytearray(N)
            for mm, rr in current_classes:
                for pos in range(rr, N, mm):
                    covered[pos] = 1
            new_score = sum(1 for x in covered if x == 0)

            delta = new_score - current_score

            if delta < 0 or random.random() < (1.0 / (1.0 + delta / max(1, temperature))):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_classes = [c[:] for c in current_classes]
                    # Run local search from new best
                    if (time.time() - start_time) < deadline - 3:
                        best_classes, best_score = local_search(best_classes, best_score, deadline - 2)
            else:
                # Revert
                for idx, m, old_r, new_r in reversed(old_changes):
                    current_classes[idx] = [m, old_r]

            # Cool down
            temperature = max(1, temperature * 0.95)

    # Final local search if time permits
    if (time.time() - start_time) < deadline - 2:
        best_classes, best_score = local_search(best_classes, best_score, deadline - 1)

    return best_classes
