"""Combined program: greedy construction with fast incremental local search and multi-restart."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    base = 51975
    # All divisors > 1 of base (odd divisors since base is odd)
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    divisors_sorted = sorted(divisors)
    divisors_set = set(divisors)

    def compute_covered(classes):
        """Compute coverage array from a list of classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return covered

    def score_of(covered):
        return int(base - covered.sum())

    def greedy_construct(order, time_budget=None):
        """Greedy: process moduli in given order, pick best residue for each."""
        if time_budget is None:
            time_budget = deadline - 25
        covered = np.zeros(base, dtype=np.uint8)
        classes = []
        for m in order:
            if time.time() > time_budget:
                break
            best_r = 0
            best_gain = -1
            for r in range(m):
                gain = int(np.sum(covered[r::m] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = 1
        return classes, covered

    def local_search_fast(classes, covered, max_iterations=200):
        """
        Fast local search using incremental coverage tracking.
        Uses a count array to avoid full recomputation.
        """
        if not classes:
            return classes, covered

        sc = score_of(covered)
        used_moduli = set(m for m, _ in classes)
        improved = True
        iteration = 0

        # Build a coverage count array for incremental updates
        # cov_count[i] = number of classes covering residue i
        cov_count = np.zeros(base, dtype=np.int32)
        for m, r in classes:
            cov_count[r::m] += 1

        def try_change_residue(i, m, old_r, new_r):
            """Try changing residue of class i from old_r to new_r.
            Returns (new_sc, new_cov_count) if improved, else None."""
            # Positions that were only covered by this class at old_r
            old_positions = np.arange(old_r, base, m)
            new_positions = np.arange(new_r, base, m)

            # Gain: positions in new_positions where cov_count == 0 (uncovered)
            gain = int(np.sum(cov_count[new_positions] == 0))
            # Loss: positions in old_positions where cov_count == 1 (only this class)
            loss = int(np.sum(cov_count[old_positions] == 1))

            if gain <= loss:
                return None

            # Apply change
            cov_count[old_positions] -= 1
            cov_count[new_positions] += 1
            new_sc = sc - gain + loss
            return new_sc, cov_count

        def revert_change(m, old_r, new_r):
            """Revert a residue change."""
            old_positions = np.arange(old_r, base, m)
            new_positions = np.arange(new_r, base, m)
            cov_count[old_positions] += 1
            cov_count[new_positions] -= 1

        while improved and iteration < max_iterations and time.time() < deadline - 10:
            improved = False
            iteration += 1

            # Try changing residue of existing classes
            for i in range(len(classes)):
                if time.time() > deadline - 10:
                    return classes, covered
                m, r = classes[i]
                best_r = r
                best_sc = sc
                best_cov = None

                # Generate candidates
                if m <= 60:
                    candidates = list(range(m))
                else:
                    candidates = [r]
                    # Add random candidates
                    for _ in range(min(40, m)):
                        candidates.append(random.randint(0, m - 1))
                    # Ensure 0 and 1 are included
                    if 0 < m:
                        candidates.append(0)
                    if 1 < m:
                        candidates.append(1)
                    # Remove duplicates
                    candidates = list(set(candidates))

                for tr in candidates:
                    if tr == r:
                        continue
                    result = try_change_residue(i, m, r, tr)
                    if result is not None:
                        new_sc, new_cov = result
                        if new_sc < best_sc:
                            # Revert previous best if any
                            if best_cov is not None:
                                revert_change(m, r, best_r)
                            best_sc = new_sc
                            best_r = tr
                            best_cov = new_cov
                        else:
                            # Revert this trial
                            revert_change(m, r, tr)

                if best_r != r:
                    # best_cov already reflects the change
                    cov_count = best_cov
                    classes[i][1] = best_r
                    sc = best_sc
                    improved = True

            # Try adding an unused modulus
            if time.time() < deadline - 15:
                unused = [m for m in divisors if m not in used_moduli]
                if unused:
                    random.shuffle(unused)
                    for m in unused[:8]:
                        if time.time() > deadline - 10:
                            break
                        best_r = 0
                        best_gain = -1
                        # Sample residues
                        for _ in range(min(25, m)):
                            r = random.randint(0, m - 1)
                            gain = int(np.sum(cov_count[r::m] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_r = r
                        # Also check r=0 and r=1 explicitly
                        for r in [0, 1]:
                            if r < m:
                                gain = int(np.sum(cov_count[r::m] == 0))
                                if gain > best_gain:
                                    best_gain = gain
                                    best_r = r
                        if best_gain > 0:
                            classes.append([m, best_r])
                            cov_count[best_r::m] += 1
                            used_moduli.add(m)
                            sc -= best_gain
                            improved = True

            # Try removing a class that contributes nothing unique
            if len(classes) > 3 and time.time() < deadline - 12:
                for i in range(len(classes) - 1, -1, -1):
                    if time.time() > deadline - 10:
                        break
                    m, r = classes[i]
                    positions = np.arange(r, base, m)
                    # Count positions where this class is the only cover
                    unique_covered = int(np.sum(cov_count[positions] == 1))
                    if unique_covered == 0:
                        # Safe to remove
                        cov_count[positions] -= 1
                        classes.pop(i)
                        used_moduli.discard(m)
                        improved = True

        # Rebuild covered from final classes for consistency
        covered_final = compute_covered(classes)
        return classes, covered_final

    best_classes = []
    best_score = base + 1

    # Phase 1: Multiple greedy constructions with different orderings
    num_greedy = 20
    for attempt in range(num_greedy):
        if time.time() > deadline - 30:
            break

        if attempt == 0:
            order = divisors_sorted
        elif attempt == 1:
            order = sorted(divisors, reverse=True)
        elif attempt == 2:
            # Weighted: prioritize medium-large moduli
            order = sorted(divisors, key=lambda m: -(m if m > 10 else 0))
        else:
            random.seed(attempt * 7919 + 13)
            order = divisors_sorted[:]
            random.shuffle(order)

        classes, covered = greedy_construct(order)
        sc = score_of(covered)
        if sc < best_score:
            best_score = sc
            best_classes = [c[:] for c in classes]

        # Phase 2: Local search on this solution
        if time.time() < deadline - 20:
            classes_ls, covered_ls = local_search_fast(classes, covered, max_iterations=80)
            sc_ls = score_of(covered_ls)
            if sc_ls < best_score:
                best_score = sc_ls
                best_classes = [c[:] for c in classes_ls]

    # Phase 3: More aggressive local search on the best solution
    if best_classes and time.time() < deadline - 15:
        covered_best = compute_covered(best_classes)
        best_classes, covered_best = local_search_fast(best_classes, covered_best, max_iterations=300)
        final_sc = score_of(covered_best)
        if final_sc < best_score:
            best_score = final_sc

    # Phase 4: Final polish - try all residue changes one more time
    if best_classes and time.time() < deadline - 5:
        covered_final = compute_covered(best_classes)
        best_classes, covered_final = local_search_fast(best_classes, covered_final, max_iterations=100)

    if not best_classes:
        best_classes = [[3, 0]]

    return best_classes
