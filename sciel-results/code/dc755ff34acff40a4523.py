"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import random
import sys

def construct():
    """
    Find a set of classes [[m, r], ...] with distinct odd moduli dividing 51975
    that minimizes the number of uncovered residues modulo 51975.
    """
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    base = 51975

    # Precompute all odd divisors > 1 of 51975
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    # Sort by size (larger moduli cover fewer residues per class, but have more residue choices)
    # We'll use a smarter ordering

    # For each modulus m, precompute the list of residue classes it covers
    # m covers residues r, r+m, r+2m, ... < base
    # Number of residues covered by any single class with modulus m is base // m

    # Greedy construction with local search
    def evaluate(classes):
        """Count uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            covered[r::m] = b'\x01' * ((base - 1 - r) // m + 1)
        return base - sum(covered)

    def greedy_construct(order):
        """Greedy: process moduli in given order, pick best residue for each."""
        covered = bytearray(base)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            # Try all residues
            for r in range(m):
                # Count newly covered
                cnt = 0
                for i in range(r, base, m):
                    if not covered[i]:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            # Mark covered
            for i in range(best_r, base, m):
                covered[i] = 1
        return classes, evaluate(classes)

    def local_search(classes, time_limit=10.0):
        """Local search: try changing residues and swapping moduli."""
        best_classes = [c[:] for c in classes]
        best_score = evaluate(best_classes)
        t0 = time.time()
        
        # Precompute coverage for current solution
        covered = bytearray(base)
        for m, r in best_classes:
            for i in range(r, base, m):
                covered[i] = 1
        
        n = len(best_classes)
        improved = True
        while improved and (time.time() - t0) < time_limit:
            improved = False
            for idx in range(n):
                m, r = best_classes[idx]
                # Try all other residues for this modulus
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Compute new coverage
                    # Remove old class
                    old_covered = bytearray(covered)
                    # Actually, let's compute delta
                    # Count residues that would be lost
                    lost = 0
                    for i in range(r, base, m):
                        # Check if any other class covers i
                        other_covers = False
                        for j in range(n):
                            if j == idx:
                                continue
                            mj, rj = best_classes[j]
                            if i % mj == rj % mj:
                                other_covers = True
                                break
                        if not other_covers:
                            lost += 1
                    
                    # Count residues that would be gained
                    gained = 0
                    for i in range(new_r, base, m):
                        if not covered[i]:
                            gained += 1
                    
                    if gained > lost:
                        # Apply change
                        for i in range(r, base, m):
                            # Check if still covered by others
                            other_covers = False
                            for j in range(n):
                                if j == idx:
                                    continue
                                mj, rj = best_classes[j]
                                if i % mj == rj % mj:
                                    other_covers = True
                                    break
                            if not other_covers:
                                covered[i] = 0
                        for i in range(new_r, base, m):
                            covered[i] = 1
                        
                        best_classes[idx] = [m, new_r]
                        best_score -= lost - gained
                        improved = True
                        break  # Restart scan from beginning
        
        return best_classes, best_score

    # Strategy: Try different orderings of moduli
    # Key insight: larger moduli are more "specialized" - try them first or last?
    # Let's try multiple strategies

    best_overall_score = base
    best_overall_classes = []

    # Strategy 1: Greedy with moduli in decreasing order
    order1 = sorted(divisors, reverse=True)
    if time.time() - start_time < DEADLINE:
        classes, score = greedy_construct(order1)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_classes = classes
        # Local search
        if time.time() - start_time < DEADLINE - 5:
            classes, score = local_search(classes, time_limit=min(15, DEADLINE - (time.time() - start_time)))
            if score < best_overall_score:
                best_overall_score = score
                best_overall_classes = classes

    # Strategy 2: Greedy with moduli in increasing order
    order2 = sorted(divisors)
    if time.time() - start_time < DEADLINE:
        classes, score = greedy_construct(order2)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_classes = classes
        if time.time() - start_time < DEADLINE - 5:
            classes, score = local_search(classes, time_limit=min(15, DEADLINE - (time.time() - start_time)))
            if score < best_overall_score:
                best_overall_score = score
                best_overall_classes = classes

    # Strategy 3: Randomized greedy with multiple restarts
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        num_restarts = min(20, int(remaining_time / 5))
        for restart in range(num_restarts):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Random order
            random_order = divisors[:]
            random.shuffle(random_order)
            classes, score = greedy_construct(random_order)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_classes = classes
            if time.time() - start_time < DEADLINE - 3:
                classes, score = local_search(classes, time_limit=min(5, DEADLINE - (time.time() - start_time)))
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_classes = classes

    # Strategy 4: Start with a subset of most "powerful" moduli
    # Moduli with base//m >= 2 can cover multiple residues
    # Try selecting a good subset
    if time.time() - start_time < DEADLINE - 10:
        # Use all divisors but try different initial residue assignments
        # Simulated annealing style
        covered = bytearray(base)
        classes = []
        for m in sorted(divisors, reverse=True):
            best_r = 0
            best_count = -1
            for r in range(m):
                cnt = sum(1 for i in range(r, base, m) if not covered[i])
                if cnt > best_count:
                    best_count = cnt
                    best_r = r
            classes.append([m, best_r])
            for i in range(best_r, base, m):
                covered[i] = 1
        
        score = base - sum(covered)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_classes = classes

        # More intensive local search
        if time.time() - start_time < DEADLINE - 5:
            classes, score = local_search(classes, time_limit=min(20, DEADLINE - (time.time() - start_time)))
            if score < best_overall_score:
                best_overall_score = score
                best_overall_classes = classes

    return best_overall_classes

# Keep the function signature as required
if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} classes, score computed by evaluator")
