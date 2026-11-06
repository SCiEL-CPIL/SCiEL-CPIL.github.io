"""Rewritten program: local search for closest strict underpacking of 1 by unit fractions."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin within 90s limit

    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]

    # Initial greedy solution: add fractions while sum < 1
    selected = set()
    total = Fraction(0, 1)
    for i in range(n):
        if total + recip[i] < 1:
            selected.add(i)
            total += recip[i]

    best_indices = list(selected)
    best_deficit = 1 - total

    # Local search: try swaps (remove one, add another) and single additions/removals
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all single removals
        for i in list(selected):
            new_total = total - recip[i]
            new_deficit = 1 - new_total
            if new_deficit > best_deficit and new_deficit < best_deficit + 0:
                # This is worse, skip
                pass
            # Actually we want to minimize deficit, so smaller deficit is better
            # Wait: deficit = 1 - sum. We want sum close to 1 but strictly below.
            # So we want to MINIMIZE deficit.
            # If we remove an element, sum decreases, deficit increases -> worse.
            # So removals only help if they allow a better addition elsewhere.
        
        # Try all single additions (of not-selected items)
        for j in range(n):
            if j not in selected:
                if total + recip[j] < 1:
                    new_total = total + recip[j]
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected.add(j)
                        total = new_total
                        best_deficit = new_deficit
                        best_indices = list(selected)
                        improved = True
                        break  # restart after improvement
        
        if improved:
            continue

        # Try all swaps: remove i, add j
        for i in list(selected):
            if time.time() > deadline:
                break
            remaining_after_remove = total - recip[i]
            for j in range(n):
                if j not in selected:
                    new_total = remaining_after_remove + recip[j]
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < best_deficit:
                            selected.remove(i)
                            selected.add(j)
                            total = new_total
                            best_deficit = new_deficit
                            best_indices = list(selected)
                            improved = True
                            break
            if improved:
                break

    # Final check: try 2-swap moves if time permits
    if time.time() < deadline - 5:
        for i1 in list(selected):
            if time.time() > deadline:
                break
            for i2 in list(selected):
                if i1 >= i2:
                    continue
                remaining_after_remove = total - recip[i1] - recip[i2]
                for j1 in range(n):
                    if j1 in selected:
                        continue
                    for j2 in range(n):
                        if j2 <= j1 or j2 in selected:
                            continue
                        new_total = remaining_after_remove + recip[j1] + recip[j2]
                        if new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_deficit:
                                selected.discard(i1)
                                selected.discard(i2)
                                selected.add(j1)
                                selected.add(j2)
                                total = new_total
                                best_deficit = new_deficit
                                best_indices = list(selected)
                                break
                    if new_deficit < best_deficit:
                        break
                if new_deficit < best_deficit:
                    break
            if new_deficit < best_deficit:
                break

    return best_indices
# EVOLVE-BLOCK-END
