"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

_DEADLINE = 85.0  # safety margin below 90s kill

def _total_frac(indices):
    t = Fraction(0, 1)
    for i in indices:
        t += Fraction(1, i + 2)
    return t

def construct():
    global _DEADLINE
    start = time.time()
    random.seed(42)

    # Greedy baseline: take smallest denominators while sum < 1
    selected = []
    total = Fraction(0, 1)
    for i in range(47):
        d = i + 2
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
        else:
            break
    best_indices = list(selected)
    best_total = total
    best_deficit = 1 - total  # positive deficit to minimize

    all_indices = list(range(47))  # indices 0..46 for denominators 2..48

    # Local improvement: try swapping one selected index with one unselected
    # to reduce the deficit (i.e., bring sum closer to 1 from below)
    def try_improve(indices, total, deficit):
        nonlocal best_indices, best_total, best_deficit
        n = len(indices)
        idx_set = set(indices)
        for si in range(n):
            if time.time() > _DEADLINE:
                return
            for uj in all_indices:
                if uj in idx_set:
                    continue
                new_total = total - Fraction(1, indices[si] + 2) + Fraction(1, uj + 2)
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < deficit:
                        new_indices = list(indices)
                        new_indices[si] = uj
                        indices = new_indices
                        total = new_total
                        deficit = new_deficit
        # Also try adding an unselected element if it keeps sum < 1
        for uj in all_indices:
            if uj in set(indices):
                continue
            new_total = total + Fraction(1, uj + 2)
            if new_total < 1:
                new_deficit = 1 - new_total
                if new_deficit < deficit:
                    indices = list(indices) + [uj]
                    total = new_total
                    deficit = new_deficit
        # Try removing an element if it helps (unlikely but check)
        for si in range(len(indices)):
            new_total = total - Fraction(1, indices[si] + 2)
            if new_total < 1:
                new_deficit = 1 - new_total
                if new_deficit < deficit:
                    new_indices = list(indices)
                    del new_indices[si]
                    indices = new_indices
                    total = new_total
                    deficit = new_deficit
        return indices, total, deficit

    # Iterative improvement with random restarts
    for restart in range(200):
        if time.time() > _DEADLINE:
            break
        # Start from best or a random subset
        if restart == 0:
            cur_indices = list(best_indices)
            cur_total = best_total
            cur_deficit = best_deficit
        else:
            # Random perturbation of best
            cur_indices = list(best_indices)
            # Remove 1-3 random elements
            for _ in range(random.randint(1, 3)):
                if cur_indices:
                    ri = random.randint(0, len(cur_indices) - 1)
                    cur_indices.pop(ri)
            # Add 1-2 random unselected elements
            remaining = [x for x in all_indices if x not in cur_indices]
            for _ in range(random.randint(1, 2)):
                if remaining:
                    cur_indices.append(random.choice(remaining))
                    remaining.remove(cur_indices[-1])
            cur_total = _total_frac(cur_indices)
            if cur_total >= 1:
                # Remove one random element to fix
                if cur_indices:
                    ri = random.randint(0, len(cur_indices) - 1)
                    cur_indices.pop(ri)
                    cur_total = _total_frac(cur_indices)
            cur_deficit = 1 - cur_total

        # Local search: repeated single-swap improvements
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            idx_set = set(cur_indices)
            for si in range(len(cur_indices)):
                if time.time() > _DEADLINE:
                    break
                for uj in all_indices:
                    if uj in idx_set:
                        continue
                    new_total = cur_total - Fraction(1, cur_indices[si] + 2) + Fraction(1, uj + 2)
                    if new_total < 1:
                        new_deficit = 1 - new_total
                        if new_deficit < cur_deficit:
                            cur_indices[si] = uj
                            cur_total = new_total
                            cur_deficit = new_deficit
                            improved = True
                            idx_set = set(cur_indices)
                            break
            # Try adding
            for uj in all_indices:
                if time.time() > _DEADLINE:
                    break
                if uj in set(cur_indices):
                    continue
                new_total = cur_total + Fraction(1, uj + 2)
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < cur_deficit:
                        cur_indices.append(uj)
                        cur_total = new_total
                        cur_deficit = new_deficit
                        improved = True
            # Try removing
            for si in range(len(cur_indices) - 1, -1, -1):
                if time.time() > _DEADLINE:
                    break
                new_total = cur_total - Fraction(1, cur_indices[si] + 2)
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < cur_deficit:
                        cur_indices.pop(si)
                        cur_total = new_total
                        cur_deficit = new_deficit
                        improved = True

        if cur_deficit < best_deficit:
            best_deficit = cur_deficit
            best_total = cur_total
            best_indices = list(cur_indices)

    if not best_indices:
        # Fallback: just take index 0 (1/2)
        best_indices = [0]

    return best_indices
# EVOLVE-BLOCK-END
