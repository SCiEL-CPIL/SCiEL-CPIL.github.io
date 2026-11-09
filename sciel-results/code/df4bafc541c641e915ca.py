"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_valid(selected):
    """Check if the selected set is valid."""
    n = len(selected)
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _count_conflicts(candidate, selected):
    """Count how many violations candidate would create."""
    count = 0
    for a in selected:
        others = [v for v in selected if v != a]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                count += 1
    # Check if candidate itself causes violations
    for a in selected:
        others = [v for v in selected if v != a]
        for triple in combinations(others, 3):
            if sum(triple) % candidate == 0:
                count += 1
    return count

def _greedy_largest_first():
    """Greedy: try adding largest numbers first."""
    selected = []
    for x in range(192, 0, -1):
        candidate = selected + [x]
        valid = True
        for a in candidate:
            others = [v for v in candidate if v != a]
            for triple in combinations(others, 3):
                if sum(triple) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected.append(x)
    return selected

def _greedy_with_priority():
    """Greedy with priority: pick numbers that cause fewest conflicts."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining:
        best = None
        best_score = float('inf')
        for x in sorted(remaining, reverse=True):
            candidate = selected + [x]
            # Quick check
            valid = True
            for a in candidate:
                others = [v for v in candidate if v != a]
                for triple in combinations(others, 3):
                    if sum(triple) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                score = len(candidate)
                if best is None or score > best_score:
                    best = x
                    best_score = score
        if best is None:
            break
        selected.append(best)
        remaining.remove(best)
    return selected

def _local_search(initial, time_limit=5):
    """Try to improve by swapping elements."""
    selected = list(initial)
    best = selected[:]
    start = time.time()
    
    while time.time() - start < time_limit:
        improved = False
        for i in range(len(selected)):
            for x in range(1, 193):
                if x in selected:
                    continue
                candidate = selected[:]
                candidate[i] = x
                candidate.sort(reverse=True)
                if _check_valid(candidate):
                    if len(candidate) > len(best):
                        best = candidate[:]
                        selected = candidate[:]
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break
    return best

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    # First pass: greedy largest first
    result = _greedy_largest_first()
    best = result[:]
    
    # Second pass: greedy with priority
    if time.time() - start_time < deadline - 10:
        result2 = _greedy_with_priority()
        if len(result2) > len(best):
            best = result2[:]
    
    # Third pass: local search improvement
    if time.time() - start_time < deadline - 5:
        remaining_time = deadline - (time.time() - start_time) - 2
        if remaining_time > 1:
            result3 = _local_search(best, time_limit=min(remaining_time, 20))
            if len(result3) > len(best):
                best = result3[:]
    
    return best
# EVOLVE-BLOCK-END
