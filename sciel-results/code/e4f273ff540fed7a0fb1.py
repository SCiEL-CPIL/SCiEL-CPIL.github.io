"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

def _check_validity(selected):
    """Check if the set is valid: no element divides sum of any 3 distinct others."""
    n = len(selected)
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [x for j, x in enumerate(sel) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints."""
    # Check 1: x doesn't divide sum of any 3 from selected
    if len(selected) >= 3:
        for t in combinations(selected, 3):
            if (t[0] + t[1] + t[2]) % x == 0:
                return False
    # Check 2: no existing element a divides sum of any 3 from (selected + [x]) \ {a}
    for a in selected:
        others_without_a = [v for v in selected if v != a]
        if len(others_without_a) >= 2:
            for i in range(len(others_without_a)):
                for j in range(i + 1, len(others_without_a)):
                    if (others_without_a[i] + others_without_a[j] + x) % a == 0:
                        return False
    return True

def _safety_score(x, selected):
    """Score how 'safe' it is to add x to selected. Higher = safer."""
    if not selected:
        return 0.0
    score = 0.0
    n = len(selected)
    for a in selected:
        others = [v for v in selected if v != a]
        violations = 0
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (others[i] + others[j] + x) % a == 0:
                        violations += 1
        score -= violations * 10.0
        if n >= 3:
            for t in combinations(selected, 3):
                if (t[0] + t[1] + t[2]) % x == 0:
                    score -= 50.0
                    break
    score += x * 0.1
    return score

def _greedy_with_safety(order=None, time_limit=10.0):
    """Greedy with safety scoring: pick the safest candidate at each step."""
    if order is None:
        order = list(range(192, 0, -1))
    selected = []
    start = time.time()
    remaining = list(order)
    
    while remaining and time.time() - start < time_limit and len(selected) < 48:
        best_x = None
        best_score = -float('inf')
        for x in remaining:
            if _can_add(x, selected):
                s = _safety_score(x, selected)
                if s > best_score:
                    best_score = s
                    best_x = x
        if best_x is None:
            break
        selected.append(best_x)
        remaining.remove(best_x)
    
    return selected

def _greedy_large_first(order=None, time_limit=50.0):
    """Greedy: try to add elements in given order, skip if invalid."""
    if order is None:
        order = list(range(192, 0, -1))
    selected = []
    start = time.time()
    for x in order:
        if time.time() - start > time_limit:
            break
        if len(selected) >= 48:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit=20.0):
    """Improve by swapping elements."""
    if not _check_validity(selected):
        selected = list(selected)
        changed = True
        start_fix = time.time()
        while changed and time.time() - start_fix < 5.0:
            changed = False
            for i in range(len(selected)):
                a = selected[i]
                others = [selected[j] for j in range(len(selected)) if j != i]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        selected.remove(a)
                        changed = True
                        break
                if changed:
                    break
        return selected
    
    start = time.time()
    best = list(selected)
    
    for iteration in range(1000):
        if time.time() - start > time_limit:
            break
        
        candidates = [x for x in range(192, 0, -1) if x not in best]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        for x in candidates[:30]:
            if _can_add(x, best):
                best.append(x)
                improved = True
                if len(best) >= 48:
                    return best
                break
        
        if not improved and len(best) >= 2:
            for i in range(len(best)):
                a = best[i]
                temp_best = best[:i] + best[i+1:]
                for x in candidates[:15]:
                    if _can_add(x, temp_best):
                        best = temp_best + [x]
                        improved = True
                        break
                if improved:
                    break
        
        if not improved and len(best) >= 4:
            for i in range(len(best)):
                for j in range(i + 1, len(best)):
                    temp_best = best[:i] + best[i+1:j] + best[j+1:]
                    for x in candidates[:10]:
                        if _can_add(x, temp_best):
                            temp_best2 = temp_best + [x]
                            for y in candidates[:10]:
                                if y != x and _can_add(y, temp_best2):
                                    best = temp_best2 + [y]
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def _polish(selected, time_limit=10.0):
    """Final polish: try to add elements in order of safety score."""
    if not selected:
        return selected
    start = time.time()
    best = list(selected)
    
    while time.time() - start < time_limit and len(best) < 48:
        candidates = [x for x in range(192, 0, -1) if x not in best]
        if not candidates:
            break
        
        scored = []
        for x in candidates:
            if _can_add(x, best):
                s = _safety_score(x, best)
                scored.append((s, x))
        
        if not scored:
            break
        
        scored.sort(reverse=True)
        added = False
        for s, x in scored[:5]:
            if _can_add(x, best):
                best.append(x)
                added = True
                break
        
        if not added:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0
    
    best_result = []
    
    result = _greedy_large_first(order=list(range(192, 0, -1)), time_limit=8.0)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() < deadline - 20:
        result = _greedy_with_safety(order=list(range(192, 0, -1)), time_limit=10.0)
        if len(result) > len(best_result):
            best_result = result
    
    for attempt in range(20):
        if time.time() > deadline - 25:
            break
        
        random.seed(attempt * 137 + 42)
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        
        result = _greedy_large_first(order=candidates, time_limit=4.0)
        if len(result) > len(best_result):
            best_result = result
    
    for low in [100, 120, 140, 150, 160, 170, 180]:
        if time.time() > deadline - 20:
            break
        order = list(range(192, low - 1, -1)) + list(range(low - 1, 0, -1))
        result = _greedy_large_first(order=order, time_limit=4.0)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() < deadline - 15 and len(best_result) > 0:
        improved = _local_search(best_result, time_limit=15.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    if time.time() < deadline - 5 and len(best_result) > 0:
        improved = _polish(best_result, time_limit=8.0)
        if len(improved) > len(best_result):
            best_result = improved
    
    if time.time() < deadline - 10:
        random.seed(999)
        candidates = list(range(192, 0, -1))
        random.shuffle(candidates)
        result = _greedy_large_first(order=candidates, time_limit=5.0)
        if len(result) > len(best_result):
            best_result = result
        if time.time() < deadline - 5:
            improved = _local_search(best_result, time_limit=5.0)
            if len(improved) > len(best_result):
                best_result = improved
            if time.time() < deadline - 2:
                improved = _polish(best_result, time_limit=3.0)
                if len(improved) > len(best_result):
                    best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
