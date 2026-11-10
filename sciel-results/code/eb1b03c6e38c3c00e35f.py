"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _conflicts(n):
    """Return list of conflict neighbors for each node in 1..n."""
    # conflicts[i] = list of j > i such that (i+j) divides (i*j)
    conflicts = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].append(b)
                conflicts[b].append(a)
    return conflicts

def _greedy_max_is(conflicts, n, order=None):
    """Greedy maximum independent set given an ordering."""
    if order is None:
        order = list(range(1, n + 1))
    selected = set()
    for v in order:
        ok = True
        for u in conflicts[v]:
            if u in selected:
                ok = False
                break
        if ok:
            selected.add(v)
    return selected

def _local_search(conflicts, n, initial, deadline):
    """Local search: try swapping out a selected vertex for an unselected one."""
    selected = set(initial)
    unselected = set(range(1, n + 1)) - selected
    best = selected
    best_size = len(selected)
    
    def can_add(v, sel):
        for u in conflicts[v]:
            if u in sel:
                return False
        return True
    
    # Try single additions
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for v in list(unselected):
            if can_add(v, selected):
                selected.add(v)
                unselected.discard(v)
                changed = True
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(selected)
    
    # Try swaps: remove one, add one or more
    for _ in range(2000):
        if time.time() >= deadline:
            break
        if not unselected:
            break
        v = random.choice(list(unselected))
        # Find which selected vertices conflict with v
        blockers = [u for u in conflicts[v] if u in selected]
        if not blockers:
            selected.add(v)
            unselected.discard(v)
            if len(selected) > best_size:
                best = set(selected)
                best_size = len(selected)
            continue
        # Try removing one blocker and adding v
        for b in blockers:
            new_sel = selected - {b}
            if can_add(v, new_sel):
                # Check if we can also add b back or other unselected
                selected = new_sel
                unselected.add(b)
                unselected.discard(v)
                selected.add(v)
                if len(selected) > best_size:
                    best = set(selected)
                    best_size = len(selected)
                break
        # Try removing two blockers
        if len(blockers) >= 2:
            for i in range(len(blockers)):
                for j in range(i + 1, len(blockers)):
                    new_sel = selected - {blockers[i], blockers[j]}
                    if can_add(v, new_sel):
                        selected = new_sel
                        unselected.add(blockers[i])
                        unselected.add(blockers[j])
                        unselected.discard(v)
                        selected.add(v)
                        if len(selected) > best_size:
                            best = set(selected)
                            best_size = len(selected)
                        break
                if len(selected) > best_size:
                    break
    
    # Final greedy fill
    for v in list(unselected):
        if time.time() >= deadline:
            break
        if can_add(v, selected):
            selected.add(v)
            unselected.discard(v)
            if len(selected) > best_size:
                best = set(selected)
                best_size = len(selected)
    
    return best

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill
    
    n = 256
    conflicts = _conflicts(n)
    
    # Start with all odds (known valid, size 128)
    odds = set(range(1, 257, 2))
    
    # Greedy with different orderings
    best = _greedy_max_is(conflicts, n, list(range(1, n + 1)))
    if len(best) < len(odds):
        best = set(odds)
    
    # Try greedy with random orderings
    for _ in range(50):
        if time.time() >= deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = _greedy_max_is(conflicts, n, order)
        if len(s) > len(best):
            best = s
    
    # Local search from best
    best = _local_search(conflicts, n, best, deadline)
    
    return sorted(best)
# EVOLVE-BLOCK-END
