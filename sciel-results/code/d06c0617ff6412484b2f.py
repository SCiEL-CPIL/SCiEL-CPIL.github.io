import time
import random

N = 256

def _compute_sums(selected):
    """Compute all pairwise sums a+b for a<=b in selected, return set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        sums.add(2 * a)
        for j in range(i + 1, n):
            sums.add(a + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check if selected is maximal. sums is the set of all pairwise sums of selected."""
    selected_set = set(selected)
    for m in range(1, N + 1):
        if m in selected_set:
            continue
        # Check if adding m creates a collision
        # m + m = 2m must not be in sums
        # m + a must not be in sums for any a in selected
        if 2 * m in sums:
            continue  # collision, so still maximal (adding m breaks Sidon)
        collision = False
        for a in selected:
            if m + a in sums:
                collision = True
                break
        if not collision:
            return False  # adding m doesn't break Sidon, so not maximal
    return True

def _greedy_random(deadline, rng):
    """Generate a Sidon set via randomized greedy."""
    elements = list(range(1, N + 1))
    rng.shuffle(elements)
    selected = []
    sums = set()
    for x in elements:
        if time.time() > deadline:
            break
        # Check if x can be added
        ok = True
        if 2 * x in sums:
            ok = False
        else:
            for a in selected:
                if x + a in sums:
                    ok = False
                    break
        if ok:
            selected.append(x)
            sums.add(2 * x)
            for a in selected[:-1]:
                sums.add(x + a)
    return selected, sums

def _local_search(selected, sums, deadline):
    """Try to remove elements while preserving maximality."""
    selected = list(selected)
    sums = _compute_sums(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            x = selected[i]
            # Try removing x
            new_selected = selected[:i] + selected[i+1:]
            if not new_selected:
                continue
            new_sums = _compute_sums(new_selected)
            if _is_maximal(new_selected, new_sums):
                selected = new_selected
                sums = new_sums
                improved = True
                break
    return selected, sums

def construct():
    deadline = time.time() + 55  # safety margin
    best = None
    best_size = N + 1
    
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        selected, sums = _greedy_random(deadline, rng)
        if not selected:
            continue
        # Ensure it's Sidon (should be by construction)
        if not _is_sidon(selected):
            continue
        # Local search to minimize
        selected, sums = _local_search(selected, sums, deadline)
        if len(selected) < best_size:
            best_size = len(selected)
            best = list(selected)
    
    if best is None:
        # Fallback: ascending greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            ok = True
            if 2 * x in sums:
                ok = False
            else:
                for a in selected:
                    if x + a in sums:
                        ok = False
                        break
            if ok:
                selected.append(x)
                sums.add(2 * x)
                for a in selected[:-1]:
                    sums.add(x + a)
        return selected
    
    return best
