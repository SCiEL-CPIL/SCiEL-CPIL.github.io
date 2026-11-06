"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _diffs_set(s):
    """Return set of positive differences of sorted list s."""
    out = set()
    n = len(s)
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            out.add(s[j] - si)
    return out

def _valid_sidon(s):
    """Check all positive differences distinct (Sidon)."""
    d = _diffs_set(s)
    return len(d) == len(s) * (len(s) - 1) // 2

def _try_add(a, x):
    """Try adding x to sorted set a; return new sorted list if Sidon, else None."""
    if x in a:
        return None
    # quick check: new differences x - v for v in a where v < x, and v - x for v > x
    new_diffs = set()
    for v in a:
        d = abs(v - x)
        if d in new_diffs:
            return None
        new_diffs.add(d)
    # check against existing diffs
    existing = _diffs_set(a)
    if new_diffs & existing:
        return None
    # merge sorted
    import bisect
    trial = a + [x]
    trial.sort()
    return trial

def _greedy_max_coverage(deadline):
    """Maximum-coverage greedy: at each step pick smallest x that covers most missing diffs."""
    a = [0]
    target = set(range(1, 31))
    existing_diffs = set()
    
    while target:
        if time.time() > deadline:
            break
        best_x = None
        best_cov = 0
        best_trial = None
        
        # Candidate x values: for each missing d and each existing v, x = v + d or v - d (if >= 0)
        candidates = set()
        for d in target:
            for v in a:
                x1 = v + d
                if 0 <= x1 <= 10**6:
                    candidates.add(x1)
                x2 = v - d
                if 0 <= x2 <= 10**6:
                    candidates.add(x2)
        
        # Also add small values near 0 that might be useful
        max_a = max(a)
        for x in range(1, min(max_a + 40, 200)):
            candidates.add(x)
        
        for x in sorted(candidates):
            if x in a:
                continue
            trial = _try_add(a, x)
            if trial is None:
                continue
            # compute coverage: how many missing diffs does this new element create?
            new_diffs = set()
            for v in a:
                new_diffs.add(abs(v - x))
            cov = len(new_diffs & target)
            if cov > best_cov:
                best_cov = cov
                best_x = x
                best_trial = trial
                # early stop if we cover all remaining
                if best_cov == len(target):
                    break
            elif cov == best_cov and best_x is not None and x < best_x:
                # tie-break: smaller x
                best_x = x
                best_trial = trial
        
        if best_trial is None:
            # Fallback: try larger values
            found = False
            y = max(a) + 1
            while y < 10**6 and time.time() < deadline:
                trial = _try_add(a, y)
                if trial is not None:
                    new_diffs = set(abs(v - y) for v in a)
                    cov = len(new_diffs & target)
                    if cov > 0:
                        best_trial = trial
                        best_x = y
                        found = True
                        break
                y += 1
            if not found:
                # Emergency: just add a far-away element
                y = 2 * max(a) + 100
                while True:
                    trial = _try_add(a, y)
                    if trial is not None:
                        best_trial = trial
                        best_x = y
                        break
                    y += 1
                if time.time() > deadline:
                    break
        
        a = best_trial
        # update target
        all_diffs = _diffs_set(a)
        target -= all_diffs
    
    return a

def _local_search(a, deadline):
    """Try to reduce diameter via 1-move and 1-swap local search."""
    a = sorted(a)
    best = max(a)
    best_a = a
    
    def try_improve(set_a):
        nonlocal best, best_a
        sa = sorted(set_a)
        if not _valid_sidon(sa):
            return
        if not set(range(1, 31)).issubset(_diffs_set(sa)):
            return
        if max(sa) < best:
            best = max(sa)
            best_a = sa
    
    # 1-move: remove one element, try re-adding at a smaller position
    for i in range(len(a)):
        if time.time() > deadline:
            break
        without = a[:i] + a[i+1:]
        # try adding back at various positions
        for pos in range(0, max(without) + 1):
            trial = without + [pos]
            if len(set(trial)) != len(trial):
                continue
            if _valid_sidon(sorted(trial)):
                try_improve(trial)
    
    # 1-swap: replace one element with another
    for i in range(len(a)):
        if time.time() > deadline:
            break
        without = a[:i] + a[i+1:]
        for pos in range(0, max(without) + 1):
            if pos in without:
                continue
            trial = without + [pos]
            if _valid_sidon(sorted(trial)):
                try_improve(trial)
    
    return best_a

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    result = _greedy_max_coverage(deadline)
    
    # Verify it's valid
    if not _valid_sidon(result):
        # fallback to simple construction
        result = [0]
        for d in range(1, 31):
            existing = _diffs_set(result)
            if d in existing:
                continue
            for x in range(1, 2000):
                trial = _try_add(result, x)
                if trial is not None:
                    new_d = set(abs(v - x) for v in result)
                    if d in new_d:
                        result = trial
                        break
            else:
                # add two elements
                y = 2 * max(result) + d + 1
                while True:
                    trial = _try_add(result, y)
                    if trial is not None:
                        trial2 = _try_add(trial, y + d)
                        if trial2 is not None:
                            result = trial2
                            break
                    y += 1
    
    # Local search to improve
    if time.time() < deadline - 5:
        result = _local_search(result, deadline)
    
    return result
# EVOLVE-BLOCK-END
