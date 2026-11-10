import time
import random
import numpy as np

N = 256
# Hard deadline: 58s wall-clock (safety margin before 60s limit)
HARD_DEADLINE = 58.0


def _is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _is_maximal(S, N=N):
    """Check if S is inclusion-maximal Sidon in [1,N]."""
    if not S or not _is_sidon(S):
        return False
    S_set = set(S)
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.add(S[i] + S[j])
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # x can be added iff 2*x not in sums AND x+s not in sums for all s in S
        if 2 * x in sums:
            continue
        blocked = False
        for s in S:
            if x + s in sums:
                blocked = True
                break
        if not blocked:
            return False
    return True


def _greedy_construct(seed, start_time, hard_deadline):
    """
    Greedy set-cover: repeatedly pick the addable element that blocks
    the most unblocked elements.
    """
    rng = random.Random(seed)
    S = []
    S_set = set()
    sums = set()
    unblocked = set(range(1, N + 1))
    
    while unblocked:
        if time.time() > hard_deadline - 2:
            break
        # Sample candidates
        if len(unblocked) > 40:
            candidates = rng.sample(list(unblocked), 40)
        else:
            candidates = list(unblocked)
        
        best_x = None
        best_score = -1
        best_blocked = set()
        
        for x in candidates:
            # Check if x is addable
            new_sums = {2 * x}
            ok = True
            for a in S:
                s = x + a
                if s in sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if not ok:
                continue
            
            # Count blocked unblocked elements
            all_sums = sums | new_sums
            blocked = set()
            for y in unblocked:
                if y == x:
                    continue
                if 2 * y in all_sums:
                    blocked.add(y)
                    continue
                y_blocked = False
                for s in S:
                    if y + s in all_sums:
                        y_blocked = True
                        break
                if not y_blocked and (y + x) in all_sums:
                    y_blocked = True
                if y_blocked:
                    blocked.add(y)
            
            # Score: number of newly blocked elements
            score = len(blocked)
            if score > best_score:
                best_score = score
                best_x = x
                best_blocked = blocked
        
        if best_x is None:
            # Try all unblocked
            for x in unblocked:
                new_sums = {2 * x}
                ok = True
                for a in S:
                    s = x + a
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
                if not ok:
                    continue
                all_sums = sums | new_sums
                blocked = set()
                for y in unblocked:
                    if y == x:
                        continue
                    if 2 * y in all_sums:
                        blocked.add(y)
                        continue
                    y_blocked = False
                    for s in S:
                        if y + s in all_sums:
                            y_blocked = True
                            break
                    if not y_blocked and (y + x) in all_sums:
                        y_blocked = True
                    if y_blocked:
                        blocked.add(y)
                best_x = x
                best_blocked = blocked
                best_score = len(blocked)
                break
        
        if best_x is None:
            break
        
        S.append(best_x)
        S_set.add(best_x)
        new_sums = {2 * best_x}
        for a in S[:-1]:
            new_sums.add(best_x + a)
        sums |= new_sums
        unblocked.discard(best_x)
        unblocked -= best_blocked
    
    S.sort()
    return S


def _local_remove(S, start_time, hard_deadline):
    """Try to remove elements from S while maintaining maximality."""
    S = list(S)
    improved = True
    while improved:
        if time.time() > hard_deadline - 1:
            break
        improved = False
        for i in range(len(S)):
            S_new = S[:i] + S[i + 1:]
            if S_new and _is_maximal(S_new):
                S = S_new
                improved = True
                break
    return S


def _hill_climb_swap(S, start_time, hard_deadline):
    """
    Try swaps: replace one element with another to enable further removal.
    """
    S = list(S)
    S_set = set(S)
    improved = True
    while improved:
        if time.time() > hard_deadline - 1:
            break
        improved = False
        for i in range(len(S)):
            if time.time() > hard_deadline - 1:
                break
            for j in range(1, N + 1):
                if j in S_set:
                    continue
                S_new = S[:i] + [j] + S[i + 1:]
                S_new.sort()
                if _is_maximal(S_new):
                    # Try removing elements
                    S_trial = list(S_new)
                    for k in range(len(S_trial)):
                        S_trial2 = S_trial[:k] + S_trial[k + 1:]
                        if S_trial2 and _is_maximal(S_trial2):
                            S = S_trial2
                            S_set = set(S)
                            improved = True
                            break
                    if improved:
                        break
    return S


def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy with multiple seeds
    max_seeds = 500
    seed = 0
    while seed < max_seeds:
        if elapsed() > HARD_DEADLINE - 15:
            break
        S = _greedy_construct(seed, start_time, HARD_DEADLINE - 15)
        if S:
            if elapsed() < HARD_DEADLINE - 8:
                S = _local_remove(S, start_time, HARD_DEADLINE - 8)
            if S and len(S) < best_size:
                best_set = S
                best_size = len(S)
        seed += 1
    
    # Phase 2: More local search
    if best_set is not None and elapsed() < HARD_DEADLINE - 5:
        for _ in range(50):
            if elapsed() > HARD_DEADLINE - 5:
                break
            S_new = _local_remove(best_set, start_time, HARD_DEADLINE - 5)
            if len(S_new) < len(best_set):
                best_set = S_new
                best_size = len(S_new)
    
    # Phase 3: Swap improvement
    if best_set is not None and elapsed() < HARD_DEADLINE - 3:
        S_new = _hill_climb_swap(best_set, start_time, HARD_DEADLINE - 3)
        if len(S_new) < len(best_set):
            best_set = S_new
            best_size = len(S_new)
    
    # Fallback
    if best_set is None:
        S = []
        sums = set()
        rng = random.Random(42)
        elems = list(range(1, N + 1))
        rng.shuffle(elems)
        for x in elems:
            new_sums = {2 * x}
            ok = True
            for a in S:
                s = x + a
                if s in sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if not ok:
                continue
            S.append(x)
            sums |= new_sums
        S.sort()
        best_set = S
    
    return best_set
