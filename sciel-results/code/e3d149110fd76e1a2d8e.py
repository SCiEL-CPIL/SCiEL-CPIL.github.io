import time
import random
import numpy as np
import itertools

_T = 0.0
_DEAD = 0.0

def construct(q, n, radius):
    global _T, _DEAD
    _T = time.time()
    _DEAD = _T + 115.0  # Safety margin

    # Fallback for general cases or if specific optimization fails
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        base = [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        base = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    best = base
    base_n = len(base)

    # Specific optimization for the hard case K_3(8,1)
    if q == 3 and n == 8 and radius == 1:
        rng = random.Random(2024)
        P = 3 ** n
        pow3 = np.array([1, 3, 9, 27, 81, 243, 729, 2187], dtype=np.int64)
        d = pow3
        r = pow3[::-1]
        idx = np.arange(P, dtype=np.int64)
        cN = ((idx[:, None] // r[None, :]) % 3).astype(np.int64)
        cN = np.ascontiguousarray(cN)

        def add_counts(cov, pos):
            cov[pos] += 1
            for k in range(n):
                base_val = cN[pos, k]
                for t in range(3):
                    if t != base_val:
                        p2 = pos.copy()
                        p2[k] = t
                        cov[p2] += 1

        def remove_counts(cov, pos):
            cov[pos] -= 1
            for k in range(n):
                base_val = cN[pos, k]
                for t in range(3):
                    if t != base_val:
                        p2 = pos.copy()
                        p2[k] = t
                        cov[p2] -= 1

        def greedy_from_seed(seed_indices):
            """
            Start with seed_indices, add words greedily to cover all points.
            """
            cov = np.zeros(P, dtype=np.int64)
            sel = list(seed_indices)
            present = set(seed_indices)
            
            # Initialize coverage with seed
            for s in seed_indices:
                add_counts(cov, cN[s])
            
            # Create a random order of all points
            order = np.arange(P, dtype=np.int64)
            rng.shuffle(order.tolist())
            
            for s in order:
                if s in present:
                    continue
                if cov[s] == 0:
                    sel.append(s)
                    present.add(s)
                    add_counts(cov, cN[s])
                if time.time() > _DEAD - 30:
                    break
            return sel, cov

        def local_search_deletion(cand, cov):
            """
            Remove redundant words. A word is redundant if removing it
            does not uncover any point (i.e., all points in its ball are
            covered by other words).
            """
            changed = True
            while changed:
                if time.time() > _DEAD - 10:
                    break
                changed = False
                L = len(cand)
                order = list(range(L))
                rng.shuffle(order)
                
                for i in order:
                    if time.time() > _DEAD - 10:
                        break
                    if i >= len(cand):
                        continue
                    pos = cand[i]
                    
                    # Temporarily remove
                    remove_counts(cov, cN[pos])
                    
                    # Check if any point is uncovered
                    # np.any(cov == 0) is fast
                    if not np.any(cov == 0):
                        # Word is redundant, remove it
                        del cand[i]
                        changed = True
                        # Note: We do NOT restore counts because we kept the removal
                        # We need to adjust the loop index or just let it continue
                        # Since we are iterating over a fixed 'order' list of indices,
                        # and we deleted an element, subsequent indices in 'order'
                        # might be invalid if they refer to positions > i.
                        # However, 'order' contains indices from the ORIGINAL list.
                        # If we delete cand[i], then cand[i+1] becomes cand[i].
                        # The next index in 'order' might be j > i.
                        # If j was valid before, is it valid now?
                        # If we delete at i, all elements at index > i shift left by 1.
                        # So if 'order' contains j, and j > i, then cand[j] is now the old cand[j+1].
                        # This is a common pitfall.
                        # To be safe, we can just restart the loop or use a while loop with a pointer.
                        # Given the time constraint, restarting the pass is safer.
                        break 
                    else:
                        # Not redundant, restore
                        add_counts(cov, cN[pos])
                
                # If we broke due to a deletion, we restart the pass
                # If we finished the pass without deletion, 'changed' is False and loop ends
                # But if we broke early, we should continue the outer while loop
                # The 'changed' flag is set to True, so the outer loop continues.
                # However, we broke out of the inner for loop.
                # We need to ensure we don't infinite loop if we keep deleting.
                # The size decreases, so it will terminate.
                
            return cand, cov

        # Phase 1: Greedy from scratch (random order)
        try:
            cov = np.zeros(P, dtype=np.int64)
            order = np.arange(P, dtype=np.int64)
            rng.shuffle(order.tolist())
            sel = []
            for s in order:
                if cov[s] == 0:
                    sel.append(s)
                    add_counts(cov, cN[s])
                if time.time() > _DEAD - 40:
                    break
            
            if len(sel) < base_n:
                sel, cov = local_search_deletion(sel, cov)
                if len(sel) < base_n:
                    best = [list(cN[s]) for s in sel]
                    base_n = len(sel)
        except Exception:
            pass

        # Phase 2: Greedy from Tetracode Seed
        if time.time() < _DEAD - 40:
            try:
                # Convert base (tetracode direct sum) to indices
                seed_indices = []
                for word in base:
                    idx_val = 0
                    for k in range(n):
                        idx_val = idx_val * 3 + word[k]
                    seed_indices.append(idx_val)
                
                sel, cov = greedy_from_seed(seed_indices)
                sel, cov = local_search_deletion(sel, cov)
                
                if len(sel) < base_n:
                    best = [list(cN[s]) for s in sel]
                    base_n = len(sel)
            except Exception:
                pass

        # Phase 3: Another random greedy with different seed
        if time.time() < _DEAD - 40:
            try:
                rng2 = random.Random(2025)
                cov = np.zeros(P, dtype=np.int64)
                order = np.arange(P, dtype=np.int64)
                rng2.shuffle(order.tolist())
                sel = []
                for s in order:
                    if cov[s] == 0:
                        sel.append(s)
                        add_counts(cov, cN[s])
                    if time.time() > _DEAD - 10:
                        break
                
                sel, cov = local_search_deletion(sel, cov)
                if len(sel) < base_n:
                    best = [list(cN[s]) for s in sel]
                    base_n = len(sel)
            except Exception:
                pass

    # Final safety check
    if not best:
        best = [list(t) for t in itertools.product(range(q), repeat=n)]
    
    return best
