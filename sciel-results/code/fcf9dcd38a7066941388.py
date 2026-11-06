"""Optimized program: fast greedy with efficient maximality check and local search."""
import random
import time

N = 256
DEADLINE = 55.0  # safety margin before 60s kill


def _compute_sums(sel):
    """Compute all pair sums a+b with a<=b for a Sidon set sel."""
    sums = set()
    n = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i, n):
            sums.add(a + sel[j])
    return sums


def _is_sidon(sel):
    """Check if sel is a Sidon set by checking all pair sums are distinct."""
    sums = set()
    n = len_sel = len(sel)
    for i in range(n):
        a = sel[i]
        for j in range(i, n):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _can_add(x, sel, sums):
    """Check if x can be added to sel while preserving Sidon property."""
    for a in sel:
        s = x + a
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True


def _greedy_construct(order):
    """
    Greedily build a Sidon set following the given order of candidates.
    Returns the constructed set.
    """
    selected = []
    sums = set()
    sel_set = set()
    for x in order:
        if x in sel_set:
            continue
        # Check if x can be added
        can = True
        for a in selected:
            s = x + a
            if s in sums:
                can = False
                break
        if can and 2 * x in sums:
            can = False
        if can:
            selected.append(x)
            sel_set.add(x)
            for a in selected:
                sums.add(x + a)
    return selected


def _is_maximal_fast(sel, sums):
    """
    Check if sel is inclusion-maximal in [1, N].
    Returns True if no missing element can be added.
    """
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        can = True
        for a in sel:
            s = x + a
            if s in sums:
                can = False
                break
        if can and 2 * x in sums:
            can = False
        if can:
            return False
    return True


def _local_search(sel, sums, max_iterations=500):
    """
    Try to reduce the size of the maximal Sidon set by removing elements
    and re-saturating.
    """
    best_sel = list(sel)
    best_size = len(sel)
    best_sums = set(sums)
    
    for _ in range(max_iterations):
        # Try removing each element
        for i in range(len(best_sel)):
            candidate = best_sel[:i] + best_sel[i+1:]
            if len(candidate) < best_size - 1:
                continue
            # Check if candidate is still Sidon
            cand_sums = _compute_sums(candidate)
            if not _is_sidon(candidate):
                continue
            # Try to re-saturate: add elements that can be added
            new_sel = list(candidate)
            new_sums = set(cand_sums)
            new_sel_set = set(new_sel)
            improved = False
            for x in range(1, N + 1):
                if x in new_sel_set:
                    continue
                can = True
                for a in new_sel:
                    s = x + a
                    if s in new_sums:
                        can = False
                        break
                if can and 2 * x in new_sums:
                    can = False
                if can:
                    new_sel.append(x)
                    new_sel_set.add(x)
                    for a in new_sel:
                        new_sums.add(x + a)
                    improved = True
            if not improved:
                # Already maximal
                if len(new_sel) < best_size:
                    best_sel = new_sel
                    best_size = len(new_sel)
                    best_sums = new_sums
            else:
                # After re-saturation, check if smaller
                if len(new_sel) < best_size:
                    # Verify it's maximal
                    if _is_maximal_fast(new_sel, new_sums):
                        best_sel = new_sel
                        best_size = len(new_sel)
                        best_sums = new_sums
    return best_sel, best_sums


def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, 256].
    Uses randomized greedy with multiple trials and local search.
    """
    start_time = time.time()
    best_result = None
    best_size = N + 1
    
    # First, do a deterministic ascending greedy to get a baseline
    order = list(range(1, N + 1))
    sel = _greedy_construct(order)
    if len(sel) < best_size:
        best_size = len(sel)
        best_result = sel
    
    # Randomized greedy trials
    rng = random.Random(42)
    num_trials = 0
    
    while time.time() - start_time < DEADLINE:
        # Create a random order
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel = _greedy_construct(order)
        sums = _compute_sums(sel)
        
        # Check if maximal
        if _is_maximal_fast(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_result = sel
                # Try local search to improve
                if time.time() - start_time < DEADLINE - 5:
                    improved_sel, improved_sums = _local_search(sel, sums, max_iterations=200)
                    if len(improved_sel) < best_size:
                        best_size = len(improved_sel)
                        best_result = improved_sel
        
        num_trials += 1
    
    # Final check: ensure we return a valid maximal Sidon set
    if best_result is None:
        # Fallback: ascending greedy
        best_result = _greedy_construct(list(range(1, N + 1)))
    
    return best_result
