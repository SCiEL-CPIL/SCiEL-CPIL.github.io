"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256
import time
import random

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    """Check if a set is Sidon (all pairwise sums distinct, incl. diagonals)."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Get all pairwise sums a+b with a<=b."""
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _can_add(sums, selected, x):
    """Can x be added to selected (given its sums) while staying Sidon?"""
    for a in selected:
        s = a + x
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _is_maximal_fast(selected, sums):
    """Check maximality: no missing element can be added. Early exit."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        if _can_add(sums, selected, x):
            return False
    return True

def _greedy_construct(order, N):
    """Greedy construction using a given order; runs to completion -> maximal."""
    selected = []
    sums = set()
    for x in order:
        if _can_add(sums, selected, x):
            selected.append(x)
            for a in selected[:-1]:
                sums.add(a + x)
            sums.add(2 * x)
    return selected

def _try_remove_elements(selected, N, deadline):
    """Repeatedly remove an element that keeps the set maximal (shrinks it)."""
    selected = list(selected)
    while time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                return selected
            trial = selected[:i] + selected[i + 1:]
            trial_sums = _get_sums(trial)
            if _is_maximal_fast(trial, trial_sums):
                selected = trial
                improved = True
                break
        if not improved:
            break
    return selected

def _swap_refine(selected, N, deadline):
    """Swap-based refinement: remove an element, then greedily re-cover what it
    was blocking. Keeps cardinality equal or smaller while staying maximal."""
    selected = list(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(selected)):
            if time.time() > deadline:
                return selected
            base = selected[:i] + selected[i + 1:]
            base_sums = _get_sums(base)
            # which missing elements does base fail to block?
            base_set = set(base)
            need = []
            for x in range(1, N + 1):
                if x in base_set:
                    continue
                if _can_add(base_sums, base, x):
                    need.append(x)
            if not need:
                # base itself is maximal -> pure removal win
                selected = base
                improved = True
                break
            # greedily add from `need` to re-block everything
            cand = list(base)
            cand_sums = set(base_sums)
            cand_set = set(base)
            for x in need:
                if time.time() > deadline:
                    break
                if _can_add(cand_sums, cand, x):
                    cand.append(x)
                    cand_set.add(x)
                    for a in cand[:-1]:
                        cand_sums.add(a + x)
                    cand_sums.add(2 * x)
            # accept only if maximal and no larger
            if len(cand) <= len(selected) and _is_maximal_fast(cand, cand_sums):
                selected = cand
                improved = True
                break
    return selected

def construct():
    deadline = time.time() + 25  # safety margin before 30s kill

    best = None
    best_size = float('inf')

    def _consider(result):
        nonlocal best, best_size
        if result is None:
            return
        if len(result) < best_size:
            best = result
            best_size = len(result)

    # Baselines: ascending (guaranteed maximal, fast) and descending
    asc_order = list(range(1, N + 1))
    _consider(_greedy_construct(asc_order, N))
    if time.time() < deadline:
        _consider(_greedy_construct(list(range(N, 0, -1)), N))

    # Randomized greedy trials + local refinement (the real driver of small size)
    rng = random.Random(42)
    while time.time() < deadline - 1.0:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        result = _greedy_construct(order, N)
        if len(result) < best_size:
            # refine the promising candidate
            result = _try_remove_elements(result, N, deadline - 1.0)
            if time.time() < deadline - 1.0:
                result = _swap_refine(result, N, deadline - 1.0)
            _consider(result)

    # Final global refinement of the best set found
    if best is not None and time.time() < deadline - 0.5:
        refined = _try_remove_elements(best, N, deadline - 0.5)
        if time.time() < deadline - 0.5:
            refined = _swap_refine(refined, N, deadline - 0.5)
        if len(refined) < best_size:
            best = refined

    if best is None:
        best = _greedy_construct(asc_order, N)

    # Safety: ensure validity; fall back to guaranteed-valid ascending greedy
    if not _is_sidon(best) or not _is_maximal_fast(best, _get_sums(best)):
        best = _greedy_construct(asc_order, N)

    return sorted(best)
# EVOLVE-BLOCK-END
