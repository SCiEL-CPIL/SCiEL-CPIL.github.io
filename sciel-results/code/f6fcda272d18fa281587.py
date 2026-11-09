"""Combined approach: fast greedy construction + local search minimization for minimal maximal Sidon sets in [1,256]."""
import time
import random

N = 256
DEADLINE = 55.0  # safety margin under 60s wall-clock

# EVOLVE-BLOCK-START

def _build_sums(S):
    """Return set of all unordered sums a+b with a<=b from S."""
    sums = set()
    n = len(S)
    for i in range(n):
        a = S[i]
        for j in range(i, n):
            sums.add(a + S[j])
    return sums

def _conflicts_valid(S, sums):
    """
    For each element y NOT in S, check if y can be added to S while preserving Sidon.
    Returns dict: y -> True if addable, False if not.
    Uses the precomputed sums for efficiency.
    """
    S_set = set(S)
    n = len(S)
    addable = {}
    for y in range(1, N + 1):
        if y in S_set:
            continue
        # Check if y + a conflicts for any a in S, or 2*y conflicts
        ok = True
        for a in S:
            s = y + a
            if s in sums:
                ok = False
                break
        if ok:
            if 2 * y in sums:
                ok = False
        # Also need to check that all y+a are distinct among themselves
        # Since a are distinct and y is fixed, y+a are all distinct automatically.
        # But we need to check they don't conflict with each other? No, y+a1 != y+a2 if a1!=a2.
        # The only potential issue is if y+a conflicts with existing sums or 2*y.
        addable[y] = ok
    return addable

def _is_maximal(S, sums=None):
    """Check if S is maximal Sidon. Returns (is_maximal, addable_list)."""
    if sums is None:
        sums = _build_sums(S)
    addable = _conflicts_valid(S, sums)
    addable_list = [y for y, ok in addable.items() if ok]
    return len(addable_list) == 0, addable_list

def _greedy_maximal_sidon():
    """Build a maximal Sidon set greedily by scanning and adding when possible."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        # Check if x can be added
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            selected.append(x)
            sums.update([x + a for a in selected[:-1]] + [2 * x])
    return selected

def _local_search_minimize(S, t_start):
    """
    Local search: repeatedly try to remove elements that preserve maximality.
    Also try swaps to enable future removals.
    """
    best = list(S)
    best_sums = _build_sums(best)
    
    while time.time() - t_start < DEADLINE - 2.0:
        improved = False
        
        # Try each element for removal
        for i in range(len(best)):
            if time.time() - t_start > DEADLINE - 2.0:
                break
            
            removed = best[i]
            S_trial = [x for j, x in enumerate(best) if j != i]
            sums_trial = _build_sums(S_trial)
            is_max, addable = _is_maximal(S_trial, sums_trial)
            
            if is_max:
                # Direct removal works!
                best = S_trial
                best_sums = sums_trial
                improved = True
                break
            else:
                # Try swap: remove best[i], add some y from addable
                # We want to eventually be able to remove something else.
                if addable:
                    # Try a few random swaps
                    sample_size = min(5, len(addable))
                    for y in random.sample(addable, sample_size):
                        if time.time() - t_start > DEADLINE - 2.0:
                            break
                        S_swap = S_trial + [y]
                        S_swap.sort()
                        sums_swap = _build_sums(S_swap)
                        
                        # Check if S_swap is Sidon (should be by construction, but verify)
                        # It should be since y was addable to S_trial
                        # Try to remove some other element from S_swap
                        for j in range(len(S_swap)):
                            if time.time() - t_start > DEADLINE - 2.0:
                                break
                            S_trial2 = [x for k, x in enumerate(S_swap) if k != j]
                            sums_trial2 = _build_sums(S_trial2)
                            is_max2, _ = _is_maximal(S_trial2, sums_trial2)
                            if is_max2:
                                # Swap + removal reduces size by 1
                                best = S_trial2
                                best_sums = sums_trial2
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        if not improved:
            # No improvement possible with current local moves.
            # Try perturbation: random removal and re-maximization
            if len(best) > 2 and random.random() < 0.3:
                idx = random.randint(0, len(best) - 1)
                S_pert = [x for j, x in enumerate(best) if j != idx]
                sums_pert = _build_sums(S_pert)
                _, addable_pert = _is_maximal(S_pert, sums_pert)
                if addable_pert:
                    # Add one random element to restore maximal
                    y = random.choice(addable_pert)
                    S_pert.append(y)
                    S_pert.sort()
                    best = S_pert
                    best_sums = _build_sums(best)
            else:
                break
    
    return best

def construct():
    """
    Construct a minimal-size maximal Sidon set in [1,256].
    Uses greedy construction followed by local search minimization.
    """
    t_start = time.time()
    
    # Start with greedy maximal Sidon set
    S = _greedy_maximal_sidon()
    
    # Local search to minimize size
    S = _local_search_minimize(S, t_start)
    
    # Verify the result is valid
    sums = _build_sums(S)
    # Check Sidon property
    n = len(S)
    seen_sums = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in seen_sums:
                # Should not happen, but fallback to greedy
                return _greedy_maximal_sidon()
            seen_sums.add(s)
    
    # Check maximality
    is_max, _ = _is_maximal(S, sums)
    if not is_max:
        # Fallback: re-maximize greedily from current set
        S_set = set(S)
        sums_set = _build_sums(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            ok = True
            for a in S:
                if (x + a) in sums_set:
                    ok = False
                    break
            if ok and (2 * x) in sums_set:
                ok = False
            if ok:
                S.append(x)
                S.sort()
                sums_set.update([x + a for a in S[:-1]] + [2 * x])
        S = sorted(S)
    
    return S

# EVOLVE-BLOCK-END
