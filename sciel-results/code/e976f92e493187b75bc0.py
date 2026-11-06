"""Fast search for large dissociated reciprocal subsets of [1,22] using integer arithmetic."""
import time
import random

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

# Precompute sorted candidate orderings for different heuristics
# Smaller elements have larger reciprocals (inv values), which are "harder" to fit
# Larger elements have smaller reciprocals, easier to fit
_INV_VAL = {x: _INV[x] for x in range(1, 23)}
# Sort by reciprocal value descending (hardest first) or ascending (easiest first)
_ORDER_DESC = sorted(_ALL, key=lambda x: _INV_VAL[x], reverse=True)  # 1,2,3,...
_ORDER_ASC = sorted(_ALL, key=lambda x: _INV_VAL[x])  # 22,21,20,...
# Mixed: alternate large and small
_ORDER_ALT = []
large = [x for x in _ALL if x >= 12]
small = [x for x in _ALL if x < 12]
i, j = 0, 0
while i < len(large) or j < len(small):
    if i < len(large):
        _ORDER_ALT.append(large[i])
        i += 1
    if j < len(small):
        _ORDER_ALT.append(small[j])
        j += 1

def _greedy_order(order, deadline):
    """
    Given an ordering of candidates, greedily add elements maintaining dissociation.
    Returns (set_of_elements, set_of_subset_sums).
    Uses fast incremental collision check.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        inv = _INV[x]
        # collision check: does shifting sums by inv hit sums?
        coll = False
        for s in sums:
            if (s + inv) in sums:
                coll = True
                break
        if coll:
            continue
        # accept
        # update sums: add s+inv for all s
        for s in list(sums):
            sums.add(s + inv)
        chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _complete(chosen, sums, deadline):
    """Try to add more elements to chosen in random order until no more fit."""
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(remaining)
        for x in remaining:
            if x in chosen:
                continue
            inv = _INV[x]
            coll = False
            for s in sums:
                if (s + inv) in sums:
                    coll = True
                    break
            if not coll:
                for s in list(sums):
                    sums.add(s + inv)
                chosen.add(x)
                remaining.remove(x)
                improved = True
                break
    return chosen, sums

def _local_search(chosen, sums, deadline):
    """
    Local search: try swapping one chosen element for one unchosen element
    to maintain or increase size. Also try removing one and adding two.
    Returns improved (chosen, sums) if found.
    """
    if time.time() > deadline:
        return chosen, sums
    
    remaining = [x for x in _ALL if x not in chosen]
    if not remaining:
        return chosen, sums
    
    best_chosen = set(chosen)
    best_sums = set(sums)
    best_size = len(chosen)
    
    # Try single swaps: remove one from chosen, add one from remaining
    for x_out in list(chosen):
        if time.time() > deadline:
            break
        inv_out = _INV[x_out]
        # Try removing x_out
        # New sums = sums without the effect of inv_out
        # sums_without = {s for s in sums if s not in {s2 + inv_out for s2 in sums_without}}
        # This is complex; instead, rebuild from scratch for the candidate set
        candidate = set(chosen) - {x_out}
        # Try adding each remaining element
        for x_in in remaining:
            if x_in in candidate:
                continue
            candidate.add(x_in)
            if len(candidate) != len(chosen):
                candidate.remove(x_in)
                continue
            # Check if candidate is dissociated
            if _is_dissociated(candidate):
                if len(candidate) > best_size:
                    best_size = len(candidate)
                    best_chosen = set(candidate)
                    best_sums = _build_sums(candidate)
            candidate.remove(x_in)
    
    # Try 2-for-1: remove one from chosen, add two from remaining
    if time.time() < deadline:
        for x_out in list(chosen):
            if time.time() > deadline:
                break
            candidate_base = set(chosen) - {x_out}
            for i, x_in1 in enumerate(remaining):
                if time.time() > deadline:
                    break
                if x_in1 in candidate_base:
                    continue
                for x_in2 in remaining[i+1:]:
                    if x_in2 in candidate_base:
                        continue
                    candidate = candidate_base | {x_in1, x_in2}
                    if len(candidate) <= len(chosen):
                        continue
                    if _is_dissociated(candidate):
                        if len(candidate) > best_size:
                            best_size = len(candidate)
                            best_chosen = set(candidate)
                            best_sums = _build_sums(candidate)
    
    if best_size > len(chosen):
        return best_chosen, best_sums
    return chosen, sums

def _build_sums(chosen):
    """Build all subset sums for a set."""
    sums = {0}
    for x in chosen:
        inv = _INV[x]
        for s in list(sums):
            sums.add(s + inv)
    return sums

def _is_dissociated(chosen):
    """Check if a set is dissociated (all subset sums distinct)."""
    sums = _build_sums(chosen)
    return len(sums) == (1 << len(chosen))

def construct():
    start = time.time()
    deadline = start + 50  # safety margin under 60s (evaluator kills at 90s)

    best = set()
    best_size = 0

    # Warm start: try structured orderings
    for order in [_ORDER_DESC, _ORDER_ASC, _ORDER_ALT]:
        if time.time() > deadline:
            break
        chosen, sums = _greedy_order(order, deadline)
        chosen, sums = _complete(chosen, sums, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Random warm starts
    for _ in range(20):
        if time.time() > deadline:
            break
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline)
        chosen, sums = _complete(chosen, sums, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Main loop: random greedy + completion + local search
    while time.time() < deadline - 3:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline - 3)
        # completion
        chosen, sums = _complete(chosen, sums, deadline - 2)
        # local search to try to improve
        if time.time() < deadline - 1:
            chosen, sums = _local_search(chosen, sums, deadline - 1)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Final polish: local search on best
    if best and time.time() < deadline:
        best_sums = _build_sums(best)
        best, best_sums = _local_search(best, best_sums, deadline)

    if not best:
        best = {1}

    return sorted(best)

# EVOLVE-BLOCK-END
