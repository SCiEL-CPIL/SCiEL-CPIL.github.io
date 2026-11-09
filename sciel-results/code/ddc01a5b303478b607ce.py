"""Combined program: meet-in-the-middle subset sum + multi-strategy search."""
import time
import random
import sys
from itertools import combinations
from collections import defaultdict

L = 5354228880  # lcm(1..24)

def _mitm_subset_sum(weights, target):
    """Meet-in-the-middle subset sum check. weights are integers, target is integer."""
    if target == 0:
        return True
    # Filter: only weights <= target matter
    weights = [w for w in weights if 0 < w <= target]
    if not weights:
        return False
    if sum(weights) < target:
        return False
    
    n = len(weights)
    if n <= 20:
        # Brute force with bitset for small n
        reachable = {0}
        for w in weights:
            new_reach = set()
            for s in reachable:
                ns = s + w
                if ns <= target:
                    new_reach.add(ns)
            reachable |= new_reach
            if target in reachable:
                return True
        return target in reachable
    
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = {0}
    for w in left:
        new_sums = set()
        for s in left_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        left_sums |= new_sums
    
    right_sums = {0}
    for w in right:
        new_sums = set()
        for s in right_sums:
            ns = s + w
            if ns <= target:
                new_sums.add(ns)
        right_sums |= new_sums
    
    # Check if any left_sum + right_sum == target
    for ls in left_sums:
        if (target - ls) in right_sums:
            return True
    return False


def _check_violation(S_list):
    """Check if any element's reciprocal equals sum of reciprocals of a subset of others.
    Returns (violator, subset) or None."""
    n = len(S_list)
    for i, x in enumerate(S_list):
        target = L // x
        others = []
        for j, y in enumerate(S_list):
            if j != i:
                others.append(L // y)
        if _mitm_subset_sum(others, target):
            return x
    return None


def _is_valid(S_list):
    """Check if S_list is a valid set (no reciprocal subsum)."""
    return _check_violation(S_list) is None


def _greedy_build(order, start_set=None):
    """Greedily add elements in given order, starting from start_set if provided."""
    selected = list(start_set) if start_set else []
    selected_set = set(selected)
    for elem in order:
        if elem in selected_set:
            continue
        # Check: can 1/elem be formed by subset of reciprocals of selected?
        target = L // elem
        others_weights = [L // y for y in selected]
        if _mitm_subset_sum(others_weights, target):
            continue
        # Check: does adding elem create a violation for any existing element?
        valid = True
        for a in selected:
            # 1/a = sum of subset of (selected \ {a}) ∪ {elem}
            # = (sum of subset of selected \ {a}) + possibly elem
            # So check: is 1/a - 1/elem formable by subset of selected \ {a}?
            target_a = L // a
            # Case 1: subset doesn't include elem -> already checked (valid before)
            # Case 2: subset includes elem -> need target_a - L//elem formable by subset of selected\{a}
            residual = target_a - (L // elem)
            if residual > 0:
                others_without_a = [L // y for y in selected if y != a]
                if _mitm_subset_sum(others_without_a, residual):
                    valid = False
                    break
            # Also check if target_a == L//elem (i.e., 1/a = 1/elem, only if a==elem, impossible)
            if not valid:
                break
        if valid:
            selected.append(elem)
            selected_set.add(elem)
    return selected


def _repair_expand(S, deadline):
    """Repair phase: remove violators. Expand phase: try adding back."""
    S_set = set(S)
    
    # Repair: iteratively remove elements involved in violations
    while time.time() < deadline - 5:
        violator = _check_violation(list(S_set))
        if violator is None:
            break
        # Find the violating subset to know which elements to consider removing
        # For now, just remove the violator (simplest)
        S_set.discard(violator)
    
    # Expand: try adding back missing elements
    missing = [x for x in range(1, 25) if x not in S_set]
    random.shuffle(missing)
    for m in missing:
        if time.time() > deadline - 5:
            break
        test_set = S_set | {m}
        if _is_valid(list(test_set)):
            S_set = test_set
    
    return list(S_set)


def _local_search(S, deadline):
    """Try to improve S by removing one and adding one or two others."""
    best = list(S)
    best_set = set(best)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for a in list(best_set):
            if time.time() > deadline - 10:
                break
            reduced = [x for x in best if x != a]
            reduced_set = set(reduced)
            # Try adding one element
            for b in range(1, 25):
                if b in reduced_set:
                    continue
                candidate = reduced + [b]
                if _is_valid(candidate):
                    if len(candidate) > len(best):
                        best = candidate
                        best_set = set(best)
                        improved = True
            # Try adding two elements
            if time.time() < deadline - 15:
                for b in range(1, 25):
                    if time.time() > deadline - 15:
                        break
                    if b in reduced_set:
                        continue
                    for c in range(b + 1, 25):
                        if c in reduced_set:
                            continue
                        candidate = reduced + [b, c]
                        if _is_valid(candidate):
                            if len(candidate) > len(best):
                                best = candidate
                                best_set = set(best)
                                improved = True
    return best


def _backtrack_search(remaining, current, deadline, best_size):
    """Backtracking search to find larger valid sets."""
    if time.time() > deadline:
        return
    if len(current) > best_size:
        best_size = len(current)
    if not remaining:
        return
    # Prune
    if len(current) + len(remaining) <= best_size:
        return
    
    for i, elem in enumerate(remaining):
        if time.time() > deadline:
            return
        # Check if elem can be added to current
        target = L // elem
        others_weights = [L // y for y in current]
        if _mitm_subset_sum(others_weights, target):
            continue
        # Check conflicts
        valid = True
        for a in current:
            residual = (L // a) - (L // elem)
            if residual > 0:
                others_without_a = [L // y for y in current if y != a]
                if _mitm_subset_sum(others_without_a, residual):
                    valid = False
                    break
        if valid:
            new_current = current + [elem]
            new_remaining = remaining[:i] + remaining[i+1:]
            best_size = _backtrack_search(new_remaining, new_current, deadline, best_size)
    return best_size


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    best = []
    best_size = 0
    
    # Strategy 1: Multiple greedy orderings
    orderings = [
        list(range(24, 0, -1)),  # descending
        list(range(1, 25)),      # ascending
        list(range(12, 25)) + list(range(1, 12)),  # medium-first
        list(range(18, 25)) + list(range(1, 18)),  # large-first
    ]
    
    for order in orderings:
        if time.time() > deadline - 20:
            break
        result = _greedy_build(order)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Strategy 2: Repair from full set with randomization
    for restart in range(50):
        if time.time() > deadline - 25:
            break
        S = list(range(1, 25))
        random.shuffle(S)
        result = _repair_expand(S, deadline - 15)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Strategy 3: Local search on best
    if time.time() < deadline - 15:
        result = _local_search(best, deadline - 10)
        if len(result) > best_size:
            best_size = len(result)
            best = result
    
    # Strategy 4: Backtracking from a good base
    if time.time() < deadline - 20 and best_size < 22:
        # Try backtracking from various bases
        for base_start in [16, 18, 20]:
            if time.time() > deadline - 10:
                break
            base = list(range(base_start, 25))
            if not _is_valid(base):
                continue
            remaining = [x for x in range(1, base_start) if x not in base]
            bt_deadline = deadline - 8
            new_size = _backtrack_search(remaining, base, bt_deadline, best_size)
            if new_size > best_size:
                best_size = new_size
                # Reconstruct: we need to track the actual set, so let's do a simpler version
                # Actually, let's just do a direct backtracking that returns the set
    
    # Strategy 5: Direct backtracking that tracks the set
    if time.time() < deadline - 15 and best_size < 22:
        def _bt_with_set(remaining, current, deadline, best_ref):
            if time.time() > deadline:
                return
            if len(current) > len(best_ref[0]):
                best_ref[0] = list(current)
            if not remaining:
                return
            if len(current) + len(remaining) <= len(best_ref[0]):
                return
            for i, elem in enumerate(remaining):
                if time.time() > deadline:
                    return
                target = L // elem
                others_weights = [L // y for y in current]
                if _mitm_subset_sum(others_weights, target):
                    continue
                valid = True
                for a in current:
                    residual = (L // a) - (L // elem)
                    if residual > 0:
                        others_without_a = [L // y for y in current if y != a]
                        if _mitm_subset_sum(others_without_a, residual):
                            valid = False
                            break
                if valid:
                    new_current = current + [elem]
                    new_remaining = remaining[:i] + remaining[i+1:]
                    _bt_with_set(new_remaining, new_current, deadline, best_ref)
        
        # Try from a few different starting points
        for base_start in [14, 16, 18]:
            if time.time() > deadline - 10:
                break
            base = list(range(base_start, 25))
            if not _is_valid(base):
                continue
            remaining = [x for x in range(1, base_start) if x not in base]
            best_ref = [best]
            _bt_with_set(remaining, base, deadline - 8, best_ref)
            if len(best_ref[0]) > best_size:
                best_size = len(best_ref[0])
                best = best_ref[0]
    
    # Final polish: try single additions
    if time.time() < deadline - 5:
        best_set = set(best)
        for elem in range(1, 25):
            if time.time() > deadline - 3:
                break
            if elem in best_set:
                continue
            candidate = best + [elem]
            if _is_valid(candidate):
                best = candidate
                best_set.add(elem)
                best_size = len(best)
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return list(best)
# EVOLVE-BLOCK-END
