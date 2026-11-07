"""Rewritten with branch-and-bound + fast incremental subset-sum conflict detection."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _is_valid(S):
    """Full validity check: no two distinct non-empty subset sums divide one another."""
    if not S:
        return True
    n = len(S)
    if n > 22:
        # Too large for full enumeration; use probabilistic/heuristic check
        # Generate a sample of subset sums
        total = sum(S)
        # Check all single-element sums against all
        singles = set(S)
        # Check pairs
        pairs = set()
        for i in range(n):
            for j in range(i+1, n):
                pairs.add(S[i] + S[j])
        # Check triples if n small enough
        if n <= 15:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        pairs.add(S[i]+S[j]+S[k])
        all_sums = singles | pairs
        all_sums_list = sorted(all_sums)
        all_sums_set = set(all_sums)
        for s in all_sums_list:
            if s == 0:
                continue
            m = 2 * s
            while m <= total:
                if m in all_sums_set:
                    return False
                m += s
        return True
    
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    sums_list = sorted(sums)
    sums_set = sums
    total = sum(S)
    for s in sums_list:
        if s == 0:
            continue
        m = 2 * s
        while m <= total:
            if m in sums_set:
                return False
            m += s
    return True

def _subset_sums_incremental(S):
    """Return the set of all non-empty subset sums of S."""
    n = len(S)
    sums = {0}
    for x in S:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        sums |= new_sums
    sums.discard(0)
    return sums

def _conflict_check(S):
    """Check if S is valid (no two subset sums divide another)."""
    n = len(S)
    if n == 0:
        return True
    if n > 20:
        # Use the larger-sums property: if all elements > total/2, then only singletons
        # can have multiples... but that's not quite right. Let's do partial check.
        # Generate subset sums up to a limit
        total = sum(S)
        sums = {0}
        for x in S:
            new = set()
            for s in sums:
                ns = s + x
                if ns <= total:
                    new.add(ns)
            sums |= new
        sums.discard(0)
        sl = sorted(sums)
        ss = set(sums)
        for s in sl:
            if s == 0:
                continue
            m = 2 * s
            while m <= total:
                if m in ss:
                    return False
                m += s
        return True
    
    sums = set()
    for mask in range(1, 1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
        sums.add(s)
    sums_list = sorted(sums)
    sums_set = sums
    total = sum(S)
    for s in sums_list:
        if s == 0:
            continue
        m = 2 * s
        while m <= total:
            if m in sums_set:
                return False
            m += s
    return True

def _try_add(S, x):
    """Check if adding x to S keeps it valid. Returns True if valid."""
    candidate = S + [x]
    return _conflict_check(candidate)

def _greedy_build(pool, deadline, order="desc"):
    """Greedy construction with given ordering."""
    current = []
    used = set()
    if order == "desc":
        candidates = sorted(pool, reverse=True)
    elif order == "asc":
        candidates = sorted(pool)
    elif order == "random":
        candidates = pool[:]
        random.shuffle(candidates)
    else:
        candidates = pool[:]
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 30:
            break
        if _try_add(current, x):
            current.append(x)
            used.add(x)
    return current

def _branch_and_bound(pool, current, best, deadline):
    """Branch and bound to extend current set."""
    n = len(pool)
    if time.time() > deadline:
        return best
    if len(current) + n <= len(best):
        return best
    if n == 0:
        if len(current) > len(best):
            best = current[:]
        return best
    
    # Try adding pool[0]
    if _try_add(current, pool[0]):
        best = _branch_and_bound(pool[1:], current + [pool[0]], best, deadline)
    # Try skipping pool[0]
    best = _branch_and_bound(pool[1:], current, best, deadline)
    return best

def _local_search(S, pool, deadline):
    """Try to improve S by swapping elements."""
    current = list(S)
    used = set(current)
    available = [x for x in pool if x not in used]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(available)
        for x in available:
            if time.time() > deadline:
                break
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                test = current[:i] + current[i+1:] + [x]
                if _conflict_check(test):
                    removed = current[i]
                    current = test
                    used.discard(removed)
                    used.add(x)
                    available.remove(x)
                    available.append(removed)
                    improved = True
                    break
    return current

def construct():
    start = time.time()
    deadline = start + 55
    pool = list(range(1, 257))
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy strategies
    for order in ["desc", "asc", "random", "random", "random"]:
        if time.time() > deadline:
            break
        s = _greedy_build(pool, deadline, order)
        if len(s) > best_size:
            best_size = len(s)
            best = s
    
    # Phase 2: Local search from best
    if time.time() < deadline:
        improved = _local_search(best, pool, deadline)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)
    
    # Phase 3: More random restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        random.seed(restart * 7919)
        s = _greedy_build(pool, deadline, "random")
        if len(s) > best_size:
            best = s
            best_size = len(s)
        # Local search
        if time.time() < deadline:
            improved = _local_search(best, pool, deadline)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
    
    # Phase 4: Try branch and bound on top of best
    if time.time() < deadline - 10 and best_size < 30:
        remaining = [x for x in pool if x not in set(best)]
        # Sort remaining by some heuristic
        remaining.sort(reverse=True)
        # Limit BnB to avoid explosion
        if len(remaining) <= 15:
            best = _branch_and_bound(remaining, best, best, deadline)
            best_size = len(best)
    
    # Final verification
    if best and not _is_valid(best):
        # Fallback: return empty or a known valid small set
        best = []
    
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
