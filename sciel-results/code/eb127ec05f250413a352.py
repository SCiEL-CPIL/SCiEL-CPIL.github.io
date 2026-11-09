"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}
_N = P['n']

# Precompute primes and compatibility info at import time
def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _build_conflict_data():
    """
    For each pair (b,c) with b<c, compute the set of a values (a != b, a != c)
    such that a | b*c. This is the "conflict" for adding a given b,c are in the set.
    """
    # conflict[b][c] = frozenset of a in 2..256, a != b, a != c, with a | b*c
    # We'll store as a dict of frozensets for fast lookup
    conflict = {}
    for b in range(2, _N + 1):
        conflict[b] = {}
        for c in range(b + 1, _N + 1):
            bc = b * c
            # Find all a in 2..256, a != b, a != c, such that a | bc
            bad = set()
            # a must divide bc, so a is a divisor of bc
            # But a must be in 2..256
            # Iterate over divisors of bc
            # Better: for a in 2..256, check if a | bc and a != b and a != c
            # But that's O(255) per pair, and we have ~32k pairs, so ~8M ops - fine
            for a in range(2, _N + 1):
                if a == b or a == c:
                    continue
                if bc % a == 0:
                    bad.add(a)
            conflict[b][c] = frozenset(bad)
    return conflict

_CONFLICT = _build_conflict_data()

def _valid_set(S):
    """Check if set S is valid (no three distinct a,b,c in S with a|bc)."""
    S = list(S)
    n = len(S)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = S[i], S[j], S[k]
                # Check all 6 permutations for a|bc
                if (b * c) % a == 0:
                    return False
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
    return True

def _can_add(S, a):
    """Check if adding a to set S keeps it valid."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            b, c = S[i], S[j]
            if b == a or c == a:
                continue
            # Check if a | b*c
            if (b * c) % a == 0:
                return False
            # Check if b | a*c
            if (a * c) % b == 0:
                return False
            # Check if c | a*b
            if (a * b) % c == 0:
                return False
    return True

def _greedy_lower_bound(candidates, current_set):
    """Greedy: add elements in order of fewest conflicts first."""
    S = list(current_set)
    remaining = list(candidates)
    while remaining:
        # Score each candidate by how many conflicts it has with current S
        best_idx = -1
        best_score = float('inf')
        for i, a in enumerate(remaining):
            if not _can_add(S, a):
                continue
            # Count conflicts: number of pairs in S that would be problematic
            # Actually just count how many elements in S it conflicts with
            score = 0
            for b in S:
                for c in S:
                    if b == c:
                        continue
                    # Check if a conflicts with pair (b,c)
                    if (b * c) % a == 0 or (a * c) % b == 0 or (a * b) % c == 0:
                        score += 1
                        break
            if score < best_score:
                best_score = score
                best_idx = i
        if best_idx == -1:
            break
        S.append(remaining[best_idx])
        remaining.pop(best_idx)
    return S

def _branch_and_bound(candidates, current_set, best_size, deadline):
    """
    Branch and bound to find a large valid set.
    candidates: list of elements not yet decided
    current_set: list of elements already in the set
    best_size: current best size found
    deadline: time limit
    """
    if time.time() > deadline:
        return current_set
    
    # Upper bound: |current_set| + len(candidates)
    if len(current_set) + len(candidates) <= best_size:
        return current_set
    
    if not candidates:
        return current_set
    
    # Try to extend greedily to get a good lower bound
    greedy = _greedy_lower_bound(candidates, current_set)
    if len(greedy) > best_size:
        best_size = len(greedy)
        current_set = list(greedy)
        # Remove elements in greedy from candidates
        greedy_set = set(greedy)
        candidates = [x for x in candidates if x not in greedy_set]
    
    if time.time() > deadline:
        return current_set
    
    # Pick the next element to branch on
    # Choose the element with the most conflicts (hardest)
    # Or just pick the first one
    a = candidates[0]
    remaining = candidates[1:]
    
    # Branch 1: include a
    if _can_add(current_set, a):
        new_set = current_set + [a]
        # Filter remaining: remove elements that conflict with a given current_set
        # An element x in remaining conflicts if adding x to new_set is invalid
        # But we can also prune: if x conflicts with a pair in current_set that includes a
        filtered_remaining = []
        for x in remaining:
            if _can_add(new_set, x):
                filtered_remaining.append(x)
        result = _branch_and_bound(filtered_remaining, new_set, best_size, deadline)
        if len(result) > best_size:
            best_size = len(result)
            current_set = result
    
    # Branch 2: exclude a
    result = _branch_and_bound(remaining, current_set, best_size, deadline)
    if len(result) > best_size:
        best_size = len(result)
        current_set = result
    
    return current_set

def _local_search(S, deadline):
    """Try to improve set S by swapping elements."""
    S = list(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_set = set(S)
        # Try adding each element not in S
        for a in range(2, _N + 1):
            if a in S_set:
                continue
            if _can_add(S, a):
                S.append(a)
                S_set.add(a)
                improved = True
        # Try removing one element and adding two others
        # This is more complex; skip for now
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s margin
    
    # Start with primes as a good initial set
    primes = [a for a in range(2, _N + 1) if _is_prime(a)]
    
    # Try to add composites
    current = list(primes)
    
    # Greedy: try to add composites in order of value
    composites = [a for a in range(2, _N + 1) if not _is_prime(a)]
    
    # Sort composites by how "compatible" they are
    # Try each composite and see if it can be added
    added = []
    for a in composites:
        if _can_add(current, a):
            current.append(a)
            added.append(a)
    
    # Now try branch and bound on the remaining composites
    remaining = [a for a in composites if a not in set(current)]
    
    if remaining and time.time() < deadline:
        # Use branch and bound
        result = _branch_and_bound(remaining, current, len(current), deadline)
        if len(result) > len(current):
            current = result
    
    # Local search
    if time.time() < deadline:
        current = _local_search(current, deadline)
    
    # Verify the result
    if not _valid_set(current):
        # Fallback to primes
        return primes
    
    return list(current)
# EVOLVE-BLOCK-END
