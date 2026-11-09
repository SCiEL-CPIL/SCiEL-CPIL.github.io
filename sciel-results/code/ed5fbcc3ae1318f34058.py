"""Rewritten program for two Sidon sets with disjoint difference sets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # seconds (Safe margin for 60s/90s limits)


def _pair_sums_ok(lst):
    """Check if all pair sums (including a+a) in lst are unique."""
    if len(lst) <= 1:
        return True
    sums = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            b = lst[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(lst):
    """Return set of nonzero differences b-a for a,b in lst, a != b."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(len(lst)):
            if i != j:
                d.add(lst[j] - a)
    return d


def _can_add(lst, x, sums_used, diffs_used):
    """
    Check if x can be added to lst maintaining Sidon property
    (pair sums unique) and that new differences don't conflict with diffs_used.
    sums_used: set of all pair sums currently in lst
    diffs_used: set of all nonzero differences currently in lst
    Returns (new_sums, new_diffs) if valid, else (None, None).
    """
    # Check new pair sums
    for a in lst:
        if (a + x) in sums_used:
            return None, None
    if (x + x) in sums_used:
        return None, None
    
    # Check new differences
    for a in lst:
        d1 = x - a
        d2 = a - x
        if d1 in diffs_used or d2 in diffs_used:
            return None, None
    
    # If we pass all checks, we need to return the updated sets
    new_sums = set(sums_used)
    for a in lst:
        new_sums.add(a + x)
    new_sums.add(x + x)

    new_diffs = set(diffs_used)
    for a in lst:
        new_diffs.add(x - a)
        new_diffs.add(a - x)

    return new_sums, new_diffs


def _score_candidate(x, A, B, sumsA, sumsB, diffsA, diffsB):
    """
    Score how good it is to add x to A or B.
    Returns (score_A, score_B) where higher is better.
    """
    score_A = 0
    score_B = 0
    
    # Check if x can be added to A
    can_add_A = True
    for a in A:
        if (a + x) in sumsA:
            can_add_A = False
            break
        d1 = x - a
        d2 = a - x
        if d1 in diffsA or d2 in diffsA:
            can_add_A = False
            break
    if can_add_A:
        if (x + x) in sumsA:
            can_add_A = False
        else:
            # Check difference disjointness with B
            for a in A:
                d1 = x - a
                d2 = a - x
                if d1 in diffsB or d2 in diffsB:
                    can_add_A = False
                    break
    
    # Check if x can be added to B
    can_add_B = True
    for b in B:
        if (b + x) in sumsB:
            can_add_B = False
            break
        d1 = x - b
        d2 = b - x
        if d1 in diffsB or d2 in diffsB:
            can_add_B = False
            break
    if can_add_B:
        if (x + x) in sumsB:
            can_add_B = False
        else:
            # Check difference disjointness with A
            for b in B:
                d1 = x - b
                d2 = b - x
                if d1 in diffsA or d2 in diffsA:
                    can_add_B = False
                    break
    
    # Score: prefer adding to the side that has fewer elements (to balance)
    # and where it causes less "crowding"
    if can_add_A:
        score_A = 100 - len(A)
    if can_add_B:
        score_B = 100 - len(B)
    
    return score_A, score_B


def _fast_greedy(t0, rng, seed_offset=0):
    """
    Fast greedy builder with randomization and scoring.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # To add variation, we can bias the preference
    prefer_A = rng.random() < 0.5

    for x in candidates:
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Score both options
        score_A, score_B = _score_candidate(x, A, B, sumsA, sumsB, diffsA, diffsB)
        
        # Decide which side to try first based on scores
        if score_A >= score_B and score_A > 0:
            sides = [0, 1]
        elif score_B > score_A and score_B > 0:
            sides = [1, 0]
        else:
            # Use random preference
            if prefer_A:
                sides = [0, 1]
            else:
                sides = [1, 0]
        
        added = False
        for side in sides:
            if side == 0:
                ns, nd = _can_add(A, x, sumsA, diffsA)
                if ns is not None:
                    # Check difference disjointness
                    if not (nd & diffsB):
                        A.append(x)
                        sumsA = ns
                        diffsA = nd
                        added = True
                        break
            else:
                ns, nd = _can_add(B, x, sumsB, diffsB)
                if ns is not None:
                    # Check difference disjointness
                    if not (nd & diffsA):
                        B.append(x)
                        sumsB = ns
                        diffsB = nd
                        added = True
                        break
        # If not added, skip x

    return A, B


def _local_improve(A, B, t0):
    """
    Try to improve A and B by swapping elements or adding new ones.
    """
    if not A or not B:
        return A, B
    
    used = set(A) | set(B)
    
    # Recompute state
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()
    
    for x in A:
        for y in A:
            if x <= y:
                sumsA.add(x+y)
            if x != y:
                diffsA.add(y-x)
    for x in B:
        for y in B:
            if x <= y:
                sumsB.add(x+y)
            if x != y:
                diffsB.add(y-x)
    
    # Try to add more elements
    for x in range(1, N+1):
        if time.time() - t0 > DEADLINE - 2:
            break
        if x in used:
            continue
        # Try A
        ns, nd = _can_add(A, x, sumsA, diffsA)
        if ns is not None and not (nd & diffsB):
            A.append(x)
            used.add(x)
            sumsA = ns
            diffsA = nd
            continue
        # Try B
        ns, nd = _can_add(B, x, sumsB, diffsB)
        if ns is not None and not (nd & diffsA):
            B.append(x)
            used.add(x)
            sumsB = ns
            diffsB = nd
            continue
    
    # Try swaps: remove one element from A and try to add a better one
    for _ in range(3):  # Limit iterations
        if time.time() - t0 > DEADLINE - 2:
            break
        
        improved = False
        # Try to improve A
        for i in range(len(A)):
            if time.time() - t0 > DEADLINE - 2:
                break
            orig = A[i]
            # Temporarily remove
            A.pop(i)
            used.discard(orig)
            
            # Recompute
            sumsA = set()
            diffsA = set()
            for x in A:
                for y in A:
                    if x <= y:
                        sumsA.add(x+y)
                    if x != y:
                        diffsA.add(y-x)
            
            # Try to add a better element
            for x in range(1, N+1):
                if x in used:
                    continue
                ns, nd = _can_add(A, x, sumsA, diffsA)
                if ns is not None and not (nd & diffsB):
                    A.append(x)
                    used.add(x)
                    sumsA = ns
                    diffsA = nd
                    improved = True
                    break
            
            # If we didn't find a better one, restore
            if not improved:
                A.insert(i, orig)
                used.add(orig)
                # Recompute
                sumsA = set()
                diffsA = set()
                for x in A:
                    for y in A:
                        if x <= y:
                            sumsA.add(x+y)
                        if x != y:
                            diffsA.add(y-x)
        
        # Try to improve B
        for i in range(len(B)):
            if time.time() - t0 > DEADLINE - 2:
                break
            orig = B[i]
            # Temporarily remove
            B.pop(i)
            used.discard(orig)
            
            # Recompute
            sumsB = set()
            diffsB = set()
            for x in B:
                for y in B:
                    if x <= y:
                        sumsB.add(x+y)
                    if x != y:
                        diffsB.add(y-x)
            
            # Try to add a better element
            for x in range(1, N+1):
                if x in used:
                    continue
                ns, nd = _can_add(B, x, sumsB, diffsB)
                if ns is not None and not (nd & diffsA):
                    B.append(x)
                    used.add(x)
                    sumsB = ns
                    diffsB = nd
                    improved = True
                    break
            
            # If we didn't find a better one, restore
            if not improved:
                B.insert(i, orig)
                used.add(orig)
                # Recompute
                sumsB = set()
                diffsB = set()
                for x in B:
                    for y in B:
                        if x <= y:
                            sumsB.add(x+y)
                        if x != y:
                            diffsB.add(y-x)
        
        if not improved:
            break
    
    return A, B


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    # Check Sidon property for A
    sumsA = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sumsA:
                return False
            sumsA.add(s)
    # Check Sidon property for B
    sumsB = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sumsB:
                return False
            sumsB.add(s)
    # Check difference disjointness
    diffsA = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffsA.add(A[j] - A[i])
    diffsB = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffsB.add(B[j] - B[i])
    if diffsA & diffsB:
        return False
    # Check all elements in [1, N]
    for x in A:
        if x < 1 or x > N:
            return False
    for x in B:
        if x < 1 or x > N:
            return False
    return True


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # Run multiple fast greedy restarts
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Use a different seed for each restart to vary the shuffle
        r = random.Random(42 + i)
        A, B = _fast_greedy(t0, r)
        
        if not A or not B:
            continue
        
        # Local improvement
        A, B = _local_improve(A, B, t0)
        
        if _validate(A, B) and len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = list(A), list(B)
            
            # If we hit the max possible, we can stop early? Max is 128.
            if best_size == 128:
                break

    # Ensure nonempty and valid
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    if not _validate(best_A, best_B):
        # Fallback
        best_A = [1]
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
