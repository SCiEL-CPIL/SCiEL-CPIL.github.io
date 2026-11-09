# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # seconds (Safe margin for 60s/90s limits)


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


def _fast_greedy(t0, rng):
    """
    Fast greedy builder with randomization.
    """
    A = []
    B = []
    sumsA = set()
    sumsB = set()
    diffsA = set()
    diffsB = set()

    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    for x in candidates:
        if time.time() - t0 > DEADLINE - 2:
            break
        
        # Try to add to A
        added = False
        ns, nd = _can_add(A, x, sumsA, diffsA)
        if ns is not None and not (nd & diffsB):
            A.append(x)
            sumsA = ns
            diffsA = nd
            added = True
        else:
            # Try to add to B
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                B.append(x)
                sumsB = ns
                diffsB = nd
                added = True
        
        # If not added, skip x

    return A, B


def _local_improve(A, B, t0):
    """
    Try to improve A and B by adding elements or swapping 1 for 2.
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
                
    # Phase 1: Try to add more elements (increases size)
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
            
    # Phase 2: Try 1-for-2 swaps (increases size by 1)
    # We try to remove one element from A and add two new ones.
    # Or remove one from B and add two new ones.
    
    # Helper to recompute state for a list
    def _recompute(lst):
        s = set()
        d = set()
        for i in range(len(lst)):
            for j in range(len(lst)):
                if i <= j:
                    s.add(lst[i] + lst[j])
                if i != j:
                    d.add(lst[j] - lst[i])
        return s, d

    # Try improving A
    for i in range(len(A)):
        if time.time() - t0 > DEADLINE - 2:
            break
        orig = A[i]
        A.pop(i)
        used.discard(orig)
        
        # Recompute A state
        sumsA, diffsA = _recompute(A)
        
        # Try to add two new elements to A
        added_one = None
        added_two = None
        
        for x in range(1, N+1):
            if x in used:
                continue
            ns, nd = _can_add(A, x, sumsA, diffsA)
            if ns is not None and not (nd & diffsB):
                added_one = x
                # Tentatively add x
                A.append(x)
                used.add(x)
                sumsA = ns
                diffsA = nd
                
                for y in range(1, N+1):
                    if y in used or y == x:
                        continue
                    ns2, nd2 = _can_add(A, y, sumsA, diffsA)
                    if ns2 is not None and not (nd2 & diffsB):
                        added_two = y
                        A.append(y)
                        used.add(y)
                        sumsA = ns2
                        diffsA = nd2
                        break
                if added_two is None:
                    # Revert x
                    A.pop()
                    used.discard(x)
                    sumsA, diffsA = _recompute(A)
                break # Found a candidate for first addition, break inner loop? No, we break outer if success?
                # Actually, if we found x but not y, we should try another x?
                # For efficiency, let's just try the first valid x. If it fails to find y, we revert and try next orig?
                # Or we can try to find y for this x. If no y, revert x and try next x?
                # Let's simplify: if we find x, we try to find y. If no y, revert x. Then try next x?
                # This is getting complex. Let's stick to: find first valid x, try to find y. If fail, revert x and stop trying to add to A for this orig?
                # Yes, stop trying to add to A for this orig.
                
        if added_two is None:
            # Revert orig removal
            A.insert(i, orig)
            used.add(orig)
            sumsA, diffsA = _recompute(A)
        # If added_two is not None, we increased size by 1. Keep it.

    # Try improving B
    for i in range(len(B)):
        if time.time() - t0 > DEADLINE - 2:
            break
        orig = B[i]
        B.pop(i)
        used.discard(orig)
        
        # Recompute B state
        sumsB, diffsB = _recompute(B)
        
        # Try to add two new elements to B
        added_one = None
        added_two = None
        
        for x in range(1, N+1):
            if x in used:
                continue
            ns, nd = _can_add(B, x, sumsB, diffsB)
            if ns is not None and not (nd & diffsA):
                added_one = x
                # Tentatively add x
                B.append(x)
                used.add(x)
                sumsB = ns
                diffsB = nd
                
                for y in range(1, N+1):
                    if y in used or y == x:
                        continue
                    ns2, nd2 = _can_add(B, y, sumsB, diffsB)
                    if ns2 is not None and not (nd2 & diffsA):
                        added_two = y
                        B.append(y)
                        used.add(y)
                        sumsB = ns2
                        diffsB = nd2
                        break
                if added_two is None:
                    # Revert x
                    B.pop()
                    used.discard(x)
                    sumsB, diffsB = _recompute(B)
                break
                
        if added_two is None:
            # Revert orig removal
            B.insert(i, orig)
            used.add(orig)
            sumsB, diffsB = _recompute(B)

    return A, B


def construct():
    t0 = time.time()
    rng = random.Random(42)

    best_A, best_B = [], []
    best_size = 0

    # Run multiple fast greedy restarts
    num_restarts = 100
    
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
