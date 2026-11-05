import time
import random
import math

N = 256
MAX_EACH = 64
# Strict deadline to ensure return before 60s evaluator kill
TIME_LIMIT = 55.0


def _valid_pair(A, B):
    """Check both Sidon and disjoint nonzero differences."""
    if not A or not B:
        return False
    
    # Check Sidon for A
    sums_A = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums_A:
                return False
            sums_A.add(s)
            
    # Check Sidon for B
    sums_B = set()
    for i in range(len(B)):
        for j in range(i, len(B)):
            s = B[i] + B[j]
            if s in sums_B:
                return False
            sums_B.add(s)
            
    # Check Disjoint Diffs
    diffs_A = set()
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                diffs_A.add(A[i] - A[j])
                
    diffs_B = set()
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                diffs_B.add(B[i] - B[j])
                
    return len(diffs_A & diffs_B) == 0

def _greedy_build(candidates_order, prefer_A_first=True, deadline=None):
    """Greedy build using fast array-based checks."""
    A = []
    B = []
    
    # Sum ranges: min 1+1=2, max 256+256=512. Index 0 unused or offset.
    # Let's use lists of size 513 (indices 0..512).
    A_sums = [False] * 513
    B_sums = [False] * 513
    
    # Diff ranges: min 1-256=-255, max 256-1=255.
    # Offset by 255. Index 0 corresponds to -255. Index 255 corresponds to 0 (nonzero diffs only, so we don't store 0).
    # Actually, we just need to check if a diff exists in the OTHER set.
    # We store diffs in a set or array? Array is faster for lookup.
    # Size 511 (indices -255..255).
    A_diffs = [False] * 511
    B_diffs = [False] * 511
    
    def get_diff_idx(d):
        return d + 255

    def can_add_to_A(x):
        # Check Sidon A
        if A_sums[2*x]:
            return False
        for a in A:
            if A_sums[x + a]:
                return False
        # Check Diff Disjointness: x-a and a-x must not be in B_diffs
        for a in A:
            d1 = x - a
            d2 = a - x
            if B_diffs[get_diff_idx(d1)] or B_diffs[get_diff_idx(d2)]:
                return False
        return True

    def can_add_to_B(x):
        # Check Sidon B
        if B_sums[2*x]:
            return False
        for b in B:
            if B_sums[x + b]:
                return False
        # Check Diff Disjointness: x-b and b-x must not be in A_diffs
        for b in B:
            d1 = x - b
            d2 = b - x
            if A_diffs[get_diff_idx(d1)] or A_diffs[get_diff_idx(d2)]:
                return False
        return True

    for x in candidates_order:
        if deadline is not None and time.time() > deadline:
            break
        okA = can_add_to_A(x)
        okB = can_add_to_B(x)
        
        if okA and okB:
            if len(A) <= len(B) and prefer_A_first:
                # Add to A
                A_sums[2*x] = True
                for a in A:
                    A_sums[x + a] = True
                    A_diffs[get_diff_idx(x - a)] = True
                    A_diffs[get_diff_idx(a - x)] = True
                A.append(x)
            else:
                # Add to B
                B_sums[2*x] = True
                for b in B:
                    B_sums[x + b] = True
                    B_diffs[get_diff_idx(x - b)] = True
                    B_diffs[get_diff_idx(b - x)] = True
                B.append(x)
        elif okA:
            A_sums[2*x] = True
            for a in A:
                A_sums[x + a] = True
                A_diffs[get_diff_idx(x - a)] = True
                A_diffs[get_diff_idx(a - x)] = True
            A.append(x)
        elif okB:
            B_sums[2*x] = True
            for b in B:
                B_sums[x + b] = True
                B_diffs[get_diff_idx(x - b)] = True
                B_diffs[get_diff_idx(b - x)] = True
            B.append(x)
    return A, B

def _local_search(A, B, max_iters=100, deadline=None):
    """Simplified local search: try adding unused elements to A or B."""
    A = list(A)
    B = list(B)
    
    A_sums = [False] * 513
    B_sums = [False] * 513
    A_diffs = [False] * 511
    B_diffs = [False] * 511
    
    def get_diff_idx(d):
        return d + 255

    # Initialize state
    for a in A:
        A_sums[2*a] = True
        for y in A:
            if y < a: # Avoid double counting? No, just mark true.
                A_sums[a+y] = True
            if y != a:
                A_diffs[get_diff_idx(a-y)] = True
                A_diffs[get_diff_idx(y-a)] = True
    for b in B:
        B_sums[2*b] = True
        for y in B:
            if y < b:
                B_sums[b+y] = True
            if y != b:
                B_diffs[get_diff_idx(b-y)] = True
                B_diffs[get_diff_idx(y-b)] = True

    used = set(A) | set(B)
    unused = [x for x in range(1, N+1) if x not in used]
    
    best_score = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    
    rng = random.Random(123)
    
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if not unused:
            break
            
        x = rng.choice(unused)
        
        # Try adding to A
        okA = True
        if A_sums[2*x]:
            okA = False
        else:
            for a in A:
                if A_sums[x+a]:
                    okA = False
                    break
        if okA:
            for a in A:
                if B_diffs[get_diff_idx(x-a)] or B_diffs[get_diff_idx(a-x)]:
                    okA = False
                    break
                    
        if okA:
            A_sums[2*x] = True
            for a in A:
                A_sums[x+a] = True
                A_diffs[get_diff_idx(x-a)] = True
                A_diffs[get_diff_idx(a-x)] = True
            A.append(x)
            unused.remove(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A, best_B = list(A), list(B)
            # Don't revert, keep the improvement
            continue
            
        # Try adding to B
        okB = True
        if B_sums[2*x]:
            okB = False
        else:
            for b in B:
                if B_sums[x+b]:
                    okB = False
                    break
        if okB:
            for b in B:
                if A_diffs[get_diff_idx(x-b)] or A_diffs[get_diff_idx(b-x)]:
                    okB = False
                    break
                    
        if okB:
            B_sums[2*x] = True
            for b in B:
                B_sums[x+b] = True
                B_diffs[get_diff_idx(x-b)] = True
                B_diffs[get_diff_idx(b-x)] = True
            B.append(x)
            unused.remove(x)
            if len(A) + len(B) > best_score:
                best_score = len(A) + len(B)
                best_A, best_B = list(A), list(B)
            continue
            
        # If neither works, do nothing (or shuffle unused? no, just next iter)
        
    return best_A, best_B


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_size = 0
    best_sets = ([], [])
    
    # Strategy 1: Deterministic greedy with a few orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        
        A2, B2 = _greedy_build(order, prefer_A_first=False, deadline=deadline)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with local search
    rng = random.Random(42)
    # Limit number of restarts to ensure we finish in time
    # Each greedy is fast. Local search is limited.
    max_restarts = 100
    
    for t in range(max_restarts):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates, deadline=deadline)
        
        # Quick check validity (greedy ensures it, but just in case of logic error)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            
            # Local search to improve
            if s >= best_size - 5:
                A_ls, B_ls = _local_search(A, B, max_iters=50, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Final polish on best found
    if best_sets[0] and best_sets[1]:
        # Use remaining time for a more thorough local search on the best solution
        remaining_time = deadline - time.time()
        if remaining_time > 1.0:
            # Estimate iterations based on time
            max_iters = int(remaining_time * 100) # Rough estimate
            A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=max_iters, deadline=deadline)
            if _valid_pair(A_ls, B_ls):
                s = len(A_ls) + len(B_ls)
                if s > best_size:
                    best_size = s
                    best_sets = (list(A_ls), list(B_ls))

    # Fallback
    if not best_sets[0] or not best_sets[1]:
        best_sets = ([1, 10], [2, 20])
        
    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
