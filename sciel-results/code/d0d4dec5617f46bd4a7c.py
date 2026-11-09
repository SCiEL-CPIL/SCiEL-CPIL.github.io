import random
import time

N = 256

def is_sidon(S):
    """Check if S is a Sidon set."""
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def is_maximal(S, N):
    """Check if S is inclusion-maximal in [1,N]."""
    S_set = set(S)
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            sums.add(a + S[j])
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Check if x can be added
        # x is addable if no a in S such that x+a in sums, AND 2x not in sums
        valid = True
        for a in S:
            if (x + a) in sums:
                valid = False
                break
        if valid:
            if (2 * x) in sums:
                valid = False
        if valid:
            return False
    return True

def greedy_sidon(order, N):
    """Greedy Sidon set construction given an order of elements."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def reduce_maximal(S, N):
    """
    Try to reduce the maximal Sidon set by removing redundant elements.
    An element x is removable if S \ {x} is still Sidon (always true) and maximal.
    """
    S = list(S)
    improved = True
    while improved:
        improved = False
        S_set = set(S)
        # Precompute sums
        sums = set()
        for i, a in enumerate(S):
            for j in range(i, len(S)):
                sums.add(a + S[j])
        
        # For each y not in S, determine which a in S block it
        # y is blocked by a if y+a in sums
        # We want to find x such that for all y not in S, y is blocked by S \ {x}
        # and x is blocked by S \ {x}
        
        # Optimize: Iterate through candidates x in S
        # For a candidate x, check if it is removable.
        
        candidates = list(S)
        random.shuffle(candidates) # Randomize order of removal attempts
        
        for x in candidates:
            # Check if x is removable
            # 1. Check if x is blocked by S \ {x}
            # x is blocked if exists a in S \ {x} such that x+a in sums_{S \ {x}}
            # sums_{S \ {x}} = sums \ {x+a for a in S, a!=x} \ {2x}
            
            # Efficient check:
            # If x is not blocked by S \ {x}, it's not removable.
            # If x is blocked, check if all y not in S are blocked by S \ {x}.
            
            # Let's just run the full is_maximal check on S \ {x} if it looks promising?
            # Or use the precomputed blocker logic.
            
            # For simplicity and correctness first, let's try the direct check but optimized.
            # Since N is small (256) and |S| is small (~15), is_maximal is fast enough.
            # 256 * 15 * 15 ops ~ 57k.
            # If we try 15 candidates, that's ~850k ops.
            # With 200 trials, that's 170M ops. Might be too slow in Python.
            
            # Let's implement the smarter check.
            
            # Precompute for current S:
            # For each y in [1, N] \ S, find the set of a in S such that y+a in sums.
            # If this set is empty, S is not maximal (but we assume it is).
            # For x to be removable:
            # For each y in [1, N] \ S:
            #   There must exist a in Blockers(y) \ {x} such that y+a is in sums_{S \ {x}}.
            #   y+a is in sums_{S \ {x}} iff y+a is in sums AND y+a is not formed by x.
            #   y+a is formed by x if y+a = x+b (b in S, b!=x) or y+a = 2x.
            
            # This is getting complex. Let's stick to a simpler heuristic:
            # Try removing x. Check is_maximal.
            # To speed up, we can first check if x is "critical".
            # If x is the ONLY blocker for some y, it's critical.
            
            # Let's compute Blockers(y) for all y.
            blockers = {}
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                bys = []
                for a in S:
                    if (y + a) in sums:
                        bys.append(a)
                if not bys:
                    # Should not happen if S is maximal
                    continue
                blockers[y] = bys
            
            # Check if x is the unique blocker for any y
            critical = False
            for y, bys in blockers.items():
                if len(bys) == 1 and bys[0] == x:
                    critical = True
                    break
            
            if critical:
                continue
                
            # Also check if x is blocked by S \ {x}
            # x is blocked if exists a in S \ {x} such that x+a in sums_{S \ {x}}
            # Approximation: if x is blocked by S, is it blocked by S \ {x}?
            # Not necessarily.
            
            # Let's just try the removal and check is_maximal.
            # But we do it for the first few non-critical candidates.
            
            candidate_S = [s for s in S if s != x]
            if is_maximal(candidate_S, N):
                S = candidate_S
                improved = True
                break
                
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best = None
    best_size = N + 1
    
    # Baseline: Ascending greedy
    order = list(range(1, N + 1))
    S = greedy_sidon(order, N)
    S = reduce_maximal(S, N)
    if len(S) < best_size:
        best = S
        best_size = len(S)
        
    random.seed(42)
    
    # Try multiple random orderings
    for trial in range(500):
        if time.time() > deadline:
            break
            
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        S = greedy_sidon(order, N)
        
        # Reduce
        if time.time() < deadline - 5:
            S = reduce_maximal(S, N)
        
        if len(S) < best_size:
            best = S
            best_size = len(S)
            
    if best is None:
        best = greedy_sidon(list(range(1, N + 1)), N)
        
    return best
