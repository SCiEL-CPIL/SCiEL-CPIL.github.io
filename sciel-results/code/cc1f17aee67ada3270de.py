import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    def is_valid(S):
        """Fast validity check using set operations."""
        elements = list(S)
        n = len(elements)
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                s_ab = a + b
                if s_ab not in S:
                    continue
                # Check for c: need a+c in S and b+c in S
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = elements[k]
                    if a + c in S and b + c in S:
                        return False
        return True

    def can_add(x, S):
        """Check if adding x to S keeps it valid. O(n^2) where n=len(S)."""
        elements = list(S)
        n = len(elements)
        # Check all triples involving x
        # Case 1: x is one of a, b, c
        # For each pair (a, b) from S, check if x+a, x+b, a+b all in S∪{x}
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                # Triple (x, a, b): need x+a, x+b, a+b in S∪{x}
                if x + a in S and x + b in S and a + b in S:
                    return False
        # Case 2: x is one of the sums
        # For each pair (a, b) from S, if a+b == x, need a+c and b+c in S for some c
        # Actually, x being a sum means we check if there exist a,b,c in S with a+b=x
        # and a+c in S and b+c in S
        for i in range(n):
            a = elements[i]
            for j in range(i+1, n):
                b = elements[j]
                if a + b == x:
                    # Need to check if there's c in S, c!=a,c!=b, with a+c in S and b+c in S
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        c = elements[k]
                        if a + c in S and b + c in S:
                            return False
        return True

    def initial_construct(M):
        """Construct initial set with large part {M,...,128} and small part from odds."""
        S = set(range(M, 129))
        K = M - 1
        # Use odds from 1 to K
        S.update(range(1, K+1, 2))
        # Verify validity
        if is_valid(S):
            return S
        # If invalid, try to repair by removing problematic elements
        # Simple approach: try different small subsets
        for small_step in [2, 3, 4]:
            S2 = set(range(M, 129))
            S2.update(range(1, K+1, small_step))
            if is_valid(S2):
                return S2
        return set(range(M, 129))

    # Try multiple initial constructions
    best_S = None
    best_size = 0
    
    for M in [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]:
        if time.time() > deadline - 5:
            break
        S_init = initial_construct(M)
        if len(S_init) > best_size:
            best_size = len(S_init)
            best_S = set(S_init)
    
    if best_S is None:
        best_S = initial_construct(64)
        best_size = len(best_S)
    
    S = set(best_S)
    
    # Local search with multiple restarts
    while time.time() < deadline - 2:
        # Try to add elements
        added_any = False
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            if len(S) >= 128:
                break
            candidates = [x for x in range(1, 129) if x not in S]
            if not candidates:
                break
            x = random.choice(candidates)
            if can_add(x, S):
                S.add(x)
                added_any = True
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        # Try swaps
        if time.time() > deadline - 2:
            break
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            if not S:
                break
            x = random.choice(list(S))
            candidates = [y for y in range(1, 129) if y not in S]
            if not candidates:
                break
            y = random.choice(candidates)
            new_S = (S - {x}) | {y}
            if is_valid(new_S):
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        # Try 2-for-1 swaps: remove 2, add 1
        if time.time() > deadline - 2:
            break
        for _ in range(10):
            if time.time() > deadline - 2:
                break
            if len(S) < 2:
                break
            x1, x2 = random.sample(list(S), 2)
            candidates = [y for y in range(1, 129) if y not in S]
            if not candidates:
                break
            y = random.choice(candidates)
            new_S = (S - {x1, x2}) | {y}
            if is_valid(new_S) and len(new_S) > len(S):
                S = new_S
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        
        # Occasionally restart with a different initial
        if random.random() < 0.05:
            if time.time() > deadline - 5:
                break
            M = random.randint(64, 80)
            S_new = initial_construct(M)
            if len(S_new) >= len(S) - 2:
                S = set(S_new)
    
    return list(best_S)
