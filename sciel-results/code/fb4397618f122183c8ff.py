import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin
    
    n = 512
    S = set()
    
    # 1. Initial Construction: Base-3 Stanley Sequence
    for x in range(1, n + 1):
        if _is_base3_ok(x):
            S.add(x)
            
    # 2. Greedy Fill
    _greedy_fill(S, n)
    
    # 3. Local Search / Swap
    while time.time() < deadline:
        # Try to add any element
        added = False
        for x in range(1, n + 1):
            if x not in S and _is_valid_add(x, S, n):
                S.add(x)
                added = True
                # After adding, we might need to re-check others? 
                # No, adding only removes validity for others, doesn't create new valid slots for existing non-members.
                # But it might make *other* non-members invalid? No, we only add if valid.
                # So we can just continue the loop.
        
        if not added:
            # Try a swap
            # Pick a random x not in S
            candidates = [x for x in range(1, n + 1) if x not in S]
            if not candidates:
                break # Full set, impossible for 4-AP free unless n is small, but just in case.
            
            x = random.choice(candidates)
            
            # Find conflicts
            # We need to find if removing some y in S makes x valid.
            # x is invalid because there exists d such that 3 other elements are in S.
            # Let's find all y that are part of a 4-AP with x.
            blockers = set()
            for d in range(1, n + 1):
                # Check patterns
                # 1. x, x+d, x+2d, x+3d
                if x + 3*d <= n:
                    if x+d in S and x+2*d in S and x+3*d in S:
                        blockers.update([x+d, x+2*d, x+3*d])
                # 2. x-d, x, x+d, x+2d
                if x-2*d >= 1 and x+2*d <= n:
                    if x-d in S and x+d in S and x+2*d in S:
                        blockers.update([x-d, x+d, x+2*d])
                # 3. x-2d, x-d, x, x+d
                if x-2*d >= 1 and x+d <= n:
                    if x-2*d in S and x-d in S and x+d in S:
                        blockers.update([x-2*d, x-d, x+d])
                # 4. x-3d, x-2d, x-d, x
                if x-3*d >= 1:
                    if x-3*d in S and x-2*d in S and x-d in S:
                        blockers.update([x-3*d, x-2*d, x-d])
                
                # Optimization: break early if d is too large?
                # Max d for x is roughly n/3.
                if d > n // 3:
                    break

            if not blockers:
                # Should not happen if _is_valid_add returned False, but just in case
                continue
                
            # Try removing a random blocker
            y = random.choice(list(blockers))
            
            # Check if S - {y} + {x} is valid
            # We know x is valid if y is removed?
            # Not necessarily. There might be *multiple* conflicting APs.
            # We must check if removing y resolves ALL conflicts for x.
            
            # Efficient check:
            # Temporarily remove y
            S.remove(y)
            if _is_valid_add(x, S, n):
                S.add(x)
                # Success! We swapped y for x. Size unchanged.
                # Now try to fill again
                _greedy_fill(S, n)
            else:
                # Restore y
                S.add(y)
                # We didn't succeed in swapping.
                
    return sorted(S)

def _is_base3_ok(x):
    while x > 0:
        if x % 3 == 2:
            return False
        x //= 3
    return True

def _is_valid_add(x, S, n):
    # Check if adding x creates a 4-AP
    for d in range(1, n // 3 + 1):
        # Pattern 1: x, x+d, x+2d, x+3d
        if x + 3*d <= n:
            if x+d in S and x+2*d in S and x+3*d in S:
                return False
        # Pattern 2: x-d, x, x+d, x+2d
        if x-2*d >= 1 and x+2*d <= n:
            if x-d in S and x+d in S and x+2*d in S:
                return False
        # Pattern 3: x-2d, x-d, x, x+d
        if x-2*d >= 1 and x+d <= n:
            if x-2*d in S and x-d in S and x+d in S:
                return False
        # Pattern 4: x-3d, x-2d, x-d, x
        if x-3*d >= 1:
            if x-3*d in S and x-2*d in S and x-d in S:
                return False
    return True

def _greedy_fill(S, n):
    changed = True
    while changed:
        changed = False
        for x in range(1, n + 1):
            if x not in S and _is_valid_add(x, S, n):
                S.add(x)
                changed = True
