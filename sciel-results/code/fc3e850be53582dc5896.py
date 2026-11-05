import time
import random

def construct():
    deadline = time.time() + 85  # safety margin
    
    def find_conflict(a, S):
        """
        Check if adding 'a' to valid set 'S' creates a conflict.
        Returns (b, c) if conflict exists, else None.
        S must be a set.
        """
        # Case 1: a is the smallest (x)
        # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
        a2 = a * a
        for b in S:
            if b <= a:
                continue
            d = b - a
            if a2 % d != 0:
                continue
            c = a + a2 // d
            if c > a and c != b and c in S:
                return (b, c)
        
        # Case 2: a is the middle (y)
        # 1/x = 1/a + 1/c => c = a*x / (a-x)
        # x < a < c
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            c = num // denom
            if c > a and c in S:
                return (x, c)
        
        # Case 3: a is the largest (z)
        # 1/x = 1/y + 1/a => y = a*x / (a-x)
        # x < y < a
        for x in S:
            if x >= a:
                continue
            denom = a - x
            num = a * x
            if num % denom != 0:
                continue
            y = num // denom
            if x < y < a and y in S:
                return (x, y)
        
        return None

    def is_valid(S):
        # Optional: verify validity for debugging, but we trust the incremental construction
        pass

    # Start with a known valid set: all numbers > 90
    S = set(range(91, 181))
    
    # Greedily add smaller numbers
    for a in range(1, 91):
        if time.time() > deadline:
            break
        if find_conflict(a, S) is None:
            S.add(a)
    
    best_S = set(S)
    best_size = len(S)
    
    all_nums = set(range(1, 181))
    
    random.seed(42)
    
    # Local search with targeted swaps
    # We track a "stuck" counter to potentially restart or change strategy
    stuck_count = 0
    max_stuck = 1000
    
    while time.time() < deadline:
        excluded = all_nums - S
        if not excluded:
            break
        
        # Try to add a random excluded element
        a = random.choice(list(excluded))
        conflict = find_conflict(a, S)
        
        if conflict is None:
            S.add(a)
            if len(S) > best_size:
                best_S = set(S)
                best_size = len(S)
            stuck_count = 0
        else:
            b, c = conflict
            # Try to remove b and add a
            S.remove(b)
            # Check if a can be added now
            # Note: S is modified. We need to check against the new S.
            # However, removing b might break other constraints? No, removing elements never creates new conflicts.
            # So we just need to check if a conflicts with the *remaining* elements.
            # The conflict was (b,c) or (c,b) or something involving a.
            # If we remove b, does a still conflict with something else?
            # It's possible a has multiple conflicts.
            # So we must run find_conflict again.
            
            new_conflict = find_conflict(a, S)
            if new_conflict is None:
                S.add(a)
                if len(S) > best_size:
                    best_S = set(S)
                    best_size = len(S)
                stuck_count = 0
            else:
                # Revert b
                S.add(b)
                # Try to remove c and add a
                S.remove(c)
                new_conflict2 = find_conflict(a, S)
                if new_conflict2 is None:
                    S.add(a)
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                    stuck_count = 0
                else:
                    # Revert c
                    S.add(c)
                    # If targeted swap failed, try a random swap as fallback
                    if S:
                        b_rand = random.choice(list(S))
                        S.remove(b_rand)
                        new_conflict3 = find_conflict(a, S)
                        if new_conflict3 is None:
                            S.add(a)
                            if len(S) > best_size:
                                best_S = set(S)
                                best_size = len(S)
                            stuck_count = 0
                        else:
                            S.add(b_rand)
                            stuck_count += 1
                            
        # If stuck for too long, we might want to perturb the solution
        # But simple restarts are risky if we lose the best state.
        # We keep best_S, so we can restore it.
        if stuck_count > max_stuck:
            # Restore best state and randomize slightly
            S = set(best_S)
            # Remove a few random elements to create space
            if S:
                for _ in range(5):
                    if not S: break
                    S.remove(random.choice(list(S)))
            stuck_count = 0

    return sorted(best_S)
