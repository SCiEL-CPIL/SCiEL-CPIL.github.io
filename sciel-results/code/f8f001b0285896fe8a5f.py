import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 80  # 80s to be safe
    
    S = set()
    for a in range(n, 0, -1):
        valid = True
        lst = list(S)
        m = len(lst)
        for i in range(m):
            b = lst[i]
            for j in range(i, m):
                c = lst[j]
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.add(a)
    
    # Local search: try to add elements by removing conflicts
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_size = len(S)
        # Try adding each element not in S
        for a in range(1, n + 1):
            if a in S:
                continue
            # Check conflicts
            conflicts = set()
            for b in S:
                if b <= a:
                    continue
                for c in S:
                    if c < b:
                        continue
                    if (b + c) % a == 0:
                        conflicts.add(b)
                        conflicts.add(c)
            # Also need to check pairs where a is the smaller one
            # For a < b <= c: a must not divide b+c
            # When we add a, we need to check all b,c in S with b > a, c >= b
            # That's what we did above.
            # But also: for existing elements x,y in S where x < y, and a is between them or below
            # Actually the constraint is only about triples (a,b,c) with a<b<=c
            # When adding new element a, the new triples are:
            # 1. a < b <= c where b,c in old S: check a ∤ (b+c)
            # 2. b < a <= c where b in old S, c in old S, c >= a: check b ∤ (a+c)
            # 3. b < c < a where b,c in old S: check b ∤ (c+a) -- wait, no
            # Actually, the constraint is: for EVERY triple (x,y,z) in S with x<y<=z, x ∤ (y+z)
            # When we add a new element a, new triples involving a are:
            # - a as smallest: a < y <= z, y,z in old S
            # - a as middle: y < a <= z, y in old S, z in old S, z >= a
            # - a as largest: y < z = a, y in old S, a in new S (but a was not in old S)
            #   Wait, z = a means the triple is (y, z1, a) where y < z1 <= a
            #   But z1 must be in old S and < a (since a not in old S)
            #   So: y < z1 < a, check y ∤ (z1 + a)
            
            # Let me redo this more carefully
            conflicts.clear()
            for b in S:
                for c in S:
                    if c < b:
                        continue
                    # Check if adding a creates a violation
                    # Triple (a, b, c) with a < b <= c: a ∤ (b+c)
                    if a < b:
                        if (b + c) % a == 0:
                            conflicts.add(b)
                            conflicts.add(c)
                    # Triple (b, a, c) with b < a <= c: b ∤ (a+c)
                    elif b < a <= c:
                        if (a + c) % b == 0:
                            conflicts.add(b)
                            conflicts.add(c)
                    # Triple (b, c, a) with b < c < a: b ∤ (c+a)
                    elif b < c < a:
                        if (c + a) % b == 0:
                            conflicts.add(b)
                            conflicts.add(c)
            
            if len(conflicts) < 1:  # Can add without removing
                S.add(a)
                improved = True
            elif len(conflicts) < len(S) * 0.3:  # Maybe worth removing
                # Try removing conflicts and adding a
                new_S = S - conflicts
                new_S.add(a)
                if len(new_S) > len(S):
                    S = new_S
                    improved = True
        # Also try removing elements to allow adding others
        # ... this gets complex
    
    return list(S)
