import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    
    # Start with upper half [129, 256] - always valid since for a<b<=c in (n/2,n], b+c>2a
    S = set(range(129, n + 1))
    
    # Try to add numbers from 128 down to 1 using residue-based feasibility check
    for x in range(128, 0, -1):
        if time.time() > deadline:
            break
        
        # Build residue counts mod x
        residue_count = {}
        for v in S:
            r = v % x
            residue_count[r] = residue_count.get(r, 0) + 1
        
        can_add = True
        checked = set()
        for r in list(residue_count.keys()):
            if r in checked:
                continue
            target = (-r) % x
            if target == r:
                # b=c allowed: if 2r ≡ 0 (mod x), one element suffices
                if 2 * r % x == 0:
                    can_add = False
                    break
            else:
                # b<c: need elements with both residues r and target
                if target in residue_count:
                    can_add = False
                    break
            checked.add(r)
            checked.add(target)
        
        if can_add:
            S.add(x)
    
    return sorted(S)
