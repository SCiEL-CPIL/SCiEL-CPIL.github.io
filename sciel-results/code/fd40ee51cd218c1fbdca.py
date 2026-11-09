def _try_place_smart(m, assign, max_iters=50):
    # 1. Direct
    for r in range(m):
        if _compat(assign, m, r):
            new_assign = dict(assign)
            new_assign[m] = r
            return True, new_assign
            
    # 2. Move one
    # Sort existing moduli by gcd(m, m2) descending
    candidates = sorted(assign.keys(), key=lambda x: math.gcd(m, x), reverse=True)
    
    for _ in range(max_iters):
        if not candidates: break
        # Pick a candidate
        p = random.choice(candidates[:5]) # Pick from top 5
        old_rp = assign[p]
        
        # Try to find a new residue for p that is compatible with others
        # And allows m to be placed?
        # This is hard to check directly.
        # Instead:
        # 1. Move p to a random valid residue (compatible with others).
        # 2. Try to place m.
        
        new_rp = -1
        for _ in range(10): # Try a few random residues for p
            cand_rp = random.randint(0, p - 1)
            if cand_rp == old_rp: continue
            # Check if cand_rp is compatible with all other moduli (except m)
            ok = True
            for m2, r2 in assign.items():
                if m2 == p: continue
                if cand_rp % math.gcd(p, m2) == r2 % math.gcd(p, m2):
                    ok = False
                    break
            if ok:
                new_rp = cand_rp
                break
                
        if new_rp == -1: continue
        
        assign[p] = new_rp
        
        # Try to place m
        placed = False
        for r in range(m):
            if _compat(assign, m, r):
                assign[m] = r
                placed = True
                break
                
        if placed:
            return True, assign
        
        assign[p] = old_rp # Revert
        
    return False, assign
