import time
from collections import defaultdict

def is_valid_set(S):
    """Check if no element in S divides the sum of any 3 distinct other elements."""
    lst = list(S)
    n = len(lst)
    if n < 4:
        return True
    
    for idx_a, a in enumerate(lst):
        others = [x for i, x in enumerate(lst) if i != idx_a]
        m = len(others)
        if m < 3:
            continue
        
        # For each pair (i,j) in others, check if there exists k != i,j
        # such that (others[i] + others[j] + others[k]) % a == 0
        # i.e., others[k] % a == (-(others[i] + others[j])) % a
        
        # Build a frequency map of residues for others
        # But we need to handle the case where k could be i or j, so we need to be careful
        # Actually, since we're checking distinct indices, let's use a different approach
        
        # For fixed a, count how many elements have each residue mod a
        residue_count = defaultdict(int)
        for x in others:
            residue_count[x % a] += 1
        
        # Now for each pair (i, j), we need to check if there's a k with the right residue
        # But k must be distinct from i and j
        # The residue we need is r = (-(others[i] + others[j])) % a
        # We need count[r] >= 1, but if others[i] % a == r or others[j] % a == r,
        # we need to make sure there's still an available element with that residue
        
        found = False
        for i in range(m):
            oi = others[i]
            ri = oi % a
            for j in range(i + 1, m):
                oj = others[j]
                rj = oj % a
                target_r = (-(oi + oj)) % a
                
                # Count how many elements have residue target_r
                cnt = residue_count.get(target_r, 0)
                
                # Subtract if oi or oj has this residue (they can't be k)
                if ri == target_r:
                    cnt -= 1
                if rj == target_r:
                    cnt -= 1
                
                if cnt > 0:
                    found = True
                    break
            if found:
                break
        
        if found:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # Strategy: try numbers from large to small
    # Larger numbers are less likely to divide sums of 3 others
    # (sum of 3 others is at most 3*192=576, so numbers > 576/3=192 can't divide any sum,
    # but our numbers are <= 192, so all could potentially divide sums)
    
    # Actually, let's think more carefully:
    # If a is large, say a > 576/2 = 288, then 2a > 576, so the only multiple is a itself.
    # But sum of 3 distinct others is at least 1+2+3=6 and at most 190+191+192=573.
    # So if a > 573, no sum can be a multiple of a. But a <= 192, so this doesn't help directly.
    
    # Better approach: prioritize numbers that are less likely to be divisors.
    # Numbers with few divisors in the range [6, 573] are safer.
    
    # Let's just try from 192 down to 1, and use the fast validity check
    
    selected = []
    
    for x in range(192, 0, -1):
        if len(selected) == 48:
            break
        if time.time() > deadline:
            break
        
        # Quick check: if x is already in selected, skip (shouldn't happen)
        if x in selected:
            continue
        
        candidate_set = list(selected) + [x]
        
        if is_valid_set(candidate_set):
            selected.append(x)
    
    return selected
