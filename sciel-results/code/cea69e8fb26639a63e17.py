def construct():
    p = 1009
    # 1. Construct the set of non-zero quadratic residues and a map to roots
    is_res = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_res[r] = True
            if roots[r] == 0:
                roots[r] = x
    
    # 2. Get sorted list of residues
    # We can just iterate 1 to p-1, but the directive says "sorted residue list".
    # Since is_res is a boolean array, we can extract the list.
    residues = [i for i in range(1, p) if is_res[i]]
    
    # 3. Sliding window / search for smallest s
    # We want the smallest s such that s, s+1, s+2, s+3, s+4 are all in is_res.
    # Since residues is sorted, we can check each r in residues.
    # If r is a valid start, it's the smallest because residues is sorted.
    
    min_s = -1
    for r in residues:
        # Check if r, r+1, r+2, r+3, r+4 are all residues
        # Note: r+4 must be < p
        if r + 4 < p:
            if is_res[r] and is_res[r+1] and is_res[r+2] and is_res[r+3] and is_res[r+4]:
                min_s = r
                break
        
    if min_s == -1:
        raise RuntimeError("No run of 5 consecutive quadratic residues found")
        
    # 4. Construct the roots
    roots_list = [roots[min_s + i] for i in range(5)]
    
    return {'start': min_s, 'roots': roots_list}
