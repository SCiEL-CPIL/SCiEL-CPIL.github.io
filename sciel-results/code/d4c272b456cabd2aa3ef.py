def construct():
    p = 1009
    # root_map[r] = x if x*x % p == r, else 0
    root_map = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            root_map[r] = x
            
    # Search for the smallest s
    # s can be from 1 to 1004
    # We need s, s+1, s+2, s+3, s+4 to be quadratic residues
    # Note: s+4 can be up to 1008, which is < p, so no wraparound issues for the indices in root_map
    # The problem states "wraparound are excluded", which aligns with s <= 1004.
    
    for s in range(1, 1005):
        # Check s, s+1, s+2, s+3, s+4
        if (root_map[s] != 0 and 
            root_map[s+1] != 0 and 
            root_map[s+2] != 0 and 
            root_map[s+3] != 0 and 
            root_map[s+4] != 0):
            
            roots = [root_map[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
            
    # This should never happen for p=1009, but just in case
    return {'start': 1, 'roots': [0,0,0,0,0]}
