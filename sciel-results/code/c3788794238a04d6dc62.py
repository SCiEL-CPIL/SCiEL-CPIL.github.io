import numpy as np

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Returns a square-difference-free set modulo 205.
    
    Strategy:
    Using CRT, Z_205 is isomorphic to Z_5 x Z_41.
    A difference d is a non-zero square mod 205 iff d mod 5 is a non-zero square mod 5
    AND d mod 41 is a non-zero square mod 41.
    
    Non-zero squares mod 5 are {1, 4}.
    This implies that elements with the same residue mod 5 never have a forbidden difference
    (since diff mod 5 would be 0).
    Also, elements with residues differing by 2 or 3 mod 5 never have a forbidden difference
    (since 2 and 3 are not squares mod 5).
    
    The "forbidden" connections only exist between residues differing by 1 or 4 mod 5.
    Specifically, between fiber u and u+1, and u and u-1.
    
    We can select ALL elements from fiber 0 (residue 0 mod 5) and ALL elements from fiber 2
    (residue 2 mod 5).
    - Within fiber 0: No conflicts (diff mod 5 is 0).
    - Within fiber 2: No conflicts (diff mod 5 is 0).
    - Between fiber 0 and 2: Diff mod 5 is 2 or 3. Neither is a square mod 5. So no conflicts.
    
    Any other fiber (1, 3, 4) would conflict with fiber 0 or 2 if non-empty, because:
    - Fiber 1 conflicts with 0 (diff 1/4) and 2 (diff 1/4).
    - Fiber 3 conflicts with 2 (diff 1/4) and 4 (diff 1/4).
    - Fiber 4 conflicts with 0 (diff 1/4) and 3 (diff 1/4).
    
    Since we take full fibers 0 and 2, fibers 1, 3, 4 must be empty to avoid conflicts.
    
    Size = 41 + 41 = 82.
    """
    m = 205
    
    # Select residues x in 0..204 such that x % 5 is in {0, 2}
    selected = []
    for x in range(m):
        if x % 5 == 0 or x % 5 == 2:
            selected.append(x)
            
    return np.array(selected, dtype=np.int64)
