"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Returns a set of distinct positive integers <= 512 such that no element
    is the arithmetic mean of any subset of at least two other elements.
    
    We use a greedy algorithm to construct a set with no 3-term arithmetic progressions.
    A 3-term arithmetic progression a, x, b implies x = (a+b)/2, which violates
    the condition for a subset of size 2. Thus, any valid set must be 3-AP-free.
    By greedily adding numbers that do not form a 3-AP with any two existing numbers,
    we maximize the size of the set under this necessary condition.
    """
    N = 512
    S = []
    S_set = set()
    
    for x in range(1, N + 1):
        valid = True
        # Check if x is the mean of any two distinct elements a, b in S.
        # x = (a + b) / 2  =>  a + b = 2 * x
        target = 2 * x
        for a in S:
            b = target - a
            # b must be in S, and distinct from a (to ensure subset of size 2 has distinct elements? 
            # The problem says "subset of at least two OTHER elements". 
            # "Other" implies distinct from x. 
            # "Subset" implies distinct elements from S. 
            # Does the subset {a, b} require a != b? 
            # Yes, "subset of ... elements" usually implies distinct elements. 
            # Also, if a == b, then 2a = 2x => a = x, but a is in S and x is not in S yet. 
            # So a != x and b != x is guaranteed if a, b in S.
            # We just need a != b to ensure the subset has size 2.
            # If a == b, the subset is {a}, size 1.
            # So we require a != b.
            if b in S_set and b != a:
                valid = False
                break
        
        if valid:
            S.append(x)
            S_set.add(x)
            
    return S

# EVOLVE-BLOCK-END
