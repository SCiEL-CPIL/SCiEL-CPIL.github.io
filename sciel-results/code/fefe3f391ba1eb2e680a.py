import time

def construct():
    N = 50
    # Find a Sidon set
    S = []
    sums = set() # Sums of pairs (i,j) with i <= j
    
    for x in range(N):
        # Check if x can be added
        valid = True
        new_sums = set()
        
        # Double sum
        if 2*x in sums:
            # Check if it's from the same element? No, sums contains sums of existing elements.
            # If 2x is in sums, it means there exist i,j in S such that i+j=2x.
            # If i=j, then i=j=x, so x is already in S (impossible as we are adding new).
            # If i!=j, then collision.
            # Wait, if i=j, then i=x. But x is not in S yet.
            # So if 2x is in sums, it must be from i!=j.
            valid = False
        
        if valid:
            for s in S:
                s_sum = x + s
                if s_sum in sums:
                    valid = False
                    break
                new_sums.add(s_sum)
            
            if valid:
                S.append(x)
                sums.update(new_sums)
                sums.add(2*x)
                
    # S is the Sidon set
    h = [0.0] * N
    for idx in S:
        h[idx] = 1.0
        
    return h
