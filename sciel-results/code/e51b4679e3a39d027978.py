def construct():
    N = 128
    # colors[i] will store the color index (0..127)
    # We use a bitmask 'forbidden' to track invalid colors for the current position
    colors = [0] * N
    
    # Position 0 has no constraints
    colors[0] = 0
    
    for i in range(1, N):
        forbidden = 0
        
        # Constraint 1: 4-term progressions ending at i
        # (i-3d, i-2d, i-d, i)
        # We need |{c_i, c_{i-d}, c_{i-2d}, c_{i-3d}}| >= 3
        # This fails if:
        # 1. All 4 are same (impossible if we prevent 3-same in prev steps, but check anyway)
        # 2. Exactly 2 distinct colors in the 4 terms.
        
        # If we ensure no 3 consecutive terms in any AP are same, then for the 4-term AP:
        # If prev 3 are all same (X,X,X), we are in a dead end (handled by preventing 3-same).
        # If prev 3 are (X,X,Y), then c_i cannot be X or Y. (Set becomes {X,Y} if c_i in {X,Y}, size 2. If c_i=Z, size 3).
        # If prev 3 are (X,Y,Z), c_i can be anything.
        
        # So, for each d:
        # Let A=c[i-d], B=c[i-2d], C=c[i-3d]
        # If A==B==C: Dead end (should not happen).
        # If (A==B and B!=C) or (A==C and C!=B) or (B==C and C!=A):
        #    The set {A,B,C} has size 2. Let the values be v1, v2.
        #    Forbid v1 and v2.
        # If A,B,C all distinct: No restriction from this d.
        
        max_d_4 = i // 3
        for d in range(1, max_d_4 + 1):
            a = colors[i - d]
            b = colors[i - 2 * d]
            c = colors[i - 3 * d]
            
            if a == b and b == c:
                # This state should be prevented by the 3-term check.
                # If it occurs, we have a contradiction.
                # We'll mark all as forbidden to trigger a fallback or error, 
                # but logically it shouldn't happen if the 3-term check is correct.
                forbidden = (1 << N) - 1
                break
            
            # Check for exactly 2 distinct values
            # This happens if any pair is equal, but not all three.
            if a == b:
                if a != c:
                    forbidden |= (1 << a)
                    forbidden |= (1 << c)
            elif a == c:
                if a != b:
                    forbidden |= (1 << a)
                    forbidden |= (1 << b)
            elif b == c:
                if b != a:
                    forbidden |= (1 << b)
                    forbidden |= (1 << a)
            # else: all distinct, no restriction
            
        # Constraint 2: Prevent forming 3 same colors in a row in any AP
        # This is a preventive constraint.
        # For any d such that i-2d >= 0:
        # If c[i-d] == c[i-2d] == X, then c[i] cannot be X.
        # Why? Because then c[i-2d], c[i-d], c[i] would be X,X,X.
        # If this happens, the next position i+d (if it exists and completes an AP) 
        # or the current position if it's part of a 4-AP starting earlier might be stuck.
        # Actually, if c[i-2d]==c[i-d]==c[i]==X, then consider the 4-AP (i-3d, i-2d, i-d, i).
        # If i-3d exists, we have X,X,X,?. The set is {X, ?}. Size <= 2. Violation.
        # So yes, we must forbid X.
        
        max_d_3 = i // 2
        for d in range(1, max_d_3 + 1):
            a = colors[i - d]
            b = colors[i - 2 * d]
            if a == b:
                forbidden |= (1 << a)
        
        # Select the smallest available color
        # Iterate through 0..127
        # We can do this by finding the lowest 0 bit in 'forbidden'
        # But 'forbidden' might have bits set beyond 127? No, we only set bits 0..127.
        # We can use a loop or a bit trick.
        # Loop is simple and fast enough for 128 iterations.
        
        for c in range(N):
            if not (forbidden & (1 << c)):
                colors[i] = c
                break
        else:
            # Fallback if no color found (should not happen)
            colors[i] = 0
            
    return colors
