def construct():
    N = 180
    S = set()
    
    # Iterate from N down to 1
    for a in range(N, 0, -1):
        conflict = False
        a_sq = a * a
        
        # Find divisors u of a_sq
        # u * v = a_sq
        # b = a + u, c = a + v
        # b, c must be in S
        # b != c => u != v => u != sqrt(a_sq) => u != a
        # Also b, c <= N
        
        # We only need to check u < v to avoid duplicate checks, 
        # or just all u.
        # Since b and c are symmetric in the equation, checking all u is fine.
        
        for u in range(1, int(a_sq**0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                
                b = a + u
                c = a + v
                
                if b > N or c > N:
                    continue
                
                # Check if both b and c are in S
                if b in S and c in S:
                    conflict = True
                    break
        
        if not conflict:
            S.add(a)
            
    return list(S)
