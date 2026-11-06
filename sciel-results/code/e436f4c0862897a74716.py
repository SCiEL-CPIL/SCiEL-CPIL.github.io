def construct():
    n = 256
    S = []
    for x in range(n, 0, -1):
        can_add = True
        # Check if x divides b+c for any b, c in S with x < b <= c
        # Since S contains only numbers > x, all b, c in S satisfy x < b <= c.
        # We need to check all pairs (b, c) in S.
        # Optimization: If x is small, this is expensive. 
        # But n=256 is small. Max pairs in S is ~130^2/2 ~ 8000. 
        # Total checks ~ 256 * 8000 ~ 2M operations. Very fast in Python.
        
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                # b <= c is guaranteed by iteration? 
                # S is a list, order doesn't matter for the set, but we need b <= c for the condition a<b<=c.
                # Actually the condition is symmetric for b and c in the sum.
                # We just need to check if x divides (b+c).
                if (b + c) % x == 0:
                    can_add = False
                    break
            if not can_add:
                break
        
        if can_add:
            S.append(x)
            
    return S
