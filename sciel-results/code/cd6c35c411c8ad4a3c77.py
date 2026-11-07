    def can_add(x, S):
        # 1. x is a number
        for b in S:
            if b == x: continue
            s1 = x + b
            if s1 > 128: continue # s1 not in S
            if s1 in S:
                for c in S:
                    if c == x or c == b: continue
                    s2 = x + c
                    if s2 > 128: continue
                    if s2 in S:
                        s3 = b + c
                        if s3 <= 128 and s3 in S:
                            return False
        
        # 2. x is a sum
        # a+b = x => b = x-a
        # a,c in S. a+c in S. b+c in S.
        # a,c in 1..128.
        # b in 1..128 => x-a in 1..128 => x-128 <= a <= x-1
        # Also a in S.
        
        for a in S:
            b = x - a
            if b < 1 or b > 128: continue
            if b == a: continue
            if b not in S: continue
            # a,b in S, a+b=x
            # Check c
            for c in S:
                if c == a or c == b: continue
                s1 = a + c
                if s1 > 128: continue
                if s1 in S:
                    s2 = b + c
                    if s2 <= 128 and s2 in S:
                        return False
                        
        return True
