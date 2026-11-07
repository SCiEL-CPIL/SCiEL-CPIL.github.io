def construct():
        n = 256
        S = set(range(2, n + 1))
        
        for a in range(2, n + 1):
            if a not in S:
                continue
            
            # Check if a divides b*c for any distinct b, c in S
            conflict = False
            # Optimize: only check b, c such that b < c to avoid duplicates?
            # But we just need to find ANY pair.
            # Iterate b in S
            for b in S:
                if b == a:
                    continue
                for c in S:
                    if c == a or c == b:
                        continue
                    if (b * c) % a == 0:
                        conflict = True
                        break
                if conflict:
                    break
            
            if conflict:
                S.remove(a)
                
        return list(S)
