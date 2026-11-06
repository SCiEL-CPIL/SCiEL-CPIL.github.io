    for M in range(7, 2000):
        if time.time() - start_time > timeout:
            break
            
        # Precompute suffix sums for this M
        # suffix[i] = sum(1/k for k in i..M)
        # We only need i from 1 to M.
        # But we start from 7.
        suffix = [Fraction(0, 1)] * (M + 2)
        # suffix[M+1] = 0
        for i in range(M, 0, -1):
            suffix[i] = suffix[i+1] + Fraction(1, i)
            
        # Quick check: suffix[7] < target?
        if suffix[7] < target:
            continue
            
        def dfs(prev, rem, path):
            nonlocal start_time, timeout
            if time.time() - start_time > timeout:
                return "TIMEOUT"
            
            if rem == 0:
                return path[:]
            if rem < 0:
                return None
            
            d_min = prev + 1
            d_bound = (rem.denominator + rem.numerator - 1) // rem.numerator
            if d_bound > d_min:
                d_min = d_bound
            
            if d_min > M:
                return None
                
            # Check if suffix[d_min] < rem
            if suffix[d_min] < rem:
                return None
                
            for d in range(d_min, M + 1):
                new_rem = rem - Fraction(1, d)
                path.append(d)
                res = dfs(d, new_rem, path)
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res is not None:
                    return res
                path.pop()
            return None

        res = dfs(0, target, [])
        if res == "TIMEOUT":
            break
        if res is not None:
            return res
