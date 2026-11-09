    def construct():
        import time
        start = time.time()
        deadline = 110 # 10s margin
        
        L = 10**11
        # Generate powerful numbers
        powerful = set()
        max_b = int(L ** (1/3)) + 2
        for b in range(1, max_b):
            b3 = b**3
            if b3 > L: break
            max_a = int((L // b3) ** 0.5) + 2
            for a in range(1, max_a):
                a2 = a*a
                p = a2 * b3
                if p > L: break
                powerful.add(p)
        
        P = sorted(powerful)
        M = len(P)
        
        # Count sums
        counts = {}
        for i in range(M):
            ai = P[i]
            # If ai + ai > 10**18, we could break, but L=10**11 so ai+ai < 2*10**11
            for j in range(i, M):
                s = ai + P[j]
                counts[s] = counts.get(s, 0) + 1
                
                if time.time() > deadline - 5:
                    # Safety break?
                    # If we break, we might miss the best N.
                    # But we have a partial count.
                    # Better to just let it run if it's fast enough.
                    pass
        
        # Find best N
        best_N = max(counts, key=counts.get)
        best_score = counts[best_N]
        
        # Retrieve pairs
        P_set = set(P)
        pairs = []
        for a in P:
            if a * 2 > best_N: break
            b = best_N - a
            if b in P_set:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}
    