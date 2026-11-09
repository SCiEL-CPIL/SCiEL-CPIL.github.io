    def construct():
        # ... setup ...
        
        # Generate sums1
        sums1 = [(0, 0)]
        for i in range(len(w1)):
            wi = w1[i]
            sums1 += [(s + wi, m | (1 << i)) for s, m in sums1]
        sums1 = [(s, m) for s, m in sums1 if s < L]
        sums1.sort(key=lambda x: x[0])
        
        # Generate sums2
        sums2 = [(0, 0)]
        for i in range(len(w2)):
            wi = w2[i]
            sums2 += [(s + wi, m | (1 << i)) for s, m in sums2]
        sums2 = [(s, m) for s, m in sums2 if s < L]
        sums2.sort(key=lambda x: x[0])
        
        # Find best pair
        # sums1 and sums2 are sorted by sum
        # We want max s1 + s2 < L
        
        s1_vals = [s for s, m in sums1]
        s2_vals = [s for s, m in sums2]
        
        best_deficit = L
        best_s1_idx = -1
        best_s2_idx = -1
        
        # Iterate s1, binary search s2
        for i, s1 in enumerate(s1_vals):
            # Find largest s2 such that s2 < L - s1
            target = L - s1
            # bisect_left gives first >= target
            # We want largest < target
            idx2 = bisect_left(s2_vals, target)
            if idx2 > 0:
                # Check idx2 - 1
                s2 = s2_vals[idx2 - 1]
                total = s1 + s2
                if total < L:
                    deficit = L - total
                    if deficit < best_deficit:
                        best_deficit = deficit
                        best_s1_idx = i
                        best_s2_idx = idx2 - 1
        
        # Reconstruct
        if best_s1_idx == -1:
             return [0]
             
        m1 = sums1[best_s1_idx][1]
        m2 = sums2[best_s2_idx][1]
        
        # ... map masks to indices ...
    