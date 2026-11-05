def construct():
    n = 256
    best_set = list(range(171, 257))
    best_size = len(best_set)
    
    # Try different starting intervals
    for K in range(100, 257):
        S = set(range(K, 257))
        # Precompute sums for initial S
        sums = set()
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i, len(s_list)):
                sums.add(s_list[i] + s_list[j])
                
        # Greedily add smaller elements
        for a in range(K-1, 0, -1):
            # Check if a divides any sum in 'sums'
            conflict = False
            # Multiples of a up to max sum (512)
            for m in range(2*a, 513, a):
                if m in sums:
                    conflict = True
                    break
            
            if not conflict:
                S.add(a)
                # Update sums
                # New pairs: (a, a) and (a, b) for b in S \ {a}
                # Since we are adding a, and S previously had elements > a,
                # the new pairs are (a, a) and (a, b) for b in old S.
                # Note: The loop above used S. We need to be careful.
                # Let's just recompute sums? No, too slow.
                # We can add new sums.
                sums.add(2*a)
                for b in S:
                    if b != a:
                        sums.add(a + b)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
