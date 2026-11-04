def construct(n):
    if n <= 1:
        return "1" if n == 1 else ""
    
    # S[1] is [1]
    S = [1]
    
    for m in range(2, n + 1):
        # Get the order of permutations of length m-1 from S
        # We scan S and collect permutations of length m-1
        # We only care about the first time each permutation appears
        # to define the sequence P_1, ..., P_{(m-1)!}
        
        seen = set()
        perms_order = []
        
        # S is a list of integers
        # We need to check every window of length m-1
        for i in range(len(S) - (m - 1) + 1):
            window = tuple(S[i:i + m - 1])
            if window not in seen:
                seen.add(window)
                perms_order.append(window)
                
        # Now construct S_m
        # We iterate through perms_order
        # For each perm P, we form block P + [m] + P
        # And overlap with current S_m
        
        new_S = []
        
        for P in perms_order:
            block = list(P) + [m] + list(P)
            
            # Find maximum overlap with the end of new_S
            # The overlap k is the largest k such that new_S[-k:] == block[:k]
            # k can be at most len(block) - 1, and at most len(new_S)
            # Also, the overlap must be a valid suffix/prefix match.
            
            max_possible_overlap = min(len(new_S), len(block) - 1)
            overlap = 0
            # Check from largest possible overlap down to 0
            # Note: We don't need to check all k, just the valid ones.
            # But for small n, linear check is fine.
            for k in range(max_possible_overlap, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if new_S[-k:] == block[:k]:
                    overlap = k
                    break
            
            # Append the rest of the block
            new_S.extend(block[overlap:])
            
        S = new_S
        
    return "".join(map(str, S))
