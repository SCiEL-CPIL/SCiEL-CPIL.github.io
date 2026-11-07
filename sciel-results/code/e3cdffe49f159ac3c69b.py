def construct():
    n = 128
    # S contains all even numbers from 2 to 128, plus selected odd numbers.
    # We greedily add odd numbers that do not create a forbidden triple.
    S = set(range(2, n + 1, 2))
    
    # Candidate odd numbers to consider
    for x in range(1, n + 1, 2):
        # Check if adding x creates a forbidden triple
        # A triple is (a, b, c) distinct such that a+b, a+c, b+c are all in S.
        # Since x is new, any new forbidden triple must include x.
        # So we check for pairs (a, b) in S such that:
        # 1. a, b, x are distinct.
        # 2. a+b in S
        # 3. a+x in S
        # 4. b+x in S
        
        # Optimization:
        # If a+b is in S, and a+x is in S, and b+x is in S.
        # Note that if a and b are both even, a+b is even.
        # If a is even and b is odd (in S), a+b is odd.
        
        valid = True
        # Iterate over all pairs in current S
        # To be efficient, we can just check all a, b in S
        # But S is size ~64+16=80. 80^2 = 6400 checks per candidate.
        # 64 candidates * 6400 = 409,600 ops. Very fast.
        
        s_list = list(S)
        for i in range(len(s_list)):
            a = s_list[i]
            if a == x:
                continue
            for j in range(i + 1, len(s_list)):
                b = s_list[j]
                if b == x:
                    continue
                # a and b are distinct and not x
                # Check if a+b, a+x, b+x are in S
                if (a + b) in S and (a + x) in S and (b + x) in S:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.add(x)
            
    return [int(x) for x in sorted(S)]
