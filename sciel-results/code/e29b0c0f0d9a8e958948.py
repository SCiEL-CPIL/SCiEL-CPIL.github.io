def construct():
    # Base set S
    S = list(range(1, 11))
    # Scale factor c
    c = 20
    # Construct A = S U cS
    A = S + [c * x for x in S]
    
    # Ensure distinct and sorted (though not strictly required for return, good for consistency)
    A = sorted(set(A))
    
    # Verify constraints
    if len(A) != 20:
        # Fallback to 1..20 if logic fails (it shouldn't)
        return list(range(1, 21))
    if max(A) > 10000:
        return list(range(1, 21))
        
    return A
