def construct():
    import math
    
    # Base set B: [129, 256]
    B = set(range(129, 257))
    
    # Calculate cost C(a) for a in [2, 128]
    # C(a) = sum(1/b for b in B if a divides b)
    # Note: B is fixed initially.
    
    candidates = []
    for a in range(2, 129):
        cost = 0.0
        for b in B:
            if b % a == 0:
                cost += 1.0 / b
        gain = 1.0 / a
        net_gain = gain - cost
        candidates.append((a, net_gain))
        
    # Sort by net_gain descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    A = []
    for a, ng in candidates:
        if ng <= 0:
            continue # No point adding if net gain is negative
            
        # Check LCM constraint with existing A
        can_add = True
        for x in A:
            if math.lcm(a, x) <= 256:
                can_add = False
                break
                
        if can_add:
            A.append(a)
            
    # Construct final set
    # We need to remove from B any element that conflicts with A?
    # Wait, the condition is lcm(a, b) > 256.
    # If we add a to A, we must ensure no b in B has lcm(a, b) <= 256.
    # As derived, for b in [129, 256], lcm(a, b) <= 256 iff a divides b.
    # So we must remove from B any b such that a|b for any a in A.
    
    final_B = set(B)
    for a in A:
        to_remove = [b for b in final_B if b % a == 0]
        for b in to_remove:
            final_B.remove(b)
            
    return A + list(final_B)
