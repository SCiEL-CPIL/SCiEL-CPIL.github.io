def construct():
    from math import gcd
    
    # Greedy: try to build the largest set
    # Start with all numbers, remove incompatible ones
    # Or: build up greedily
    
    n = 512
    # Try greedy: sort by some criterion, add if compatible
    # Let's try adding in order of increasing "difficulty"
    
    # Actually, let's try: start with {1}, then greedily add numbers
    # that are compatible with all current members
    
    selected = [1]
    for x in range(2, n + 1):
        compatible = True
        for y in selected:
            from math import lcm
            if lcm(x, y) > n:
                compatible = False
                break
        if compatible:
            selected.append(x)
    
    return selected
