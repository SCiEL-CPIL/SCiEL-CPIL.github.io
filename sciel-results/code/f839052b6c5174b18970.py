def construct():
    import math
    from itertools import combinations

    best_set = []
    best_sum = 0.0

    # Candidate anchors: small even numbers
    anchors = list(range(2, 20, 2)) + [22, 26, 30, 34]

    for anchor in anchors:
        S = [anchor]
        # Add all compatible numbers in descending order of 1/x (ascending x)
        # To maximize sum, we should include as many small numbers as possible.
        # Greedy: try to add x from 2 to 256 if compatible.
        # But we must ensure pairwise compatibility.
        
        current_S = [anchor]
        
        for x in range(2, 257):
            if x == anchor:
                continue
            # Check if x is compatible with all in current_S
            compatible = True
            for y in current_S:
                if math.lcm(x, y) <= 256:
                    compatible = False
                    break
            if compatible:
                current_S.append(x)
                
        s_val = sum(1.0/x for x in current_S)
        if s_val > best_sum:
            best_sum = s_val
            best_set = current_S

    return best_set
