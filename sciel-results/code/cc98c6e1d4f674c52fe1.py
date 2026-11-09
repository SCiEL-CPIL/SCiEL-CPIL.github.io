import time

def construct():
    current_set = []
    sums_set = set()
    
    # Pre-calculate or just check on the fly.
    # n = 4096
    for x in range(1, 4097):
        # Check if x can be added
        # We need to check all triples involving x
        # Types:
        # 1. (x, x, x) -> sum 3x
        # 2. (x, x, y) -> sum 2x + y for y in current_set
        # 3. (x, y, z) -> sum x + y + z for y, z in current_set, y <= z
        
        # Optimization: We can break early if any collision is found.
        
        valid = True
        
        # Check 3x
        if 3*x in sums_set:
            valid = False
        
        if valid:
            # Check 2x + y
            for y in current_set:
                if 2*x + y in sums_set:
                    valid = False
                    break
        
        if valid:
            # Check x + y + z
            k = len(current_set)
            for i in range(k):
                y = current_set[i]
                for j in range(i, k):
                    z = current_set[j]
                    if x + y + z in sums_set:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            current_set.append(x)
            # Update sums_set
            sums_set.add(3*x)
            for y in current_set[:-1]: # All previous elements
                sums_set.add(2*x + y)
            k = len(current_set) - 1
            for i in range(k):
                y = current_set[i]
                for j in range(i, k):
                    z = current_set[j]
                    sums_set.add(x + y + z)
                    
    return current_set
