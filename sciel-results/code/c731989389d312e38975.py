def construct():
    n = 12
    # Greedy construction
    # a[0] must be 1
    a = [1]
    sums = {0, 1}
    
    for k in range(1, n):
        # Try smallest possible next integer
        # It must be distinct from existing
        # And not create duplicate sums
        candidate = a[-1] + 1
        while True:
            new_sums = set()
            valid = True
            # Check if candidate creates duplicates
            # New sums are s + candidate for s in sums
            # They must be disjoint from sums
            # And candidate itself must not be in sums? No, candidate is new element.
            # But if candidate is in sums, then candidate = sum(subset) implies dup?
            # Yes, if candidate = sum(S'), then {candidate} and S' are two subsets with same sum.
            # So candidate must not be in sums.
            if candidate in sums:
                candidate += 1
                continue
            
            # Check disjointness
            # We can just generate new sums and check intersection
            # For efficiency, we can just check if (sums + candidate) intersects sums
            # sums + candidate = {s + candidate for s in sums}
            
            # Optimization: if candidate > max(sums), then sums + candidate > max(sums) >= min(sums)
            # So they are disjoint.
            if candidate > max(sums):
                # Valid
                a.append(candidate)
                sums.update(s + candidate for s in sums)
                break
            else:
                # Check intersection
                # Create set of new sums
                new_sums_set = {s + candidate for s in sums}
                if new_sums_set & sums:
                    candidate += 1
                else:
                    a.append(candidate)
                    sums.update(new_sums_set)
                    break
            
            # Safety break to avoid infinite loop if logic fails
            if candidate > 1048576:
                break
                
    # If greedy failed or took too long, fallback to powers of 2
    if len(a) < n:
        return [1 << i for i in range(n)]
        
    return a
