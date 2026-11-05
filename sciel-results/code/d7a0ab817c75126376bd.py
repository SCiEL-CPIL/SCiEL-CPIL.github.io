def construct():
    n = 12
    selected = []
    current_sums = {0}
    
    for _ in range(n):
        candidate = 1
        while True:
            # Skip if already used
            if candidate in selected:
                candidate += 1
                continue
            
            # Efficiently check distinctness
            # New sums are current_sums union {s + candidate for s in current_sums}
            # We need len(new_sums) == 2 * len(current_sums)
            # This is equivalent to checking if (s + candidate) is not in current_sums for any s
            # AND (s + candidate) are distinct among themselves (which they are if candidate > 0)
            # Actually, if s1 + c == s2 + c, then s1 == s2. So new sums are distinct internally.
            # We just need to check if any new sum collides with an old sum.
            
            is_valid = True
            for s in current_sums:
                if (s + candidate) in current_sums:
                    is_valid = False
                    break
            
            if is_valid:
                selected.append(candidate)
                # Update current_sums
                # new_sums = current_sums | {s + candidate for s in current_sums}
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + candidate)
                current_sums = new_sums
                break
            else:
                candidate += 1
                
    return selected
