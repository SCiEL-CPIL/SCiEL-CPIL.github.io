def construct():
    # Generate numbers with no digit 2 in base 3
    # These are 3-AP free
    # We need 12 such numbers in [0, 128]
    candidates = []
    for i in range(129):
        # Check if i has no digit 2 in base 3
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            candidates.append(i)
        if len(candidates) == 12:
            break
            
    return candidates
