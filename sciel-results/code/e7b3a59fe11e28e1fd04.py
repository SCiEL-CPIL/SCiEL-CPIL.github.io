def construct():
    n = 12
    sums_bitset = 1  # Bit 0 set, representing sum 0
    result = []
    
    for _ in range(n):
        v = 1
        while True:
            # Check if adding v creates any duplicate sums
            # new_sums is the set of sums obtained by adding v to all existing sums
            new_sums = sums_bitset << v
            # If there is any overlap between existing sums and new sums, it's invalid
            if (sums_bitset & new_sums) == 0:
                sums_bitset |= new_sums
                result.append(v)
                break
            v += 1
            
    return result
