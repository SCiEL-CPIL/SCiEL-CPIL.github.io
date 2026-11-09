def construct():
    # Precompute modular inverses for 1..10 modulo 101
    inverses = [pow(i, -1, 101) for i in range(1, 11)]
    
    # Initialize best solution for each residue
    # min_count[r] = minimum number of denominators to sum to r
    # best_subset[r] = the subset of denominators
    min_count = [11] * 101
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    for mask in range(1, 1 << 10):
        total_sum = 0
        count = 0
        terms = []
        
        for i in range(10):
            if mask & (1 << i):
                denom = i + 1
                total_sum += inverses[i]
                count += 1
                terms.append(denom)
        
        residue = total_sum % 101
        
        # Update if we found a smaller subset for this residue
        if count < min_count[residue]:
            min_count[residue] = count
            best_subset[residue] = terms
    
    # Construct the final list of 101 rows
    result = []
    for r in range(101):
        # All residues should be covered
        if best_subset[r] is None:
            # Fallback (should not occur for p=101, n=10)
            result.append([1])
        else:
            result.append(best_subset[r])
            
    return result
