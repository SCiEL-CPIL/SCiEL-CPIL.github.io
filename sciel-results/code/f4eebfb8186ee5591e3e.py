import itertools

def construct():
    # Precompute modular inverses of 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)
        
    # best_subset[r] will store the list of terms for residue r
    # Initially None, meaning not yet assigned
    best_subset = [None] * 101
    assigned_count = 0
    
    # Iterate by subset length from 1 to 10
    # We want the shortest subset for each residue
    for length in range(1, 11):
        # Generate all combinations of 'length' distinct elements from 1..10
        for terms in itertools.combinations(range(1, 11), length):
            # Calculate the sum of inverses modulo 101
            residue = sum(invs[x] for x in terms) % 101
            
            # If this residue hasn't been assigned a subset yet,
            # assign this one. Since we iterate by increasing length,
            # this is the minimum length subset for this residue.
            if best_subset[residue] is None:
                best_subset[residue] = list(terms)
                assigned_count += 1
                
        # If all 101 residues are covered, we can stop early
        if assigned_count == 101:
            break
            
    return best_subset
