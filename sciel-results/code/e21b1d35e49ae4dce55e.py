import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 17
    MIN_DIST = 6
    
    # 1. Generate the 32 RM words
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # 2. Find a basis for these 32 words
    # The words are linear combinations of basis vectors.
    # We can extract a basis using Gaussian elimination on the list of words.
    basis = []
    for w in words:
        if w == 0:
            continue
        # Check if w is in span of basis
        is_dep = False
        temp = w
        for b in basis:
            if temp.bit_length() > 0:
                # Find highest set bit
                hb = temp.bit_length() - 1
                # Find basis vector with this highest bit
                # This simple approach is O(k^2) but k is small
                # Better to maintain basis in row echelon form
                pass
        
    # Let's do proper Gaussian elimination to get a basis
    # Convert words to a list of vectors
    vecs = words[:]
    basis = []
    for v in vecs:
        if v == 0:
            continue
        pivot = v.bit_length() - 1
        # Check if we already have a basis vector with this pivot
        found = False
        for i, b in enumerate(basis):
            if b.bit_length() - 1 == pivot:
                # Eliminate
                v = v ^ b
                found = True
                break
        if not found:
            # Add to basis
            basis.append(v)
            # We should actually reduce other basis vectors? 
            # For just checking span, a simple basis is enough.
            # But to check if a new vector is in the span, we need to reduce it.
            
    # Let's re-implement basis extraction and span checking more robustly.
    # We want a list of basis vectors `basis` such that we can easily check if a vector is in their span.
    # We'll keep the basis in a reduced form.
    
    def get_basis(vecs):
        b = []
        for v in vecs:
            if v == 0:
                continue
            # Reduce v by existing basis
            # Sort basis by pivot bit descending?
            # Let's just iterate and eliminate
            changed = True
            while changed:
                changed = False
                for bv in b:
                    if bv == 0:
                        continue
                    # If highest bit of v is set in bv, eliminate
                    # Actually, standard Gaussian elimination:
                    # Find highest bit of v
                    if v == 0:
                        break
                    pivot = v.bit_length() - 1
                    # Find basis vector with this pivot
                    for i, bv in enumerate(b):
                        if bv.bit_length() - 1 == pivot:
                            v = v ^ bv
                            changed = True
                            break
            if v != 0:
                b.append(v)
        return b

    basis = get_basis(words)
    # basis should have 5 vectors
    
    def expand_basis(basis, N, min_dist, deadline):
        # basis: list of vectors
        # Returns expanded basis or the same if failed
        current_dim = len(basis)
        target_dim = 8 # Aim for 256 words
        
        # Precompute subspace? 
        # For dim 5, 32 vectors.
        # For dim 6, 64 vectors.
        # We can generate the subspace list each time we add a vector.
        
        subspace = [0]
        for b in basis:
            subspace += [s ^ b for s in subspace]
            
        for _ in range(target_dim - current_dim):
            found = False
            # Iterate through all possible vectors
            # Optimization: We can skip vectors in the current subspace
            # Create a set of subspace for O(1) lookup
            sub_set = set(subspace)
            
            for v in range(2**N):
                if time.time() > deadline:
                    break
                if v in sub_set:
                    continue
                
                # Check if v is a valid extension
                # 1. wt(v) >= min_dist
                if v.bit_count() < min_dist:
                    continue
                
                # 2. wt(u ^ v) >= min_dist for all u in subspace (non-zero)
                valid = True
                for u in subspace:
                    if u == 0:
                        continue
                    if (u ^ v).bit_count() < min_dist:
                        valid = False
                        break
                if valid:
                    # Add v to basis
                    basis.append(v)
                    # Update subspace
                    new_subspace = []
                    for s in subspace:
                        new_subspace.append(s)
                        new_subspace.append(s ^ v)
                    subspace = new_subspace
                    sub_set = set(subspace)
                    found = True
                    break
            
            if not found:
                break
            if time.time() > deadline:
                break
                
        return basis

    # Expand the basis
    basis = expand_basis(basis, N, MIN_DIST, deadline)
    
    # Generate all codewords from the final basis
    final_words = [0]
    for b in basis:
        final_words += [w ^ b for w in final_words]
        
    # Verify minimum distance
    # This is O(M^2). For M=256, 256^2/2 = 32k checks. Fast.
    min_d = N + 1
    for i in range(len(final_words)):
        for j in range(i+1, len(final_words)):
            d = (final_words[i] ^ final_words[j]).bit_count()
            if d < min_d:
                min_d = d
                if min_d < MIN_DIST:
                    break
        if min_d < MIN_DIST:
            break
            
    if min_d >= MIN_DIST:
        return final_words
    else:
        # Fallback to the original 32 words if something went wrong
        return words
