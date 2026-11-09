def construct():
    import time
    import random
    import numpy as np

    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s kill

    N = 17
    MIN_D = 6
    SIZE = 1 << N

    # Precompute weights for all 2^17 words
    # Using a fast method to generate bit counts
    # i.bit_count() is fast in Python 3.10+
    weights = np.fromiter((i.bit_count() for i in range(SIZE)), dtype=np.int8, count=SIZE)

    # The known optimal linear code is [17, 8, 6] with 256 codewords.
    # We first construct this linear code using a greedy basis search.
    # Then we attempt to extend it with non-linear words.
    
    best_code = []
    best_size = 0

    # --- Phase 1: Construct Linear Code [17, 8, 6] ---
    # Greedy basis construction.
    # We want a basis b1, ..., b8 such that all 256 linear combinations have weight >= 6.
    # Strategy: Try to add vectors one by one.
    # To speed up, we can pre-filter candidates.
    # A vector g is a candidate for the next basis vector if min_{c in C} wt(g^c) >= 6.
    
    def try_greedy_linear(time_limit):
        current_code = [0]
        basis = []
        target_basis_size = 8
        
        # Candidates: all non-zero vectors.
        # We sort them by weight descending? Or just random?
        # High weight vectors are safer for the first few steps.
        # Let's try a mix: start with high weight, then random.
        
        # Create a list of indices 1..SIZE-1
        indices = list(range(1, SIZE))
        
        # Sort by weight descending to prioritize "strong" vectors early
        # This might help find a valid basis faster.
        indices.sort(key=lambda x: -weights[x])
        
        for g in indices:
            if time.time() > time_limit:
                break
            if len(basis) >= target_basis_size:
                break
                
            # Check if g can be added to the basis
            valid = True
            # Optimization: check against current_code
            # current_code size is 2^len(basis)
            for c in current_code:
                if weights[g ^ c] < MIN_D:
                    valid = False
                    break
            
            if valid:
                basis.append(g)
                # Expand code
                # New words are c ^ g for c in current_code
                # We can do this in a loop or vectorized?
                # For small sizes, loop is fine.
                new_words = [c ^ g for c in current_code]
                current_code.extend(new_words)
                
        return current_code

    # Try to find the linear code
    code = try_greedy_linear(deadline - 5)
    if len(code) > best_size:
        best_code = code
        best_size = len(code)

    # If we didn't find 256, we might have less.
    # But usually, the greedy method with sorted weights finds the [17,8,6] code quickly.
    
    if best_size >= 256:
        # --- Phase 2: Extend with Non-linear Words ---
        # We have a linear code C of size 256.
        # We want to find words w not in C such that min_{c in C} d(w, c) >= 6.
        # If we find such w, we add it.
        # Then we check for more words against the new set C U {w}.
        
        C = best_code
        C_set = set(C)
        C_np = np.array(C, dtype=np.int32)
        
        added = 0
        limit_added = 100  # Allow more additions if possible
        
        # Candidates: all words not in C.
        # We scan in an order that might find valid words quickly.
        # High weight words? Or random?
        # Let's scan in random order to avoid bias, or weight order.
        # Weight order (descending) might find words far from the code faster?
        # Actually, the code contains words of various weights.
        # Let's just scan all indices again, skipping those in C.
        
        indices = list(range(SIZE))
        # Shuffle to be safe? Or keep deterministic?
        # Deterministic is better for reproducibility if it works.
        # Let's try scanning by weight descending again.
        indices.sort(key=lambda x: -weights[x])
        
        for w in indices:
            if time.time() > deadline - 2:
                break
            if added >= limit_added:
                break
                
            if w in C_set:
                continue
                
            # Check distance to all c in C
            # Vectorized: dists = weights[w ^ C_np]
            # min(dists) >= MIN_D
            
            try:
                dists = weights[w ^ C_np]
                if dists.min() >= MIN_D:
                    # Add w
                    C.append(w)
                    C_set.add(w)
                    C_np = np.append(C_np, w)
                    added += 1
            except Exception:
                # Fallback if numpy fails (unlikely)
                valid = True
                for c in C:
                    if weights[w ^ c] < MIN_D:
                        valid = False
                        break
                if valid:
                    C.append(w)
                    C_set.add(w)
                    C_np = np.append(C_np, w)
                    added += 1
                    
        # Return the extended code
        return C

    # If we failed to find a large linear code, return what we have
    return best_code if best_code else [0]
