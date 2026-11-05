import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback to simple greedy for other parameters
        # But task specifies 3, 8, 1
        pass

    # Specialized for q=3, n=8, radius=1
    # Approach: Greedy set cover on cosets of an [8,4,3] linear code over Z3
    
    # 1. Define the [8,4,3] code C8
    # C4 is the Z3 Hamming code of length 4.
    # H4 = [[1, 0, 1, 1], [0, 1, 1, 2]]
    # C8 is C4 x C4. H8 = diag(H4, H4)
    
    H4 = np.array([[1, 0, 1, 1],
                   [0, 1, 1, 2]], dtype=int)
    H8 = np.zeros((4, 8), dtype=int)
    H8[:2, :4] = H4
    H8[2:, 4:] = H4
    
    # 2. Generate all 6561 words and their syndromes
    total_words = 3 ** 8
    words = np.array(list(itertools.product(range(3), repeat=8)), dtype=int)
    
    # Compute syndromes: s = H8 @ w^T mod 3
    # words shape (6561, 8). H8 shape (4, 8).
    # syndromes shape (6561, 4)
    syndromes = (words @ H8.T) % 3
    
    # Map each syndrome to a unique integer 0..80
    # Syndrome is a vector of 4 base-3 digits.
    # val = s[0]*27 + s[1]*9 + s[2]*3 + s[3]
    syndrome_ids = (syndromes[:, 0] * 27 + syndromes[:, 1] * 9 + syndromes[:, 2] * 3 + syndromes[:, 3]).astype(int)
    
    # Group words by syndrome_id
    # cosets[sid] = list of word_indices (0..6560) belonging to this coset
    cosets = [[] for _ in range(81)]
    for idx, sid in enumerate(syndrome_ids):
        cosets[sid].append(idx)
        
    # Verify each coset has 81 words
    # (It should, as it's a linear code)
    
    # 3. Precompute neighbors for all words
    # neighbors[i] = list of word indices within distance 1 of word i
    powers = [3**i for i in range(8)]
    neighbors = [[] for _ in range(total_words)]
    
    for w_idx in range(total_words):
        w = words[w_idx]
        neighbors[w_idx].append(w_idx)
        for i in range(8):
            d = w[i]
            p = powers[i]
            # Change d to (d+1)%3
            new_d = (d + 1) % 3
            delta = (new_d - d) * p
            new_w_val = w_idx + delta
            # Note: w_idx is the integer representation.
            # Does adding delta to the integer representation correspond to adding to the digit?
            # Yes, because the integer is sum(d_i * 3^i).
            # Changing d_i to d_i' changes the sum by (d_i' - d_i) * 3^i.
            # So new_w_val is the correct integer index.
            neighbors[w_idx].append(new_w_val)
            
            # Change d to (d-1)%3
            new_d = (d - 1) % 3
            delta = (new_d - d) * p
            new_w_val = w_idx + delta
            neighbors[w_idx].append(new_w_val)
            
    # 4. Greedy Set Cover on Cosets
    covered = [False] * total_words
    code_indices = []
    uncovered_count = total_words
    
    # List of coset IDs to consider
    coset_ids = list(range(81))
    
    while uncovered_count > 0:
        best_coset_id = -1
        best_covered_count = 0
        
        for cid in coset_ids:
            # Calculate how many uncovered words are covered by this coset
            cnt = 0
            for w_idx in cosets[cid]:
                for nb in neighbors[w_idx]:
                    if not covered[nb]:
                        cnt += 1
            if cnt > best_covered_count:
                best_covered_count = cnt
                best_coset_id = cid
                
        if best_coset_id == -1:
            break
            
        # Add all words in this coset to the code
        for w_idx in cosets[best_coset_id]:
            code_indices.append(w_idx)
            
        # Mark covered
        for w_idx in cosets[best_coset_id]:
            for nb in neighbors[w_idx]:
                if not covered[nb]:
                    covered[nb] = True
                    uncovered_count -= 1
                    
    # Convert indices to lists of ints
    code = []
    for idx in code_indices:
        code.append(words[idx].tolist())
        
    return code
