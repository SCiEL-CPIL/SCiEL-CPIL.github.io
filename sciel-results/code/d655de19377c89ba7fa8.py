import numpy as np
import itertools

def kissing_number11() -> np.ndarray:
    d = 11
    # Generate all weight-4 vectors
    vectors = []
    supports = []
    for indices in itertools.combinations(range(d), 4):
        for signs in itertools.product([-1, 1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, sign in zip(indices, signs):
                v[idx] = sign
            vectors.append(v)
            supports.append(indices)
    
    # Precompute triples for each vector index
    # triple_map: (C, p) -> set of supports that have used this triple
    triple_map = {}
    
    selected_vectors = []
    selected_indices = []
    
    # Shuffle indices for randomization
    indices = list(range(len(vectors)))
    np.random.shuffle(indices)
    
    for idx in indices:
        v = vectors[idx]
        S = supports[idx]
        
        # Check conflicts
        valid = True
        # Check for duplicates
        # Since we are shuffling unique vectors, duplicates are not possible in the input list.
        # But wait, the generation loop creates unique vectors.
        # So no duplicate check needed against the list, only against selected.
        
        # Check triple conflicts
        non_zero = np.nonzero(v)[0]
        for comb in itertools.combinations(non_zero, 3):
            C = tuple(sorted(comb))
            p = tuple(v[i] for i in C)
            key = (C, p)
            
            if key in triple_map:
                used_supports = triple_map[key]
                # If any used support is different from S, conflict
                for T in used_supports:
                    if T != S:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            selected_vectors.append(v)
            # Update triple_map
            for comb in itertools.combinations(non_zero, 3):
                C = tuple(sorted(comb))
                p = tuple(v[i] for i in C)
                key = (C, p)
                if key not in triple_map:
                    triple_map[key] = set()
                triple_map[key].add(S)
                
    return np.array(selected_vectors)
