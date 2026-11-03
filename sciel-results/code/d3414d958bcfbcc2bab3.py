import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    total_words = q ** n
    
    # Generate all words
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    
    # Precompute neighbors
    # neighbors_list[i]: list of indices
    # neighbors_set[i]: set of indices
    neighbors_list = []
    neighbors_set = []
    
    for i in range(total_words):
        w = words[i]
        neigh = {i}
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                # Compute index
                idx = 0
                base = 1
                for k in range(n):
                    idx += w[k] * base
                    base *= q
                neigh.add(idx)
            w[pos] = orig
        neighbors_list.append(list(neigh))
        neighbors_set.append(neigh)
        
    def greedy_construct(seed):
        random.seed(seed)
        covered = np.zeros(total_words, dtype=bool)
        code = []
        
        # Initialize gains
        gains = np.array([len(neighbors_list[i]) for i in range(total_words)], dtype=np.int32)
        
        while True:
            # Check if all covered
            if np.all(covered):
                break
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[random.randint(0, len(candidates)-1)]
            
            code.append(best_idx)
            
            # Update covered
            for u in neighbors_list[best_idx]:
                covered[u] = True
            
            # Recompute gains
            # gains[i] = number of u in neighbors[i] such that not covered[u]
            # Vectorized approach might be tricky with sparse updates, 
            # but N=2187 is small.
            for i in range(total_words):
                # Sum of 1 for u in neighbors_list[i] if not covered[u]
                # Using numpy sum on a boolean mask
                # neighbors_list[i] is a list of ints
                # We can do: np.sum(~covered[neighbors_list[i]])
                # This creates a temporary array. 2187 * 15 size.
                # 2187 iterations of this.
                gains[i] = np.sum(~covered[neighbors_list[i]])
                
        # Redundancy Removal
        # Convert code to set for faster lookup? No, we need to iterate.
        # We use neighbors_set for O(1) membership.
        
        # Sort code to make removal deterministic?
        # Let's just iterate.
        changed = True
        while changed:
            changed = False
            # Iterate backwards
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                c_set = neighbors_set[c]
                essential = False
                for u in c_set:
                    # Check if u is covered by any other codeword
                    covered_by_other = False
                    for j in range(len(code)):
                        if j == i:
                            continue
                        if u in neighbors_set[code[j]]:
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        essential = True
                        break
                if not essential:
                    code.pop(i)
                    changed = True
                    
        return [words[i].tolist() for i in code]

    best_code = None
    best_size = float('inf')
    
    # Run trials
    # We can increase trials if time permits
    num_trials = 100 
    
    for t in range(num_trials):
        if time.time() > deadline:
            break
        code = greedy_construct(t)
        size = len(code)
        if size < best_size:
            best_size = size
            best_code = code
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        
    return best_code
