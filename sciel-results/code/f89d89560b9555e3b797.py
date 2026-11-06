<<<<<<< SEARCH
def _local_search(n, time_limit, seed=42):
    """
    Attempts to find a valid tournament on n vertices using local search.
    Returns orientation list if successful, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Initialize random tournament
    # in_bits[i] bitmask of j such that j -> i
    in_bits = [0] * n
    # orientation list for return
    orientation = [0] * (n * (n - 1) // 2)
    
    # Build initial random tournament
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                # i -> j
                orientation[idx] = 0
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                orientation[idx] = 1
                in_bits[i] |= (1 << j)
            idx += 1
            
    # Precompute lists of 4-subsets for affected sets calculation
    # tuples: list of (a,b,c,d)
    # states: list of 1 if bad, 0 if good
    # vertex_to_tuples: list of lists of indices into tuples
    tuples = []
    states = []
    idx_t = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    tuples.append((a, b, c, d))
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    is_bad = 1 if (common & ~mask) == 0 else 0
                    states.append(is_bad)
                    idx_t += 1
                    
    bad = sum(states)
    if bad == 0:
        return orientation

    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
            
    # Precompute edge index map for O(1) lookup
    # edge_idx[i][j] = index in orientation list for pair (i,j) where i<j
    # We need to map (u,v) to index.
    # Index formula for (i,j) with i<j:
    # sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
    # Or simpler: just store a 2D array or a dictionary.
    # n is small (<=90), so a 2D list of size 90x90 is fine.
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Get affected tuple indices
        # Union of tuples containing i or j
        # Since we need to check state change, we iterate over unique indices
        # Using a set is slow. We can just iterate both lists and use a set for deduplication
        # or just iterate and handle duplicates carefully?
        # Actually, a 4-set can contain both i and j.
        # If we iterate vertex_to_tuples[i] and vertex_to_tuples[j], we might visit the same tuple twice.
        # We can use a set of indices.
        aff_idx = set(vertex_to_tuples[i])
        aff_idx.update(vertex_to_tuples[j])
        
        # Compute delta bad
        delta = 0
        changed_states = [] # list of (idx, new_state)
        
        # Determine current direction
        # i -> j iff (in_bits[j] >> i) & 1
        if (in_bits[j] >> i) & 1:
            # Currently i -> j
            # Flip to j -> i
            # i removed from In[j], j added to In[i]
            new_in_i = in_bits[i] | (1 << j)
            new_in_j = in_bits[j] & ~(1 << i)
            flip_to_j_to_i = True
        else:
            # Currently j -> i
            # Flip to i -> j
            # j removed from In[i], i added to In[j]
            new_in_i = in_bits[i] & ~(1 << j)
            new_in_j = in_bits[j] | (1 << i)
            flip_to_j_to_i = False
            
        for idx_t in aff_idx:
            a, b, c, d = tuples[idx_t]
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            # Get new in_bits for a,b,c,d
            ni_a = new_in_i if a == i else (new_in_j if a == j else in_bits[a])
            ni_b = new_in_i if b == i else (new_in_j if b == j else in_bits[b])
            ni_c = new_in_i if c == i else (new_in_j if c == j else in_bits[c])
            ni_d = new_in_i if d == i else (new_in_j if d == j else in_bits[d])
            
            common = ni_a & ni_b & ni_c & ni_d
            is_bad_now = 1 if (common & ~mask) == 0 else 0
            
            old_state = states[idx_t]
            if old_state != is_bad_now:
                delta += (is_bad_now - old_state)
                changed_states.append((idx_t, is_bad_now))
        
        # Acceptance
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Apply flip
            if flip_to_j_to_i:
                # i -> j becomes j -> i
                in_bits[i] |= (1 << j)
                in_bits[j] &= ~(1 << i)
                orientation[edge_idx[i][j]] = 1 # j -> i
            else:
                in_bits[i] &= ~(1 << j)
                in_bits[j] |= (1 << i)
                orientation[edge_idx[i][j]] = 0 # i -> j
                
            # Update states
            for idx_t, state in changed_states:
                states[idx_t] = state
            bad += delta
            
            if bad == 0:
                return orientation
                
            T *= cooling
            if T < T_min:
                T = T_min
        else:
            pass # Reject
        
    return None
=======
def _local_search(n, time_limit, seed=42):
    """
    Attempts to find a valid tournament on n vertices using local search.
    Returns orientation list if successful, else None.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Initialize random tournament
    # in_bits[i] bitmask of j such that j -> i
    in_bits = [0] * n
    # orientation list for return
    orientation = [0] * (n * (n - 1) // 2)
    
    # Build initial random tournament
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                # i -> j
                orientation[idx] = 0
                in_bits[j] |= (1 << i)
            else:
                # j -> i
                orientation[idx] = 1
                in_bits[i] |= (1 << j)
            idx += 1
            
    # Precompute lists of 4-subsets for affected sets calculation
    tuples = []
    states = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    tuples.append((a, b, c, d))
                    mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                    common = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
                    is_bad = 1 if (common & ~mask) == 0 else 0
                    states.append(is_bad)
                    
    bad = sum(states)
    if bad == 0:
        return orientation

    vertex_to_tuples = [[] for _ in range(n)]
    for i, t in enumerate(tuples):
        for v in t:
            vertex_to_tuples[v].append(i)
            
    # Precompute edge index map for O(1) lookup
    edge_idx = [[-1] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
    
    # Precompute affected set masks for each vertex pair
    # affected_mask[i][j] = bitmask of tuple indices affected by flipping edge (i,j)
    affected_masks = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            mask = 0
            for idx_t in vertex_to_tuples[i]:
                mask |= (1 << idx_t)
            for idx_t in vertex_to_tuples[j]:
                mask |= (1 << idx_t)
            affected_masks[i][j] = mask
            affected_masks[j][i] = mask
    
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    no_improve_count = 0
    best_bad = bad
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        # Get affected tuple indices
        aff_idx = set(vertex_to_tuples[i])
        aff_idx.update(vertex_to_tuples[j])
        
        # Compute delta bad
        delta = 0
        changed_states = [] # list of (idx, new_state)
        
        # Determine current direction
        if (in_bits[j] >> i) & 1:
            # Currently i -> j
            new_in_i = in_bits[i] | (1 << j)
            new_in_j = in_bits[j] & ~(1 << i)
            flip_to_j_to_i = True
        else:
            # Currently j -> i
            new_in_i = in_bits[i] & ~(1 << j)
            new_in_j = in_bits[j] | (1 << i)
            flip_to_j_to_i = False
            
        for idx_t in aff_idx:
            a, b, c, d = tuples[idx_t]
            mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
            
            ni_a = new_in_i if a == i else (new_in_j if a == j else in_bits[a])
            ni_b = new_in_i if b == i else (new_in_j if b == j else in_bits[b])
            ni_c = new_in_i if c == i else (new_in_j if c == j else in_bits[c])
            ni_d = new_in_i if d == i else (new_in_j if d == j else in_bits[d])
            
            common = ni_a & ni_b & ni_c & ni_d
            is_bad_now = 1 if (common & ~mask) == 0 else 0
            
            old_state = states[idx_t]
            if old_state != is_bad_now:
                delta += (is_bad_now - old_state)
                changed_states.append((idx_t, is_bad_now))
        
        # Acceptance
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            # Apply flip
            if flip_to_j_to_i:
                in_bits[i] |= (1 << j)
                in_bits[j] &= ~(1 << i)
                orientation[edge_idx[i][j]] = 1
            else:
                in_bits[i] &= ~(1 << j)
                in_bits[j] |= (1 << i)
                orientation[edge_idx[i][j]] = 0
                
            # Update states
            for idx_t, state in changed_states:
                states[idx_t] = state
            bad += delta
            
            if bad == 0:
                return orientation
            
            if bad < best_bad:
                best_bad = bad
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Restart temperature if stuck
            if no_improve_count > 500:
                T = min(1.0, T * 2)
                no_improve_count = 0
            
            T *= cooling
            if T < T_min:
                T = T_min
        else:
            pass
        
    return None
>>>>>>> REPLACE
