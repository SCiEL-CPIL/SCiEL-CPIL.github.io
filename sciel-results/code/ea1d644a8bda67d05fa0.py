import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Try to find a solution for n=31
    n = 31
    # Initialize random tournament
    # N_plus[i] is a bitmask of vertices i points to
    N_plus = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                N_plus[i] |= (1 << j)
            else:
                N_plus[j] |= (1 << i)
                
    # Precompute all 4-sets and their indices?
    # Or just iterate?
    # We need to map a set to an index to update the is_bad array.
    # Let's create a list of all 4-sets.
    all_4sets = list(itertools.combinations(range(n), 4))
    num_4sets = len(all_4sets)
    
    # Map from set (tuple) to index
    # This might be slow to build.
    # Alternative: Just store the list and iterate through all_4sets to initialize is_bad.
    # For updates, we need to find the index of the affected sets.
    # Since we generate the affected sets, we need their indices.
    # We can build a dictionary: set -> index.
    set_to_idx = {s: i for i, s in enumerate(all_4sets)}
    
    # Initialize is_bad and energy
    is_bad = [False] * num_4sets
    energy = 0
    for i, s in enumerate(all_4sets):
        a, b, c, d = s
        mask = N_plus[a] & N_plus[b] & N_plus[c] & N_plus[d]
        if mask == 0:
            is_bad[i] = True
            energy += 1
            
    best_energy = energy
    best_N_plus = N_plus[:]
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 1e-5
    
    while time.time() < deadline and T > min_T:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Determine current direction
        # If N_plus[u] has bit v, then u -> v
        # If N_plus[v] has bit u, then v -> u
        
        has_u_to_v = (N_plus[u] >> v) & 1
        has_v_to_u = (N_plus[v] >> u) & 1
        
        # Flip: if u->v, make v->u. Else make u->v.
        # We just toggle the edge.
        
        # Identify affected 4-sets
        # Sets containing u or v
        # We can just iterate all_4sets and check if u in s or v in s?
        # That's 31465 checks per step. 31465 * 10000 steps = 3e8 ops. Might be too slow in Python.
        
        # Better: Precompute for each vertex i, the list of indices of 4-sets containing i.
        # sets_containing[i] = [idx1, idx2, ...]
        
        # Let's build this.
        sets_containing = [[] for _ in range(n)]
        for i, s in enumerate(all_4sets):
            for v_idx in s:
                sets_containing[v_idx].append(i)
                
        # Now for flip (u, v):
        # Affected indices are union of sets_containing[u] and sets_containing[v]
        
        affected_indices = set(sets_containing[u])
        affected_indices.update(sets_containing[v])
        
        # Calculate old energy contribution
        old_contrib = 0
        for idx in affected_indices:
            if is_bad[idx]:
                old_contrib += 1
                
        # Perform flip
        if has_u_to_v:
            # u -> v becomes v -> u
            N_plus[u] &= ~(1 << v)
            N_plus[v] |= (1 << u)
        else:
            # v -> u becomes u -> v
            N_plus[v] &= ~(1 << u)
            N_plus[u] |= (1 << v)
            
        # Calculate new energy contribution
        new_contrib = 0
        for idx in affected_indices:
            a, b, c, d = all_4sets[idx]
            mask = N_plus[a] & N_plus[b] & N_plus[c] & N_plus[d]
            if mask == 0:
                is_bad[idx] = True
                new_contrib += 1
            else:
                is_bad[idx] = False
                
        delta_E = new_contrib - old_contrib
        
        if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
            energy += delta_E
            if energy < best_energy:
                best_energy = energy
                best_N_plus = N_plus[:]
        else:
            # Revert
            if has_u_to_v:
                N_plus[u] |= (1 << v)
                N_plus[v] &= ~(1 << u)
            else:
                N_plus[v] |= (1 << u)
                N_plus[u] &= ~(1 << v)
                
            # Revert is_bad
            for idx in affected_indices:
                a, b, c, d = all_4sets[idx]
                mask = N_plus[a] & N_plus[b] & N_plus[c] & N_plus[d]
                if mask == 0:
                    is_bad[idx] = True
                else:
                    is_bad[idx] = False
                    
        T *= cooling
        
    if best_energy == 0:
        # Convert best_N_plus to orientation list
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_N_plus[i] >> j) & 1:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': n, 'orientation': orientation}
    else:
        # Fallback to seed solution
        p = 83
        squares = {x * x % p for x in range(1, p)}
        return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
