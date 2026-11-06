import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Primes up to 127
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    num_primes = len(primes)
    
    # Precompute factor vectors
    def get_vec(n):
        v = [0] * num_primes
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        return tuple(v)
        
    vecs = {i: get_vec(i) for i in range(2, 129)}
    zero_vec = (0,) * num_primes

    # Global ID registry for vectors to speed up set operations
    vec_to_id = {zero_vec: 0}
    next_id = 1
    
    def get_id(vec):
        nonlocal next_id
        if vec in vec_to_id:
            return vec_to_id[vec]
        vec_to_id[vec] = next_id
        next_id += 1
        return vec_to_id[vec]

    # Beam Search
    # State: (seq, prefix_vecs, prod_ids)
    # seq: list of ints
    # prefix_vecs: list of tuples (V0, V1, ..., Vk)
    # prod_ids: set of int IDs representing all block products
    
    # Initial state
    beam = [([], [zero_vec], set())]
    
    beam_width = 200  # Reduced to allow more iterations
    
    best_seq = []
    
    for n in range(2, 129):
        if time.time() > deadline - 10.0:  # Leave time for polish
            break
            
        new_beam = []
        v_n = vecs[n]
        
        for seq, pvecs, pset in beam:
            # Option 1: Exclude n
            new_beam.append((seq, pvecs, pset))
            
            # Option 2: Include n
            last_pvec = pvecs[-1]
            # Vector addition
            v_new_list = [last_pvec[i] + v_n[i] for i in range(num_primes)]
            v_new = tuple(v_new_list)
            
            valid = True
            new_prod_ids = []
            
            # Check all suffixes ending at n
            # Products are v_new - pvecs[i]
            for i in range(len(pvecs)):
                diff_list = [v_new[j] - pvecs[i][j] for j in range(num_primes)]
                diff_tup = tuple(diff_list)
                diff_id = get_id(diff_tup)
                
                if diff_id in pset:
                    valid = False
                    break
                
                new_prod_ids.append(diff_id)
            
            if valid:
                new_seq = seq + [n]
                new_pvecs = pvecs + [v_new]
                # Update pset
                new_pset = pset.union(new_prod_ids)
                new_beam.append((new_seq, new_pvecs, new_pset))
        
        # Select top beam_width states from new_beam
        new_beam.sort(key=lambda x: len(x[0]), reverse=True)
        beam = new_beam[:beam_width]
        
        # Track best
        if beam and len(beam[0][0]) > len(best_seq):
            best_seq = beam[0][0]
        
        # Early termination
        if beam and len(beam[0][0]) == 127:
            return beam[0][0]

    # Polish pass: try to insert skipped numbers into best_seq
    if best_seq and time.time() < deadline - 2.0:
        best_seq = _polish_sequence(best_seq, vecs, zero_vec, get_id, num_primes, deadline)
    
    return best_seq if best_seq else []

def _polish_sequence(seq, vecs, zero_vec, get_id, num_primes, deadline):
    """Try to insert skipped numbers into the sequence to make it longer."""
    if not seq:
        return seq
    
    seq_set = set(seq)
    skipped = [n for n in range(2, 129) if n not in seq_set]
    
    # Build prefix vectors and product set for the current sequence
    pvecs = [zero_vec]
    for n in seq:
        last = pvecs[-1]
        v_n = vecs[n]
        v_new = tuple(last[i] + v_n[i] for i in range(num_primes))
        pvecs.append(v_new)
    
    # Compute all block products
    prod_ids = set()
    for i in range(len(pvecs)):
        for j in range(i+1, len(pvecs)):
            diff = tuple(pvecs[j][k] - pvecs[i][k] for k in range(num_primes))
            prod_ids.add(get_id(diff))
    
    improved = False
    for n in skipped:
        if time.time() > deadline - 1.0:
            break
        
        v_n = vecs[n]
        
        # Try inserting n at each position
        for pos in range(len(seq) + 1):
            # Build new sequence with n inserted at pos
            new_seq = seq[:pos] + [n] + seq[pos:]
            
            # Check if strictly increasing
            if any(new_seq[i] >= new_seq[i+1] for i in range(len(new_seq)-1)):
                continue
            
            # Rebuild prefix vectors
            new_pvecs = [zero_vec]
            for m in new_seq:
                last = new_pvecs[-1]
                v_m = vecs[m]
                v_new = tuple(last[i] + v_m[i] for i in range(num_primes))
                new_pvecs.append(v_new)
            
            # Check all block products for duplicates
            new_prod_ids = set()
            valid = True
            for i in range(len(new_pvecs)):
                for j in range(i+1, len(new_pvecs)):
                    diff = tuple(new_pvecs[j][k] - new_pvecs[i][k] for k in range(num_primes))
                    pid = get_id(diff)
                    if pid in new_prod_ids:
                        valid = False
                        break
                    new_prod_ids.add(pid)
            
            if valid:
                seq = new_seq
                seq_set = set(seq)
                skipped.remove(n)
                improved = True
                break
        
        if not improved:
            break
    
    return seq
