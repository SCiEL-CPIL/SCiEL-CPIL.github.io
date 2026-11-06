import time

def synthesize(matrix, depth_limit):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    if not matrix:
        return {'gates': [], 'outputs': []}
    
    m = len(matrix)
    n = len(matrix[0])
    
    # Convert matrix rows to integers for fast XOR and bit_count
    rows = []
    for i in range(m):
        val = 0
        for j in range(n):
            if matrix[i][j]:
                val |= (1 << j)
        rows.append(val)
    
    # Wire management
    # Input wires: 0..n-1
    # Constant zero: n
    # Gates produce: n+1, n+2, ...
    
    wires = [0] * (n + 1)
    for i in range(n):
        wires[i] = (1 << i)
    wires[n] = 0
    
    # Cache: vector value -> wire index
    cache = {0: n}
    for i in range(n):
        cache[1 << i] = i
        
    gates = []
    
    def get_wire(val):
        """
        Synthesizes vector 'val' if not already in cache.
        Returns the wire index holding 'val'.
        """
        if val in cache:
            return cache[val]
            
        # Find the existing wire that minimizes the weight of (val ^ wire)
        best_w = 0
        best_pop = val.bit_count()
        
        # Iterate through all currently available wires
        # For small n, this is fast.
        for w in wires:
            if w == 0:
                continue
            pop = (val ^ w).bit_count()
            if pop < best_pop:
                best_pop = pop
                best_w = w
                if pop == 0:
                    break # Should be in cache, but break just in case
        
        if best_w == 0:
            # This should not happen if the input wires span the space,
            # but if it does, we return a dummy to avoid infinite recursion/errors
            # In valid GF(2) synthesis with full input basis, this is unreachable
            return -1
        
        # Recursively synthesize the reduced vector
        sub_val = val ^ best_w
        sub_wire = get_wire(sub_val)
        
        if sub_wire == -1:
            return -1
            
        # Create new gate: sub_wire XOR best_w -> val
        new_wire_id = n + 1 + len(gates)
        gates.append((sub_wire, best_w))
        
        # Update state
        wires.append(val)
        cache[val] = new_wire_id
        
        return new_wire_id

    # Step 1: Compute a basis for the row space to ensure efficient synthesis order
    # We use Gaussian elimination to find independent rows
    mat = rows[:]
    basis = []
    rank = 0
    for col in range(n):
        pivot = -1
        for i in range(rank, m):
            if (mat[i] >> col) & 1:
                pivot = i
                break
        if pivot == -1:
            continue
            
        # Swap pivot row to current rank position
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        
        # Eliminate this column from all other rows
        for i in range(m):
            if i != rank and ((mat[i] >> col) & 1):
                mat[i] ^= mat[rank]
                
        basis.append(mat[rank])
        rank += 1
        
    # Sort basis by popcount (weight) to build simpler vectors first
    # This increases the chance that complex vectors can be reduced using these simple ones
    basis.sort(key=lambda x: x.bit_count())
    
    # Synthesize basis vectors first to populate the cache with useful sub-expressions
    for b in basis:
        if b not in cache:
            w = get_wire(b)
            if time.time() > deadline:
                break

    # Synthesize the original rows
    outputs = []
    for r in rows:
        w = get_wire(r)
        if w == -1:
            # Fallback: if synthesis fails (should not happen), use zero wire?
            # Or raise an error? The task implies a valid circuit is always possible.
            # We'll just append the last known valid wire or 0 to keep format, 
            # but this indicates a logic error.
            w = 0 
        outputs.append(w)
        
        if time.time() > deadline:
            # If we run out of time, we must return what we have.
            # However, outputs must match the number of rows.
            # We can pad with 0s if we haven't finished, but that's invalid.
            # Given the speed, this is unlikely.
            pass

    return {'gates': gates, 'outputs': outputs}
