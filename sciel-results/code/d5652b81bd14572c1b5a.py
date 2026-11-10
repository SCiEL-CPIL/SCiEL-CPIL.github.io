import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    d = 11
    
    # 1. Generate pool of weight 4 vectors
    pool = []
    for idx in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for i, s in zip(idx, signs):
                v[i] = s
            pool.append(v)
    pool = np.array(pool)
    N = len(pool)
    
    # 2. Compute conflict matrix
    # Conflict if dot > 2
    D = pool @ pool.T
    # D[i, j] > 2
    # We can store C as a list of bitsets
    # bitset[i] is an int where bit j is 1 if C[i, j] is True
    
    # To speed up bitset creation, we can use numpy to create the boolean matrix first
    # Then convert rows to ints.
    # N=5280, N^2 = 27M booleans. 27MB. Feasible.
    C_bool = D > 2
    
    # Convert to bitsets
    # Python ints are efficient for popcount via bit_count() in 3.8+? 
    # Or use bin(x).count('1') which is slow.
    # We can use a lookup table for 8-bit chunks or just use the fact that 
    # modern Python 3.10+ has int.bit_count() which is very fast (C implementation).
    
    bitsets = [0] * N
    for i in range(N):
        # C_bool[i] is a boolean array of length N
        # We need to pack it into an integer.
        # We can do this by iterating or using numpy packbits.
        # np.packbits(C_bool[i]) gives bytes.
        # Then int.from_bytes(..., 'little')
        b = np.packbits(C_bool[i])
        bitsets[i] = int.from_bytes(b, 'little')
        
    # 3. Simulated Annealing / Local Search
    best_set = []
    best_size = 0
    
    def get_conflicts(S_bits, S_indices):
        # S_bits is the bitset of the current set
        # S_indices is the list of indices in the set
        # Return number of conflicts (edges within S)
        # Sum of popcount(bitsets[i] & S_bits) for i in S_indices
        # This counts each edge twice.
        count = 0
        for i in S_indices:
            count += (bitsets[i] & S_bits).bit_count()
        return count // 2

    def add_vertex(S_bits, S_indices, v):
        S_bits |= (1 << v)
        S_indices.append(v)
        return S_bits, S_indices

    def remove_vertex(S_bits, S_indices, v):
        S_bits &= ~(1 << v)
        S_indices.remove(v) # This is O(N). Can optimize with a set or dict.
        return S_bits, S_indices

    # To make remove fast, we can maintain a set of indices.
    # But we need to iterate. A list is fine for ~500 elements.
    
    # We will run multiple restarts
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize S with a random greedy set
        # Greedy: pick random order, add if no conflict
        order = np.random.permutation(N)
        S_bits = 0
        S_indices = []
        
        for v in order:
            if time.time() > deadline - 5:
                break
            # Check conflict: (bitsets[v] & S_bits) == 0
            if (bitsets[v] & S_bits) == 0:
                S_bits |= (1 << v)
                S_indices.append(v)
                
        current_size = len(S_indices)
        current_conflicts = 0 # Should be 0 for greedy
        
        # Local Search: Try to improve size
        # Strategy: 
        # 1. Try to add any vertex with 0 conflicts.
        # 2. If stuck, try to swap: remove u, add v.
        #    Accept if it reduces conflicts or allows future adds.
        #    Since we start with 0 conflicts, we want to maintain 0 conflicts while increasing size.
        #    This is the Independent Set problem.
        
        # Better Strategy for IS:
        # While there is a vertex v not in S with 0 conflicts with S:
        #   Add v
        # If none, try to remove a vertex u from S such that it "unblocks" the most vertices.
        # i.e., maximize the number of v not in S that have 0 conflicts with S \ {u}.
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add
            # Find all v not in S such that (bitsets[v] & S_bits) == 0
            # This is O(N).
            added = False
            # To speed up, we can maintain a list of "available" vertices?
            # No, just scan.
            for v in range(N):
                if time.time() > deadline - 5:
                    break
                if not (S_bits >> v) & 1: # v not in S
                    if (bitsets[v] & S_bits) == 0:
                        S_bits |= (1 << v)
                        S_indices.append(v)
                        improved = True
                        added = True
                        # Continue scanning to add more if possible
                        # But adding one might block others, so we must re-scan or be careful.
                        # Since we only add if 0 conflicts, adding v doesn't create conflicts with existing S.
                        # So we can continue scanning.
            
            if not added:
                # 2. Try to swap
                # For each u in S, calculate how many v not in S become "available" if u is removed.
                # A vertex v is available if (bitsets[v] & S_bits) == 0.
                # If we remove u, S_bits becomes S_bits ^ (1 << u).
                # v becomes available if (bitsets[v] & (S_bits ^ (1 << u))) == 0.
                # This implies (bitsets[v] & S_bits) must be 0 or only have bit u set.
                # i.e., bitsets[v] & S_bits is either 0 or (1 << u).
                
                # We want to find u that maximizes the number of such v.
                # This is O(|S| * N).
                # |S| ~ 500, N = 5280. 2.6M ops. Fast.
                
                best_u = -1
                best_count = 0
                best_v = -1
                
                for u in S_indices:
                    # Calculate potential gains
                    # v is a candidate if v not in S and (bitsets[v] & S_bits) in {0, 1<<u}
                    # We can precompute for each v, what is its conflict mask with S?
                    # Let's just iterate v for each u? That's O(|S|^2 * N) which is too slow.
                    
                    # Alternative:
                    # For each v not in S, let mask_v = bitsets[v] & S_bits.
                    # If mask_v == 0, v can be added immediately (handled in step 1).
                    # If mask_v has exactly 1 bit set, say bit u, then removing u allows adding v.
                    # So for each u, the number of v that can be added by removing u is the count of v not in S such that (bitsets[v] & S_bits) == (1 << u).
                    
                    # We can compute this for all u in one pass over v not in S.
                    gains = {u: 0 for u in S_indices}
                    
                    for v in range(N):
                        if (S_bits >> v) & 1:
                            continue
                        mask = bitsets[v] & S_bits
                        if mask == 0:
                            # Should have been added in step 1
                            pass
                        elif mask & (mask - 1) == 0: # Power of 2
                            # Find which bit
                            u = mask.bit_length() - 1
                            if u in gains:
                                gains[u] += 1
                                if gains[u] > best_count:
                                    best_count = gains[u]
                                    best_u = u
                                    best_v = v # We just need one v to add, or we can add all?
                                    # If we remove u, we can add ALL v with mask == 1<<u.
                                    # So the gain is gains[u].
                                    
                    if best_u != -1 and best_count > 0:
                        # Remove best_u
                        S_bits &= ~(1 << best_u)
                        S_indices.remove(best_u)
                        
                        # Add all v that are now available?
                        # We identified best_v, but there might be others.
                        # Let's just re-run the "Try to add" loop.
                        improved = True
                        # To be efficient, we can just set improved=True and let the outer loop handle the adds.
                        # But we might want to add multiple at once.
                        # For now, just one swap per iteration.
                        
        # Update best
        if len(S_indices) > best_size:
            best_size = len(S_indices)
            best_set = S_indices.copy()
            
    if best_size == 0:
        # Fallback to weight 2
        points2 = []
        for i in range(d):
            for j in range(i + 1, d):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        v = np.zeros(d, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        points2.append(v)
        return np.array(points2, dtype=np.int64)
        
    result = pool[best_set]
    return np.around(result).astype(np.int64)
