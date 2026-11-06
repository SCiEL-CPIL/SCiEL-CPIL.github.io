import time

N = 9
TOTAL_VERTS = 1 << N

# Precompute neighbor masks as 512-bit integers
# neighbor_masks[v] has bit w set iff v and w differ in exactly one bit
neighbor_masks = [0] * TOTAL_VERTS
for v in range(TOTAL_VERTS):
    m = 0
    for b in range(N):
        w = v ^ (1 << b)
        m |= (1 << w)
    neighbor_masks[v] = m

def construct():
    beam_width = 50000
    max_steps = 200
    
    # State: (path_tuple, exclusion_mask)
    # Using tuples for immutability and faster hashing
    
    # Start with a fixed prefix to break symmetry and reduce initial branching
    initial_path = (0, 1, 3)
    initial_excl = (1 << 0) | (1 << 1) | (1 << 3) | neighbor_masks[0] | neighbor_masks[1]
    
    beam = [(initial_path, initial_excl)]
    
    best_path = list(initial_path)
    best_len = len(initial_path)
    
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > 115:
            break
            
        next_beam = []
        seen_states = set()
        
        for path, excl in beam:
            cur = path[-1]
            # Bit-parallel: get all valid next moves in one AND operation
            valid = neighbor_masks[cur] & ~excl
            
            c = valid
            while c:
                lb = c & (-c)
                w = lb.bit_length() - 1
                c ^= lb
                
                # New exclusion: add w, and all neighbors of cur (cur is no longer last)
                new_excl = excl | (1 << w) | neighbor_masks[cur]
                
                # Deduplicate based on (last_vertex, exclusion_mask)
                key = (w, new_excl)
                if key not in seen_states:
                    seen_states.add(key)
                    new_path = path + (w,)
                    next_beam.append((new_path, new_excl))
        
        if not next_beam:
            break
        
        # Score: prioritize states with more available moves (greedy)
        # Also consider path length as secondary criterion
        scored_beam = []
        for path, excl in next_beam:
            cur = path[-1]
            valid = neighbor_masks[cur] & ~excl
            score_val = valid.bit_count()
            scored_beam.append((score_val, len(path), path, excl))
        
        # Sort by score descending, then by path length descending
        scored_beam.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        beam = [(path, excl) for _, _, path, excl in scored_beam[:beam_width]]
        
        if len(beam[0][0]) > best_len:
            best_len = len(beam[0][0])
            best_path = list(beam[0][0])
    
    return best_path
