import time
import random

N = 9
NUM_VERTICES = 1 << N

# Precompute neighbor masks
nbr_masks = [0] * NUM_VERTICES
for v in range(NUM_VERTICES):
    m = 0
    for b in range(N):
        m |= 1 << (v ^ (1 << b))
    nbr_masks[v] = m

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_path = []
    
    def beam_search(seed, beam_width=200):
        random.seed(seed)
        
        # State: (path_list, used_mask, disallowed_mask)
        # used_mask: bitmask of vertices in the path
        # disallowed_mask: bitmask of vertices that are neighbors of non-consecutive vertices in the path
        #               i.e., neighbors of path[0], path[1], ..., path[-2]
        
        # Initialize with vertex 0
        beam = [([0], 1, 0)]
        
        current_best_len = 1
        
        while beam and time.time() < deadline:
            next_beam = []
            
            for path, used, disallowed in beam:
                last_v = path[-1]
                
                # Candidates are neighbors of last_v that are not used and not disallowed
                cand_mask = nbr_masks[last_v] & ~used & ~disallowed
                
                if cand_mask == 0:
                    # Dead end
                    continue
                
                # Expand all candidates
                cm = cand_mask
                while cm:
                    lsb = cm & -cm
                    n = lsb.bit_length() - 1
                    
                    new_path = path + [n]
                    new_used = used | lsb
                    # The new disallowed set includes neighbors of the new 'last_v' (which was last_v in old state)
                    # Because last_v becomes non-consecutive to the vertex after n
                    new_disallowed = disallowed | nbr_masks[last_v]
                    
                    next_beam.append((new_path, new_used, new_disallowed))
                    cm ^= lsb
            
            if not next_beam:
                break
            
            # Score and prune
            # Heuristic: number of available moves from the new last vertex
            scored_beam = []
            for path, used, disallowed in next_beam:
                last_v = path[-1]
                avail_mask = nbr_masks[last_v] & ~used & ~disallowed
                score = avail_mask.bit_count()
                
                # Add small random noise to break ties and encourage diversity
                noise = random.random() * 0.5
                scored_beam.append((score + noise, path, used, disallowed))
            
            # Sort by score descending
            scored_beam.sort(key=lambda x: x[0], reverse=True)
            
            # Keep top beam_width
            beam = [(p, u, d) for s, p, u, d in scored_beam[:beam_width]]
            
            # Update best length
            if len(beam[0][0]) > current_best_len:
                current_best_len = len(beam[0][0])
                
        # Return the longest path found in this seed
        if beam:
            return beam[0][0]
        return []

    # Run multiple seeds with varying beam widths
    seed = 0
    while time.time() < deadline:
        # Vary beam width to explore different tradeoffs
        if seed % 10 == 0:
            bw = 150
        elif seed % 10 == 1:
            bw = 200
        elif seed % 10 == 2:
            bw = 250
        else:
            bw = 200
            
        p = beam_search(seed, bw)
        if len(p) > len(best_path):
            best_path = p
        seed += 1
        
    return best_path
