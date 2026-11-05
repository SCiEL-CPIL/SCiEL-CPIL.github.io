import time
import random

N = 9
MAX_TIME = 115.0

# Precompute neighbors bitmask for all vertices
NEIGHBOR_MASKS = [0] * (1 << N)
for v in range(1 << N):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    NEIGHBOR_MASKS[v] = mask

def _beam_search(width=50, seed=0):
    """
    Beam search for the longest snake.
    State: (score, path, used_mask, forbidden_mask, current_vertex)
    """
    random.seed(seed)
    
    # Initial state
    initial_path = [0]
    initial_used = 1
    initial_forbidden = 0
    initial_current = 0
    
    # Beam contains tuples: (score, path, used, forbidden, current)
    beam = [(0, initial_path, initial_used, initial_forbidden, initial_current)]
    
    best_path = initial_path
    best_len = len(initial_path)
    
    while beam:
        # Check time
        if time.time() > MAX_TIME:
            break
            
        next_beam = []
        
        for score, path, used, forbidden, current in beam:
            # Get neighbors of current
            nbrs_mask = NEIGHBOR_MASKS[current]
            
            # Valid neighbors: not in used, not in forbidden
            valid_mask = nbrs_mask & ~used & ~forbidden
            
            while valid_mask:
                # Extract lowest set bit
                lsb = valid_mask & -valid_mask
                u = lsb.bit_length() - 1
                valid_mask ^= lsb
                
                # Calculate new state
                new_used = used | lsb
                new_forbidden = forbidden | NEIGHBOR_MASKS[current]
                new_current = u
                new_path = path + [u]
                
                # Score: count valid extensions from new_current
                # Valid extensions from u must not be in new_used and not in new_forbidden
                next_valid_mask = NEIGHBOR_MASKS[new_current] & ~new_used & ~new_forbidden
                
                # Count bits
                next_score = next_valid_mask.bit_count()
                
                # Add to next_beam candidates
                # Random tie-breaker to diversify
                tie_breaker = random.random()
                next_beam.append((next_score, tie_breaker, new_path, new_used, new_forbidden, new_current))
                
                if len(new_path) > best_len:
                    best_len = len(new_path)
                    best_path = new_path
        
        if not next_beam:
            break
            
        # Sort next_beam by score descending, then tie_breaker descending
        next_beam.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Keep top 'width'
        beam = next_beam[:width]
        
    return best_path

def construct():
    start_time = time.time()
    best_path = []
    best_len = 0
    
    # Try different widths and seeds
    seeds = [42, 123, 999, 7, 555, 314, 159, 265, 358, 979]
    widths = [20, 50, 100, 200]
    
    i = 0
    while time.time() - start_time < MAX_TIME:
        # Cycle through widths and seeds
        w = widths[i % len(widths)]
        s = seeds[i % len(seeds)] + i
        
        p = _beam_search(width=w, seed=s)
        
        if len(p) > best_len:
            best_len = len(p)
            best_path = p
            
        i += 1
        
    return best_path
