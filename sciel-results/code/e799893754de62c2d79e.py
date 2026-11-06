"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
SIZE = 1 << N
DEADLINE = 145.0  # safety margin before 150s kill

# Precompute neighbors for O(1) validity checks
_NEIGHBORS = None
def _get_neighbors():
    global _NEIGHBORS
    if _NEIGHBORS is None:
        _NEIGHBORS = np.zeros((SIZE, N), dtype=np.int32)
        for v in range(SIZE):
            for b in range(N):
                _NEIGHBORS[v, b] = v ^ (1 << b)
    return _NEIGHBORS

def _is_valid_extension(candidate, used_mask, path_arr, path_len):
    """Check if candidate can be appended to path of length path_len.
    Valid if: not used, and not adjacent to any vertex except the last one."""
    if used_mask[candidate]:
        return False
    # Check adjacency to all previous vertices (indices 0..path_len-2)
    # The last vertex (index path_len-1) is allowed to be adjacent
    for i in range(path_len - 1):
        if (candidate ^ path_arr[i]).bit_count() == 1:
            return False
    return True

def _beam_search(deadline, beam_width=50, max_steps=200):
    """Beam search for longest induced path in Q_9."""
    neighbors = _get_neighbors()
    start_time = time.time()
    
    # Each beam entry: (path_list, used_mask)
    # We track paths as lists for simplicity, with numpy masks for speed
    beams = [([0], np.zeros(SIZE, dtype=bool))]
    beams[0][1][0] = True
    
    best_path = beams[0][0]
    
    step = 0
    while step < max_steps:
        if time.time() > deadline:
            break
        
        new_beams = []
        seen_candidates = set()
        
        for path, used_mask in beams:
            cur = path[-1]
            path_len = len(path)
            
            # Try all 9 neighbors
            for b in range(N):
                cand = neighbors[cur, b]
                if cand in seen_candidates:
                    continue
                if used_mask[cand]:
                    continue
                
                # Check if cand is adjacent to any vertex in path[:-1]
                valid = True
                for i in range(path_len - 1):
                    if (cand ^ path[i]).bit_count() == 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                seen_candidates.add(cand)
                new_path = path + [cand]
                new_mask = used_mask.copy()
                new_mask[cand] = True
                new_beams.append((new_path, new_mask))
        
        if not new_beams:
            break
        
        # Sort by length (all same length) then by some heuristic - use path[-1] popcount or just take first beam_width
        # Better: keep diverse paths. Use a simple scoring: prefer paths where the last vertex has more available extensions
        # For now, just keep the first beam_width (or all if fewer)
        if len(new_beams) > beam_width:
            # Simple diversity: sample from new_beams
            random.shuffle(new_beams)
            new_beams = new_beams[:beam_width]
        
        beams = new_beams
        best_path = max(beams, key=lambda x: len(x[0]))[0]
        step += 1
        
        if time.time() > deadline:
            break
    
    return best_path

def construct():
    start_time = time.time()
    
    # First, run the original greedy to get a baseline
    greedy_path = [0]
    greedy_used = {0}
    while True:
        cur, nxt = greedy_path[-1], None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in greedy_used or any((v ^ p).bit_count() == 1 for p in greedy_path[:-1]):
                continue
            nxt = v
            break
        if nxt is None:
            break
        greedy_path.append(nxt)
        greedy_used.add(nxt)
    
    best_path = greedy_path
    
    # Try beam search with different parameters
    for beam_width in [20, 50, 100, 200]:
        if time.time() > DEADLINE - 5:
            break
        try:
            bs_path = _beam_search(DEADLINE - 5, beam_width=beam_width)
            if len(bs_path) > len(best_path):
                best_path = bs_path
        except Exception:
            continue
    
    return best_path
# EVOLVE-BLOCK-END
