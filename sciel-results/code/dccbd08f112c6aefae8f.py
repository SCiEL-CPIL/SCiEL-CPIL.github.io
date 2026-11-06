"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START
def _is_valid_snake(path):
    """Check if path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return True
    used = set(path)
    if len(used) != n:
        return False
    for i in range(n - 1):
        diff = path[i] ^ path[i + 1]
        if diff & (diff - 1) != 0:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _beam_search_construct(deadline, beam_width=8, restarts=5):
    """Beam search over multiple partial snakes simultaneously."""
    start_time = time.time()
    best_path = []
    best_len = 0
    
    for restart in range(restarts):
        if time.time() > deadline:
            break
        
        random.seed(42 + restart * 1000)
        bit_order = list(range(N))
        random.shuffle(bit_order)
        
        # Initialize beam with diverse starting vertices
        beam = []
        for i in range(beam_width):
            start_v = random.randint(0, TOTAL_VERTICES - 1)
            beam.append({
                'path': [start_v],
                'used': {start_v},
                'bit_order': bit_order[:]
            })
        
        # Expand beam iteratively
        while time.time() < deadline:
            new_beam = []
            for entry in beam:
                if time.time() > deadline:
                    break
                
                path = entry['path']
                used = entry['used']
                bit_order = entry['bit_order']
                cur = path[-1]
                
                # Find all valid extensions
                candidates = []
                for b in bit_order:
                    v = cur ^ (1 << b)
                    if v in used:
                        continue
                    # Check non-adjacency with all but last vertex
                    ok = True
                    for p in path[:-1]:
                        if (v ^ p).bit_count() == 1:
                            ok = False
                            break
                    if ok:
                        candidates.append(v)
                
                if not candidates:
                    # This path is stuck; keep it in beam only if it's the best so far
                    if len(path) > best_len:
                        best_path = path[:]
                        best_len = len(path)
                    continue
                
                # Score candidates by lookahead
                scored_candidates = []
                for v in candidates:
                    # Count how many extensions are possible after v
                    score = 0
                    for b2 in bit_order:
                        v2 = v ^ (1 << b2)
                        if v2 in used or v2 == cur:
                            continue
                        ok2 = True
                        for p in path[:-1]:
                            if (v2 ^ p).bit_count() == 1:
                                ok2 = False
                                break
                        if ok2:
                            score += 1
                    # Add small random perturbation to break ties
                    score += random.random() * 0.01
                    scored_candidates.append((score, v))
                
                # Sort by score descending and take top candidates
                scored_candidates.sort(key=lambda x: -x[0])
                
                # Add top extensions to new beam
                for _, v in scored_candidates[:min(2, len(scored_candidates))]:
                    new_path = path + [v]
                    new_used = used | {v}
                    new_beam.append({
                        'path': new_path,
                        'used': new_used,
                        'bit_order': bit_order[:]
                    })
                    if len(new_path) > best_len:
                        best_path = new_path[:]
                        best_len = len(new_path)
            
            if not new_beam:
                break
            
            # Keep only the best beam_width paths
            new_beam.sort(key=lambda x: -len(x['path']))
            # If all paths have same length, add diversity via random sampling
            if len(new_beam) > beam_width:
                # Keep top beam_width, but ensure some diversity
                top = new_beam[:beam_width // 2]
                remaining = new_beam[beam_width // 2:]
                # Randomly sample from remaining
                sample_size = min(beam_width - len(top), len(remaining))
                if sample_size > 0:
                    indices = random.sample(range(len(remaining)), sample_size)
                    top.extend(remaining[i] for i in indices)
                new_beam = top[:beam_width]
            
            beam = new_beam
        
        # Final check
        if len(best_path) > 0 and _is_valid_snake(best_path):
            return best_path
    
    return best_path

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    result = _beam_search_construct(deadline, beam_width=8, restarts=5)
    
    if result and _is_valid_snake(result):
        return result
    
    # Fallback: simple greedy
    path = [0]
    used = {0}
    while True:
        cur = path[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            ok = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path.append(nxt)
        used.add(nxt)
    
    return path
# EVOLVE-BLOCK-END
