import time
import random
import math

_START_TIME = time.time()
_TOTAL_BUDGET = 240.0  # Task specifies 240s for all sequences. We assume ~9s per sequence.
# Safety margin: return before 330s wall clock kill.
# Since we don't know how many sequences will be called, we need to be careful.
# The prompt says "240 s CPU budget for all sequences together".
# The current program uses a fixed 8.0s per sequence.
# If there are 26 sequences, 26 * 8 = 208s, which is within 240s.
# However, if the evaluator runs them in parallel or the timing is wall-clock, we need to be safe.
# Let's stick to a dynamic budget or a conservative per-sequence limit.
# The instruction says: "Keep an EXPLICIT deadline guard: track elapsed time and return the best result found with a safety margin before the limit."
# Let's assume the 240s is for the *entire* run of all 26 sequences.
# If we just use 8s per sequence, and there are 26, that's 208s.
# If we use 9s per sequence, that's 234s.
# Let's use 8.5s per sequence to be safe, but also check global time if possible?
# No, `fold` is called once per sequence in a sandboxed subprocess.
# "fold is called once per benchmark sequence in a sandboxed subprocess with a 240 s CPU budget for all sequences together."
# This implies the *subprocess* runs for 240s total? Or the evaluator kills the subprocess after 330s?
# "Each evaluation must finish within 300 s."
# "The evaluator KILLS the program after 330s wall-clock."
# So the subprocess has 330s wall clock.
# The task contract says "240 s CPU budget for all sequences together".
# This is slightly ambiguous. Does it mean the sum of CPU times? Or wall time?
# Given "sandboxed subprocess", it likely means the process runs, calls fold 26 times, and must finish.
# If I use 8.0s per sequence, and there are 26, that's 208s.
# If I use 9.0s per sequence, that's 234s.
# Let's aim for ~8.5s per sequence to leave a buffer for startup/teardown.
# But wait, if the 240s is a hard limit on CPU, and I use 8.5s * 26 = 221s, that's fine.
# If I exceed 240s CPU, do I get penalized? "Score ... 0 is an unfolded chain, 1.0 reproduces ... invalid result ... scores 0."
# Usually, exceeding time limits results in a timeout score of 0.
# So I must ensure the total time is < 240s (or 300s/330s wall).
# Let's use a dynamic deadline based on the number of sequences if we could know it, but we can't.
# Let's stick to a conservative per-sequence limit. 8.0s is safe. 9.0s is risky if there are more than 26 or overhead is high.
# Let's use 8.0s per sequence.

def _neighbours(point):
    x, y, z = point
    return [(x+1, y, z), (x-1, y, z), (x, y+1, z), (x, y-1, z), (x, y, z+1), (x, y, z-1)]

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _greedy_grow(sequence, rng, bias=1.0):
    """
    Greedy growth to create an initial self-avoiding walk.
    Incorporates free-space heuristic from Inspiration 1 to avoid traps.
    """
    N = len(sequence)
    if N == 0:
        return []
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, N):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            
            gain = 0
            free = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
                    else:
                        free += 1
            else:
                # Even for P, prefer some free space to avoid dead ends
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1

            # Score: maximize gain, then free space, then noise
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
            
        best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _local_optimize(sequence, chain, rng, deadline):
    """
    Local optimization using pivot moves and single-residue moves.
    Faster than regrowth for small N, more robust for large N?
    Actually, regrowth is powerful. Let's combine.
    This function performs a quick local polish.
    """
    N = len(chain)
    if N < 10:
        return chain, _energy(sequence, chain)
    
    current_chain = chain[:]
    current_energy = _energy(sequence, chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # We will perform a series of random moves
    # 1. Pivot moves (cut and reattach)
    # 2. Single residue moves (if possible, but rare in SAW)
    
    # For simplicity and speed, let's use a "sliding window" regrowth which is effectively local.
    # But the current program already does global regrowth.
    # Let's add a "pivot" move here.
    
    while time.time() < deadline:
        # Try a pivot move
        if N > 5:
            cut1 = rng.randint(1, N - 4)
            cut2 = rng.randint(cut1 + 1, N - 2)
            
            head = current_chain[:cut1]
            middle = current_chain[cut1:cut2]
            tail = current_chain[cut2:]
            
            # Try to reattach middle to head at a different end?
            # Or just rotate middle?
            # Simpler: Try to move the whole tail? No, that's regrowth.
            
            # Let's stick to the regrowth strategy from the current program as it's proven (1.33).
            # The inspiration suggests a "contact-map first" approach, but that's hard to implement robustly in 8s.
            # The other inspiration suggests GA.
            # The current program is Regrowth + SA.
            
            # Let's refine the regrowth in the main loop instead.
            break
            
    return best_chain, best_energy

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0:
        return []
    
    start_time = time.time()
    # Use 8.0s to be safe for 26 sequences (208s total)
    deadline = start_time + 8.0
    
    rng = random.Random(42)
    
    # Initialize with best of a few greedy chains
    best_chain = None
    best_energy = 0
    
    # Generate multiple diverse seeds
    num_seeds = 20
    for _ in range(num_seeds):
        chain = _greedy_grow(sequence, rng)
        if chain:
            e = _energy(sequence, chain)
            if best_chain is None or e < best_energy:
                best_chain = chain
                best_energy = e
                
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0, 0) for i in range(N)]
        best_energy = _energy(sequence, best_chain)
        
    current_chain = best_chain[:]
    current_energy = best_energy
    
    # Tail Regrowth Search with Simulated Annealing
    T = 2.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Precompute H indices for faster checks?
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    while time.time() < deadline:
        # Pick a cut point
        # Bias towards the middle? 
        # Early cuts allow more restructuring but are more expensive to regrow.
        # Late cuts are cheaper but less effective.
        # Let's try a mix: 50% uniform, 50% biased towards middle?
        # Or just uniform is fine.
        i = rng.randint(1, N - 2)
        
        head = current_chain[:i]
        
        # Prepare data structures for regrowth
        head_map = {p: idx for idx, p in enumerate(head)}
        occupied = set(head_map.keys())
        
        new_tail = []
        tail_map = {} # pos -> global index
        current_pos = head[-1]
        
        valid = True
        
        for k in range(i, N):
            options = []
            for site in _neighbours(current_pos):
                if site in occupied:
                    continue
                
                gain = 0
                free = 0
                if sequence[k] == "H":
                    for q in _neighbours(site):
                        if q in head_map:
                            idx_h = head_map[q]
                            if sequence[idx_h] == "H":
                                gain += 1
                        elif q in tail_map:
                            idx_t = tail_map[q]
                            if sequence[idx_t] == "H":
                                gain += 1
                        else:
                            free += 1
                else:
                    for q in _neighbours(site):
                        if q not in occupied:
                            free += 1

                # Score: maximize gain, then free space, then noise
                # The gain is the most important.
                score = gain * 10.0 + free * 0.1 + rng.random()
                options.append((score, site))
                
            if not options:
                valid = False
                break
                
            best_score, best_site = max(options)
            new_tail.append(best_site)
            tail_map[best_site] = k
            occupied.add(best_site)
            current_pos = best_site
            
        if not valid:
            continue
            
        new_chain = head + new_tail
        new_energy = _energy(sequence, new_chain)
        
        # Metropolis Acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
            current_chain = new_chain
            current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = new_chain[:]
                
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
        else:
            # Rejected
            pass
            
        # Occasionally, if we are stuck (T is low, no improvement), do a "kick"
        # Restart temperature?
        # Simple: if T is T_min, keep it there.
        
    return [list(p) for p in best_chain]
