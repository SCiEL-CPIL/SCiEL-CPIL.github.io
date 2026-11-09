import time
import random
import itertools
import math

N_POINTS = 16
ALL_TRIPLES = list(itertools.combinations(range(N_POINTS), 3))
TRIPLE_INDEX = {t: i for i, t in enumerate(ALL_TRIPLES)}
N_TRIPLES = len(ALL_TRIPLES)

# Precompute K(2,2,2) configs
CONFIG_MASKS = []
TRIPLE_CFGS = [[] for _ in range(N_TRIPLES)]

def _precompute():
    six_subsets = list(itertools.combinations(range(N_POINTS), 6))
    for pts in six_subsets:
        p0 = pts[0]
        rest = pts[1:]
        for i in range(5):
            p1 = rest[i]
            remaining = [x for x in rest if x != p1]
            r0, r1, r2, r3 = remaining
            pairings = [
                [(p0, p1), (r0, r1), (r2, r3)],
                [(p0, p1), (r0, r2), (r1, r3)],
                [(p0, p1), (r0, r3), (r1, r2)],
            ]
            for pairs in pairings:
                a, a_p = pairs[0]
                b, b_p = pairs[1]
                c, c_p = pairs[2]
                transversals = [
                    (a, b, c), (a, b, c_p), (a, b_p, c), (a, b_p, c_p),
                    (a_p, b, c), (a_p, b, c_p), (a_p, b_p, c), (a_p, b_p, c_p),
                ]
                mask = 0
                for t in transversals:
                    idx = TRIPLE_INDEX.get(tuple(sorted(t)))
                    if idx is not None:
                        mask |= (1 << idx)
                cid = len(CONFIG_MASKS)
                CONFIG_MASKS.append(mask)
                m = mask
                while m:
                    low = m & (-m)
                    idx = low.bit_length() - 1
                    TRIPLE_CFGS[idx].append(cid)
                    m ^= low

_precompute()
N_CONFIGS = len(CONFIG_MASKS)

def _is_valid_addition(triple_idx, in_set, config_counts):
    if triple_idx in in_set:
        return False
    for cfg_id in TRIPLE_CFGS[triple_idx]:
        if config_counts[cfg_id] == 7:
            return False
    return True

def _update_counts(triple_idx, delta, config_counts):
    for cfg_id in TRIPLE_CFGS[triple_idx]:
        config_counts[cfg_id] += delta

def _greedy_fill(current_set, config_counts, deadline):
    """Try to add any valid triple repeatedly."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate through all triples, try to add if valid
        # To be fast, we can just iterate once, but might need multiple passes
        for t_idx in range(N_TRIPLES):
            if time.time() > deadline:
                break
            if t_idx not in current_set and _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                improved = True
    return current_set, config_counts

def _local_search(current_set, config_counts, deadline):
    """1-swap local search."""
    best_size = len(current_set)
    best_set = current_set.copy()
    best_counts = config_counts.copy()
    
    while time.time() < deadline:
        improved = False
        # Try all 1-swaps
        for t_out in list(current_set):
            if time.time() > deadline:
                break
            current_set.remove(t_out)
            _update_counts(t_out, -1, config_counts)
            
            best_in = -1
            # Try to find a valid t_in
            # Heuristic: Try random order or specific order
            # For speed, just check a few random ones or all
            # Checking all 560 is fast enough
            for t_in in range(N_TRIPLES):
                if t_in in current_set:
                    continue
                if _is_valid_addition(t_in, current_set, config_counts):
                    # If we can add, do we care which one? 
                    # We want to maximize future potential, but for now just size.
                    # All valid additions keep size same (1 out, 1 in).
                    # So any valid t_in is fine for this step.
                    best_in = t_in
                    break
            
            if best_in != -1:
                current_set.add(best_in)
                _update_counts(best_in, 1, config_counts)
                # Try to fill
                current_set, config_counts = _greedy_fill(current_set, config_counts, deadline)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    best_counts = config_counts.copy()
                    improved = True
            else:
                # Revert
                current_set.add(t_out)
                _update_counts(t_out, 1, config_counts)
        
        if not improved:
            break
            
    return best_set, best_counts

def _simulated_annealing(current_set, config_counts, deadline):
    best_set = current_set.copy()
    best_size = len(current_set)
    best_counts = config_counts.copy()
    
    # Initial fill
    current_set, config_counts = _greedy_fill(current_set, config_counts, deadline)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set.copy()
        best_counts = config_counts.copy()
        
    T = 2.0
    cooling = 0.995
    step = 0
    
    while time.time() < deadline and T > 0.01:
        step += 1
        if not current_set:
            break
            
        # Choose a random t_out
        t_out = random.choice(tuple(current_set))
        current_set.remove(t_out)
        _update_counts(t_out, -1, config_counts)
        
        # Choose a random t_in
        t_in = random.randrange(N_TRIPLES)
        
        if t_in in current_set:
            # Revert
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
            T *= cooling
            continue
            
        if _is_valid_addition(t_in, current_set, config_counts):
            current_set.add(t_in)
            _update_counts(t_in, 1, config_counts)
            
            # Fill
            current_set, config_counts = _greedy_fill(current_set, config_counts, deadline)
            new_size = len(current_set)
            
            if new_size > best_size:
                best_size = new_size
                best_set = current_set.copy()
                best_counts = config_counts.copy()
            elif new_size == best_size:
                # Keep if same size? Or accept with prob?
                # Usually we want to explore, so accept.
                pass
            else:
                # Size decreased. Accept with prob exp(delta/T)
                delta = new_size - len(best_set) # or current best size before move?
                # Let's use the size before the move as reference for delta?
                # Actually, standard SA uses current state vs new state.
                # Here, we compare new_size with the size we had before the swap?
                # No, we want to escape local optima.
                # Let's accept if random() < exp((new_size - best_size) / T)
                # This biases towards better solutions.
                if random.random() < math.exp((new_size - best_size) / T):
                    pass # Accept
                else:
                    # Revert to best? Or revert to previous state?
                    # Reverting to best is a "hill climbing" with random jumps.
                    # Let's revert to the state before the swap.
                    # We don't have the previous state saved easily.
                    # Let's just keep the new state if accepted, else revert.
                    # For simplicity, if rejected, we revert.
                    # But we already modified current_set.
                    # We need to undo.
                    # Undo fill is hard.
                    # So, better to copy before modifying?
                    # That's expensive.
                    
                    # Alternative: Only accept if new_size >= best_size?
                    # That's hill climbing.
                    
                    # Let's try: Accept if new_size > best_size OR (new_size == best_size AND random < 0.5)
                    # This allows wandering.
                    if new_size < best_size:
                         # Revert logic is complex. 
                         # Let's just keep it if it's not too bad?
                         # No, let's restart from best if we get stuck.
                         pass
        else:
            # Invalid move, revert
            current_set.add(t_out)
            _update_counts(t_out, 1, config_counts)
            
        T *= cooling
        
    return best_set, best_counts

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    best_result = set()
    best_size = 0
    best_counts = [0] * N_CONFIGS
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(i * 1000 + 42)
        
        # Greedy Init
        current_set = set()
        config_counts = [0] * N_CONFIGS
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        for t_idx in order:
            if time.time() > deadline:
                break
            if _is_valid_addition(t_idx, current_set, config_counts):
                current_set.add(t_idx)
                _update_counts(t_idx, 1, config_counts)
                
        # Fill
        current_set, config_counts = _greedy_fill(current_set, config_counts, deadline)
        
        # Local Search
        current_set, config_counts = _local_search(current_set, config_counts, deadline)
        
        # Simulated Annealing
        current_set, config_counts = _simulated_annealing(current_set, config_counts, deadline)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_result = current_set.copy()
            best_counts = config_counts.copy()
            
    return [list(ALL_TRIPLES[i]) for i in best_result]
