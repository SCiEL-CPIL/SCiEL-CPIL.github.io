# EVOLVE-BLOCK-START
import time
import random
import math

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    if not chain:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i in range(n):
        if sequence[i] == 'H':
            p = chain[i]
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and abs(i - j) > 1:
                    if sequence[j] == 'H':
                        contacts += 1
    return -contacts

def _deterministic_fold(sequence, dimension, time_limit, rng):
    """
    Deterministic Backtracking with Greedy Heuristic.
    Prioritizes moves that create new H-H contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    start = (0, 0, 0)
    chain = [start]
    occupied = {start}
    
    # Pre-compute H positions for quick check
    is_h = [c == 'H' for c in sequence]
    
    def backtrack(idx, current_chain, current_occupied, current_contacts):
        if idx == n:
            return current_chain, current_contacts
        
        last_pos = current_chain[-1]
        neighbors = list(_neighbours(last_pos))
        
        # Shuffle to break symmetry but keep somewhat deterministic order
        # Actually, for determinism, we should sort or use a fixed order
        # but we want to explore promising ones first.
        # We'll sort neighbors by potential contact gain.
        
        scored_neighbors = []
        for nb in neighbors:
            if nb in current_occupied:
                continue
            
            gain = 0
            if is_h[idx]:
                # Check if placing H at nb creates contacts with existing H's
                for q in _neighbours(nb):
                    if q in current_occupied:
                        # Find which index is at q
                        # This is O(N) if we don't have a map. 
                        # Let's maintain a pos_to_idx map? 
                        # For backtracking, maintaining the map is expensive to update/rollback.
                        # Alternative: just count H neighbors in occupied set? 
                        # We need to know if the neighbor is H.
                        # We can store occupied as set of (pos, is_h) ? No, pos is unique.
                        # Let's use a helper to check if a neighbor is H.
                        # Since we are backtracking, we can just check the sequence indices 
                        # if we had a pos->idx map. 
                        # Let's stick to a simpler heuristic: 
                        # If current is H, prefer neighbors that are close to other Hs?
                        # Actually, checking if nb is adjacent to any existing H is O(6) if we 
                        # can quickly check if a point is an H.
                        
                        # Let's maintain a set of H positions: h_occupied
                        pass
                        
            scored_neighbors.append((0, nb)) # Placeholder for gain
            
        # To make it efficient, we need to know if a neighbor is adjacent to an existing H.
        # Let's maintain h_occupied set.
        
        # Re-implementing inner logic for efficiency
        # We'll pass h_occupied as well.
        
        # For now, let's just randomize but with a bias?
        # No, the directive asks for deterministic/different strategy.
        # Let's try a simple greedy: if I place H here, does it touch any H?
        
        # To do this efficiently in backtracking, we need to know which occupied points are H.
        # We can maintain a set `h_occupied`.
        
        # Let's restart the function structure to include h_occupied.
        return None, 0

    # The above inner function is getting complex. 
    # Let's use a simpler "Greedy Growth with Backtracking" approach.
    
    # Actually, a better different strategy: 
    # **Iterative Deepening A* (IDA*)** is too slow for 136 residues.
    # **Deterministic Simulated Annealing** (using a fixed seed and no randomness in acceptance if T=0, i.e., Greedy Hill Climbing with restarts) is similar to current.
    
    # Let's try **Greedy Growth with Local Repair**.
    # 1. Grow the chain greedily: always pick the neighbor that maximizes immediate H-H contacts.
    #    If no contact gain, pick the one with the most free neighbors (to avoid dead ends).
    # 2. If stuck, backtrack.
    
    def greedy_backtrack(idx, current_chain, current_occupied, h_occupied, current_contacts):
        if idx == n:
            return current_chain, current_contacts
        
        last_pos = current_chain[-1]
        neighbors = list(_neighbours(last_pos))
        
        # Filter valid neighbors
        valid_neighbors = []
        for nb in neighbors:
            if nb not in current_occupied:
                valid_neighbors.append(nb)
                
        if not valid_neighbors:
            return None, 0
            
        # Score neighbors
        scored = []
        for nb in valid_neighbors:
            gain = 0
            if is_h[idx]:
                # Check if nb is adjacent to any existing H
                for q in _neighbours(nb):
                    if q in h_occupied:
                        gain += 1
            # Tie-breaker: prefer neighbors with more free space (degree)
            free_space = 0
            for q in _neighbours(nb):
                if q not in current_occupied:
                    free_space += 1
            # We want to maximize gain, then free_space
            # Note: gain is immediate. free_space is heuristic for future.
            scored.append((-gain, -free_space, nb))
            
        scored.sort()
        
        # Try in order of best score
        for _, _, nb in scored:
            new_chain = current_chain + [nb]
            new_occupied = set(current_occupied)
            new_occupied.add(nb)
            
            new_h_occupied = set(h_occupied)
            new_contacts = current_contacts
            if is_h[idx]:
                new_h_occupied.add(nb)
                # Count new contacts
                for q in _neighbours(nb):
                    if q in h_occupied:
                        new_contacts += 1
                        
            result_chain, result_contacts = greedy_backtrack(idx + 1, new_chain, new_occupied, new_h_occupied, new_contacts)
            if result_chain is not None:
                return result_chain, result_contacts
                
        return None, 0

    # Initial call
    # Note: This is exponential. For N=136, it will likely fail or take too long.
    # We need a time limit.
    
    # Since pure backtracking is too slow for long chains, we will use a **Hybrid**:
    # 1. Greedy Growth (fast, no backtracking) to get a valid fold.
    # 2. Local Search (Pivot/Kink) to improve it.
    # But the current program *already* does Random Growth + SA.
    # The difference: **Deterministic Greedy Growth** (no random shuffle, always pick best)
    # followed by **Deterministic Hill Climbing** (always accept improvement, no random acceptance).
    
    # Let's implement **Deterministic Greedy Growth**.
    
    def greedy_grow(sequence, dimension, rng):
        n = len(sequence)
        if n == 0: return []
        if n == 1: return [(0,0,0)]
        
        is_h = [c == 'H' for c in sequence]
        start = (0, 0, 0)
        chain = [start]
        occupied = {start}
        h_occupied = {start} if is_h[0] else set()
        
        for i in range(1, n):
            last_pos = chain[-1]
            neighbors = list(_neighbours(last_pos))
            
            valid_neighbors = []
            for nb in neighbors:
                if nb not in occupied:
                    valid_neighbors.append(nb)
                    
            if not valid_neighbors:
                return None # Stuck
                
            # Score
            best_nb = None
            best_score = -1
            for nb in valid_neighbors:
                gain = 0
                if is_h[i]:
                    for q in _neighbours(nb):
                        if q in h_occupied:
                            gain += 1
                # Tie-breaker: free space
                free_space = 0
                for q in _neighbours(nb):
                    if q not in occupied:
                        free_space += 1
                score = gain * 10 + free_space
                if score > best_score:
                    best_score = score
                    best_nb = nb
                    
            chain.append(best_nb)
            occupied.add(best_nb)
            if is_h[i]:
                h_occupied.add(best_nb)
                
        return chain

    # Phase 1: Greedy Growth
    chain = greedy_grow(sequence, dimension, rng)
    
    if chain is None:
        # Fallback to random if greedy fails (rare but possible)
        chain = [(i, 0, 0) for i in range(n)]
        
    # Phase 2: Deterministic Hill Climbing (Local Search)
    # Use Kink Moves (2-residue flips) and Pivot Moves.
    # Accept if energy improves. Stop if no improvement in X steps.
    
    current_chain = chain
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 5000
    
    while True:
        if time.time() - start_time > time_limit - 0.5:
            break
        if no_improve_count > max_no_improve:
            break
            
        # Try a random move
        # Use Kink moves (flip 2 residues) as they are local and fast
        # Or Pivot moves. Let's stick to Pivot for consistency with seed but deterministic acceptance.
        
        if len(current_chain) < 3:
            break
            
        pivot_idx = rng.randint(1, len(current_chain) - 2)
        
        # Perform Pivot Move
        p_pivot = current_chain[pivot_idx]
        r_axis = rng.choice([0, 1, 2])
        angle = rng.choice([1, -1])
        
        new_tail = []
        valid = True
        occupied_tail = set()
        head_set = set(current_chain[:pivot_idx])
        
        for i in range(pivot_idx + 1, len(current_chain)):
            orig = current_chain[i]
            dx = orig[0] - p_pivot[0]
            dy = orig[1] - p_pivot[1]
            dz = orig[2] - p_pivot[2]
            
            if r_axis == 0:
                if angle == 1:
                    nx, ny, nz = dx, -dz, dy
                else:
                    nx, ny, nz = dx, dz, -dy
            elif r_axis == 1:
                if angle == 1:
                    nx, ny, nz = dz, dy, -dx
                else:
                    nx, ny, nz = -dz, dy, dx
            else:
                if angle == 1:
                    nx, ny, nz = -dy, dx, dz
                else:
                    nx, ny, nz = dy, -dx, dz
                    
            new_point = (p_pivot[0] + nx, p_pivot[1] + ny, p_pivot[2] + nz)
            
            if new_point in head_set or new_point in occupied_tail:
                valid = False
                break
                
            occupied_tail.add(new_point)
            new_tail.append(new_point)
            
        if not valid:
            no_improve_count += 1
            continue
            
        new_chain = current_chain[:pivot_idx + 1] + new_tail
        new_energy = _energy(sequence, new_chain)
        
        if new_energy < current_energy:
            current_chain = new_chain
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_chain = list(new_chain)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
    return best_chain

def fold(sequence, dimension):
    """
    Deterministic Greedy Growth + Hill Climbing.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0, 0]]
    
    total_time_limit = 8.5
    start_time = time.time()
    
    best_chain_overall = None
    best_energy_overall = float('inf')
    
    # Multiple restarts with different random seeds for the "random" parts of hill climbing
    # but the growth is deterministic given the sequence.
    # To get diversity, we can vary the tie-breaking in greedy growth?
    # Or just run Hill Climbing from the same start?
    # Let's run Hill Climbing with different random seeds for move selection.
    
    num_restarts = 3
    time_per_run = total_time_limit / num_restarts
    
    for r in range(num_restarts):
        remaining = total_time_limit - (time.time() - start_time)
        if remaining < 0.5:
            break
            
        rng = random.Random(999 + r)
        
        t_limit = time_per_run if r < num_restarts - 1 else remaining
        chain = _deterministic_fold(sequence, dimension, t_limit, rng)
        
        if chain:
            energy = _energy(sequence, chain)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_chain_overall = chain
                
    if best_chain_overall is None:
        best_chain_overall = [(i, 0, 0) for i in range(n)]
        
    return [list(p) for p in best_chain_overall]

# EVOLVE-BLOCK-END
