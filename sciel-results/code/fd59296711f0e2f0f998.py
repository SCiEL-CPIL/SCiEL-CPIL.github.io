"""HP lattice folding using Monte Carlo Growth and Segment Moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # 210s limit, keep 15s safety margin
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _mc_growth(sequence, dimension, rng, temp=1.0):
    """
    Grow the chain using a Monte Carlo approach.
    At each step, pick a random neighbor of the current end.
    If it's valid, accept it. If not, try another.
    To improve over greedy, we can actually just pick a random valid neighbor,
    but to allow 'backtracking' or 'exploration', we could use a pool of options.
    However, simple random growth is often better than greedy for diversity.
    Let's use a slightly biased random growth: prefer H-H contacts but allow randomness.
    """
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            # Calculate immediate gain
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            options.append((gain, site))
        
        if not options:
            return None
            
        # Select option. 
        # If we want more diversity, we can pick randomly from top k or use softmax.
        # For now, let's pick randomly from all valid options, weighted slightly by gain.
        # This avoids the "greedy trap" where the first good option is always taken.
        
        # Simple random selection from valid options, but weighted by gain to keep it decent
        # Actually, pure random is often better for MC growth to avoid bias.
        # Let's try: pick a random valid site.
        # To balance, let's pick from top 3 if gain > 0, else random.
        
        if options:
            # Sort by gain desc
            options.sort(key=lambda x: x[0], reverse=True)
            # Pick randomly from the top min(3, len(options)) if top gain > 0
            # Otherwise pick randomly from all
            if options[0][0] > 0:
                k = min(3, len(options))
                site = options[rng.randint(0, k-1)][1]
            else:
                site = options[rng.randint(0, len(options)-1)][1]
            
            chain.append(site)
            occupied[site] = i
            
    return chain

def _segment_move(sequence, chain, rng, max_iters=1000):
    """
    Local search using segment moves.
    Pick a segment of the chain (i, j) and try to rotate/translate it to a new valid position.
    This is more powerful than single-residue moves for escaping local minima.
    """
    current = list(chain)
    n = len(current)
    if n < 4:
        return current
    
    where = {p: i for i, p in enumerate(current)}
    current_energy = _energy(sequence, current)
    
    for _ in range(max_iters):
        if _time_remaining() < 5:
            break
            
        # Pick segment length
        # Small segments (2-4) are faster and more likely to find valid moves
        seg_len = rng.randint(2, min(4, n - 2))
        # Pick start index
        start_idx = rng.randint(1, n - seg_len - 1)
        end_idx = start_idx + seg_len - 1
        
        # Extract segment
        seg = current[start_idx:end_idx+1]
        # The neighbors of the segment are start_idx-1 and end_idx+1
        left_neighbor = current[start_idx-1]
        right_neighbor = current[end_idx+1]
        
        # Remove segment from occupied map
        for p in seg:
            del where[p]
        
        # Try to find a new position for the segment
        # The segment must be connected to left_neighbor and right_neighbor.
        # This is a "rotation" or "translation" move.
        # We can try translating the segment by a small vector.
        # Or, more effectively, try to "swing" the segment around one of its endpoints.
        
        # Let's try a "pivot" move: rotate the segment around start_idx or end_idx.
        # Actually, a simpler effective move is to pick a random new position for the whole segment
        # that is adjacent to the left neighbor and allows the chain to continue to the right neighbor.
        # This is hard to sample directly.
        
        # Alternative: Try to move the *entire* segment by a vector (dx, dy) where (dx,dy) is small.
        # This is a "translation" move.
        
        # Let's try translation by (-1, 0), (1, 0), (0, -1), (0, 1)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        rng.shuffle(moves)
        
        moved = False
        for dx, dy in moves:
            new_seg = [(p[0]+dx, p[1]+dy) for p in seg]
            
            # Check if new positions are valid (unoccupied)
            valid = True
            for p in new_seg:
                if p in where:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check connectivity
            # First residue of new_seg must be adjacent to left_neighbor
            if not _is_adjacent(new_seg[0], left_neighbor):
                continue
            # Last residue of new_seg must be adjacent to right_neighbor
            if not _is_adjacent(new_seg[-1], right_neighbor):
                continue
            
            # Check internal connectivity of new_seg (should be preserved by translation)
            # Translation preserves relative distances, so if old seg was connected, new is too.
            # But we must ensure the *whole* chain is still self-avoiding.
            # We already checked new_seg against 'where' (which excludes the segment itself).
            # We also need to ensure new_seg doesn't intersect the rest of the chain.
            # 'where' contains the rest of the chain. So if no p in new_seg is in where, it's fine.
            
            # Apply move
            for i, p in enumerate(new_seg):
                where[p] = start_idx + i
                current[start_idx + i] = p
            
            new_energy = _energy(sequence, current)
            
            if new_energy < current_energy:
                current_energy = new_energy
                moved = True
                break
            else:
                # Revert
                for i, p in enumerate(seg):
                    where[p] = start_idx + i
                    current[start_idx + i] = p
        
        # If no move was made, we might try a more complex move or just continue
        # To avoid getting stuck, we can occasionally accept worse moves (Simulated Annealing style)
        # But for simplicity, let's stick to hill-climbing for now.
        
    return current

def fold(sequence, dimension):
    """
    Fold sequence using MC Growth and Segment Moves.
    """
    _init_time()
    rng = random.Random(54321 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Time budget allocation
    # We want to do many restarts of MC Growth, and refine the best ones.
    
    # Determine restarts based on time
    # MC Growth is O(N^2) roughly.
    # Let's aim for ~50-100 restarts depending on N.
    
    if n <= 20:
        num_restarts = 200
    elif n <= 50:
        num_restarts = 100
    else:
        num_restarts = 50
        
    for restart in range(num_restarts):
        if _time_remaining() < 10:
            break
            
        # 1. MC Growth
        chain = _mc_growth(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Refine with Segment Moves
        # Use a fraction of remaining time for refinement
        # Or fixed iterations
        iters = max(100, 1000 - n * 5)
        chain = _segment_move(sequence, chain, rng, max_iters=iters)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
            
    if best is None:
        # Fallback
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
