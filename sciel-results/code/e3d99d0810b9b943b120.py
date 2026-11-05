"""HP lattice protein folding using Monte Carlo Crank Moves."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Global deadline guard
_DEADLINE = None
_BUDGET_FRACTION = 0.95  # Use 95% of the 330s budget to be safe

def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 330 * _BUDGET_FRACTION
    return _DEADLINE

def _is_deadline_exceeded():
    return time.time() > _get_deadline()

def _manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1]) + abs(a[2]-b[2])

def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    """Compute energy as -number of H-H contacts."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(sequence)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _initial_grow(sequence, dimension, rng):
    """Generate an initial self-avoiding walk using randomized greedy growth."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0, 0)]
    
    # Start at origin
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Score: prefer positions with more H-H contacts and more free space
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == "H":
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 2.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick best option
        options.sort(reverse=True)
        site = options[0][1]
        chain.append(site)
        occupied[site] = i
    
    return chain

def _crank_search(sequence, chain, dimension, rng, time_limit=1.0):
    """Improve a chain using Monte Carlo Crank moves."""
    n = len(sequence)
    if n <= 2:
        return chain
    
    # Build position map
    pos_map = {p: i for i, p in enumerate(chain)}
    is_h = [1 if s == "H" else 0 for s in sequence]
    
    def compute_energy():
        contacts = 0
        for i, p in enumerate(chain):
            if is_h[i]:
                for q in _neighbours(p):
                    j = pos_map.get(q)
                    if j is not None and j > i + 1 and is_h[j]:
                        contacts += 1
        return -contacts

    current_energy = compute_energy()
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature schedule
    T_start = 1.5
    T_end = 0.05
    iterations = 8000
    alpha = (T_end / T_start) ** (1.0 / iterations)
    T = T_start
    
    start_time = time.time()
    
    for iteration in range(iterations):
        if _is_deadline_exceeded() or (time.time() - start_time > time_limit):
            break
        
        # Pick a pivot index. Usually internal.
        # Crank moves are most effective for internal pivots.
        if n > 4:
            pivot = rng.randint(2, n - 3)
        else:
            pivot = rng.randint(1, n - 2)
        
        p_pivot = chain[pivot]
        
        # Try all 6 neighbours as new positions for the pivot
        # Note: In a crank move, we rotate the segment [pivot+1..end] around the pivot.
        # However, a simpler and very effective variant is "pivot move" where we 
        # rotate the tail around a pivot.
        # Let's implement the standard pivot move:
        # Pick pivot i. The segment chain[0..i-1] is fixed.
        # We try to move chain[i] to a neighbor of chain[i-1]? No, that's not right.
        # Standard pivot: Pick i. The segment chain[i..n-1] is rotated around chain[i-1].
        # So chain[i] moves to a neighbor of chain[i-1].
        
        # Let's use a "segment pivot" or "crank" where we pick an internal point
        # and rotate the tail.
        
        # Actually, the most robust simple move for SAW is:
        # 1. Pick i (1 to n-1).
        # 2. Try to move chain[i] to a neighbor of chain[i-1] that is not occupied by chain[0..i-2].
        #    But this breaks the chain connectivity if we just move chain[i].
        #    We must move the entire tail chain[i..n-1].
        
        # This is expensive to check for large tails.
        # Alternative: "Crank" move on a specific bond.
        # Let's stick to a simpler, effective move: "Knock-on" or "End move"?
        # No, let's do a proper Pivot Move but optimized.
        
        # Pivot Move:
        # 1. Choose pivot index i (1 <= i < n).
        # 2. The tail is chain[i..n-1].
        # 3. We want to rotate the tail around chain[i-1].
        # 4. This means chain[i] moves to a new position p_new adjacent to chain[i-1].
        # 5. The rest of the tail shifts accordingly.
        # 6. We need to check if the new tail is self-avoiding and doesn't clash with the head.
        
        # To make this fast, we only try pivots where the tail is short?
        # Or we just do it and check.
        
        # Let's try a different move: "Crank" on a 2-step segment?
        # Or just use the "Single Residue Move" from the parent but with better acceptance?
        # The parent used single residue moves which are very restrictive.
        
        # Let's implement a "Tail Move":
        # Pick i. Try to move the entire tail chain[i..n-1] by one step in one of 6 directions.
        # This is a "Ratchet" or "Drift" move.
        
        # Let's implement a "Pivot Move" properly but with a check.
        
        # Optimization: Only try pivots i where i is small or n-i is small?
        # No, that limits search.
        
        # Let's try "Crank Move" on a specific bond (i-1, i).
        # We rotate the tail starting at i around i-1.
        
        # To keep it simple and fast, let's try moving the END of the chain?
        # End moves are cheap but slow to mix.
        
        # Let's go back to the parent's single residue move but improve it?
        # The parent's move was: move chain[i] to a neighbor of chain[i-1] AND chain[i+1].
        # This is a "kink" move or "flip". It's very restrictive.
        
        # Better move: "Pivot" where we rotate the tail.
        # Let's implement a fast check for tail rotation.
        
        # If we rotate the tail chain[i..n-1] around chain[i-1]:
        # The new position of chain[i] must be a neighbor of chain[i-1].
        # Let's pick a direction d.
        # new_chain[i] = chain[i-1] + d
        # For k > i:
        #   new_chain[k] = new_chain[k-1] + (chain[k] - chain[k-1])
        # This preserves the relative geometry of the tail.
        
        # We need to check:
        # 1. new_chain[i] is not in pos_map[0..i-2]
        # 2. new_chain[k] is not in pos_map[0..i-2] for all k > i
        # 3. new_chain[k] is not in new_chain[j] for i <= j < k
        
        # This is O(N) per move.
        
        # Let's try this.
        
        i = rng.randint(1, n - 1)
        
        # If i == n-1, it's just an end move.
        
        head_end = i - 1
        head_sites = [chain[j] for j in range(head_end + 1)]
        head_set = set(head_sites)
        
        # Current tail
        tail = chain[i:]
        
        # Try 6 directions
        # The direction is relative to the bond (chain[i-1], chain[i])?
        # No, we just pick a neighbor of chain[i-1].
        
        for d in _neighbours(chain[head_end]):
            if d in head_set:
                continue
            
            # Construct new tail
            new_tail = [d]
            # Apply the same relative steps as the old tail
            for k in range(1, len(tail)):
                prev_step = (tail[k][0]-tail[k-1][0], tail[k][1]-tail[k-1][1], tail[k][2]-tail[k-1][2])
                new_pos = (new_tail[-1][0] + prev_step[0], 
                           new_tail[-1][1] + prev_step[1], 
                           new_tail[-1][2] + prev_step[2])
                new_tail.append(new_pos)
            
            # Check self-avoidance within new tail
            if len(set(new_tail)) != len(new_tail):
                continue
            
            # Check clash with head
            if any(p in head_set for p in new_tail):
                continue
            
            # Valid move!
            # Calculate energy change
            # Old energy contribution of tail contacts with head and internal tail
            # New energy contribution
            
            # To be fast, we can just recompute full energy?
            # For N=136, full energy is O(N * 6) ~ 800 ops.
            # 8000 iterations * 800 ops = 6.4M ops. Very fast.
            
            # Let's just swap and recompute.
            
            # Save old state
            old_tail = list(tail)
            old_chain = list(chain)
            
            # Apply new tail
            for k in range(i, n):
                chain[k] = new_tail[k-i]
            
            # Update pos_map
            for k in range(i, n):
                del pos_map[old_tail[k-i]]
                pos_map[chain[k]] = k
            
            new_energy = compute_energy()
            delta = new_energy - current_energy
            
            if delta < 0 or rng.random() < math.exp(-delta / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
            else:
                # Revert
                for k in range(i, n):
                    chain[k] = old_tail[k-i]
                for k in range(i, n):
                    del pos_map[chain[k]]
                    pos_map[old_tail[k-i]] = k
            
            T *= alpha
            break # Only one move per iteration
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation using multi-start greedy growth and crank search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    # Multiple restarts with greedy growth + crank search
    # We have ~310s budget.
    # Let's do dynamic restarts.
    
    for restart in range(50):
        if _is_deadline_exceeded():
            break
        
        chain = _initial_grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply crank search
        # Allocate time based on remaining budget
        time_left = _get_deadline() - time.time()
        if time_left < 1.0:
            break
            
        # Give each restart a slice of time, or just run until deadline?
        # If we run until deadline for the first restart, we do only 1 restart.
        # Better: time_limit = min(5.0, time_left / (50 - restart))
        
        # Let's use a fixed time limit per restart, but increase it if we have time?
        # No, keep it simple. 5 seconds per restart. 50 restarts = 250s.
        # This is safe.
        
        chain = _crank_search(sequence, chain, dimension, rng, time_limit=5.0)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
