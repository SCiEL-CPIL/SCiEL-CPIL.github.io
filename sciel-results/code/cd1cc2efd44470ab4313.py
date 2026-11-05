import random
import time
import math

def _energy(sequence, chain):
    """Compute HP contact energy efficiently using a set of positions."""
    n = len(chain)
    if n == 0:
        return 0
    
    # Map position to index for O(1) lookup
    where = {p: i for i, p in enumerate(chain)}
    
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            # Check 4 neighbors
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                q = (x+dx, y+dy)
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _generate_initial_chain(sequence):
    """
    Generate a compact initial chain using a greedy heuristic with backtracking.
    This aims to maximize early contacts while avoiding dead ends.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    best_chain = None
    best_score = -1
    
    # Try multiple random seeds for the greedy growth
    for _ in range(20):
        chain = [(0, 0)]
        occupied = {(0, 0): 0}
        success = True
        
        for i in range(1, n):
            last_pos = chain[-1]
            prev_pos = chain[-2] if i > 1 else None
            
            candidates = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                # Avoid backtracking
                if i > 1:
                    if (dx, dy) == (prev_pos[0] - last_pos[0], prev_pos[1] - last_pos[1]):
                        continue
                
                new_pos = (last_pos[0] + dx, last_pos[1] + dy)
                if new_pos in occupied:
                    continue
                
                # Calculate score for this move
                # 1. Immediate energy gain (new H-H contacts)
                # 2. Openness (number of free neighbors for the new position)
                
                # Check new contacts
                gain = 0
                if sequence[i] == "H":
                    for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        neighbor = (new_pos[0] + ndx, new_pos[1] + ndy)
                        if neighbor in occupied:
                            j = occupied[neighbor]
                            if sequence[j] == "H" and j != i - 1: # Not the previous residue
                                gain += 1
                
                # Check openness
                openness = 0
                for ndx, ndy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    next_pos = (new_pos[0] + ndx, new_pos[1] + ndy)
                    if next_pos not in occupied:
                        openness += 1
                
                # Score: Weighted sum of gain and openness
                # We prioritize gain, but need enough openness to not get stuck
                score = gain * 10 + openness * 2 + random.random()
                
                candidates.append((score, new_pos))
                
            if not candidates:
                success = False
                break
                
            # Sort by score and pick the best, or randomly from top 3 to allow diversity
            candidates.sort(reverse=True)
            if len(candidates) > 3 and random.random() < 0.2:
                _, new_pos = random.choice(candidates[:3])
            else:
                _, new_pos = candidates[0]
                
            chain.append(new_pos)
            occupied[new_pos] = i
            
        if success and len(chain) == n:
            e = _energy(sequence, chain)
            # We want to maximize e (which is negative, so higher is better)
            if e > best_score:
                best_score = e
                best_chain = chain
                
    if best_chain:
        return best_chain
    else:
        # Fallback: Simple snake
        side = int(math.ceil(math.sqrt(n)))
        chain = []
        for row in range(side):
            if row % 2 == 0:
                for col in range(side):
                    if len(chain) == n:
                        return chain
                    chain.append((col, row))
            else:
                for col in range(side-1, -1, -1):
                    if len(chain) == n:
                        return chain
                    chain.append((col, row))
        return chain

def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0):
    """
    Perform local search using a mix of move types:
    1. Single step move (if space available)
    2. Segment reversal
    3. Pivot move
    """
    n = len(chain)
    if n == 0:
        return chain
    
    current_chain = chain[:]
    current_energy = _energy(sequence, current_chain)
    best_chain = current_chain[:]
    best_energy = current_energy
    
    start_time = time.time()
    
    # Precompute occupied set for fast checks
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    for i in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Choose a move type
        move_type = random.choice(["pivot", "reversal", "step"])
        
        new_chain = None
        new_energy = None
        
        if move_type == "pivot" and n > 2:
            # Pivot move
            k = random.randint(1, n - 2)
            rot = random.choice([1, 2, 3])
            
            pivot_x, pivot_y = current_chain[k]
            head = current_chain[:k+1]
            tail = current_chain[k+1:]
            
            head_set = set(head)
            
            new_tail = []
            valid = True
            for px, py in tail:
                dx = px - pivot_x
                dy = py - pivot_y
                if rot == 1: # 90
                    nx, ny = -dy, dx
                elif rot == 2: # 180
                    nx, ny = -dx, -dy
                else: # 270
                    nx, ny = dy, -dx
                    
                new_pos = (pivot_x + nx, pivot_y + ny)
                if new_pos in head_set:
                    valid = False
                    break
                new_tail.append(new_pos)
                
            if valid:
                new_chain = head + new_tail
                # Check for self-intersection in new_tail? 
                # Rigid rotation preserves internal distances, so no internal collision.
                # But we must ensure new_tail doesn't intersect itself? 
                # Wait, if the tail was self-avoiding, and we rotate it rigidly, it remains self-avoiding.
                # The only risk is collision with the head, which we checked.
                new_energy = _energy(sequence, new_chain)
                
        elif move_type == "reversal" and n > 3:
            # Segment reversal
            # Pick i < j such that reversing chain[i:j+1] is valid
            # For efficiency, pick small segments first, then larger
            # Simple approach: pick random i, j
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            
            # Check if reversal is valid
            # The endpoints of the reversed segment are (i-1, i) and (j, j+1)
            # After reversal, the new endpoints are (i-1, j) and (i, j+1)
            # We need to ensure (i-1, j) is a unit step and (i, j+1) is a unit step
            # And that the reversed segment doesn't collide with the rest of the chain.
            
            if i > 0 and j < n - 1:
                p_im1 = current_chain[i-1]
                p_j = current_chain[j]
                p_i = current_chain[i]
                p_jp1 = current_chain[j+1]
                
                # Check unit steps
                if abs(p_im1[0] - p_j[0]) + abs(p_im1[1] - p_j[1]) != 1:
                    continue
                if abs(p_i[0] - p_jp1[0]) + abs(p_i[1] - p_jp1[1]) != 1:
                    continue
                    
                # Check collision
                # The segment current_chain[i:j+1] is reversed.
                # The internal positions are the same, just in reverse order.
                # So they don't collide with each other.
                # We need to check if any position in current_chain[i:j+1] (excluding i and j, which are now connected to neighbors)
                # collides with the rest of the chain (current_chain[0:i] and current_chain[j+1:n]).
                
                rest_set = set(current_chain[:i]) | set(current_chain[j+1:])
                
                # Actually, the positions themselves are the same. The only change is the order.
                # So the set of occupied positions is identical!
                # Wait, if the set of positions is identical, then there are no new collisions.
                # The only check needed is the connectivity at the boundaries.
                # Which we already did.
                
                new_chain = current_chain[:i] + current_chain[i:j+1][::-1] + current_chain[j+1:]
                new_energy = _energy(sequence, new_chain)
                
        elif move_type == "step" and n > 1:
            # Single step move
            # Pick a random index k (not 0 or n-1 to avoid breaking ends easily, or allow ends too)
            k = random.randint(1, n - 2)
            p_k = current_chain[k]
            
            # Try to move p_k to a neighbor
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_p = (p_k[0] + dx, p_k[1] + dy)
                
                # Check if new_p is occupied by any other node
                if new_p in occupied:
                    # If it's occupied by k-1 or k+1, it's a backtrack, which is invalid for self-avoidance?
                    # No, self-avoidance means no two nodes at same position.
                    # If new_p is occupied by k-1, then moving k to k-1's position is invalid.
                    # So we just check if new_p is in occupied.
                    continue
                    
                # Check if move breaks connectivity
                # k-1 and k+1 must still be unit steps from new_p
                p_km1 = current_chain[k-1]
                p_kp1 = current_chain[k+1]
                
                if abs(p_km1[0] - new_p[0]) + abs(p_km1[1] - new_p[1]) != 1:
                    continue
                if abs(p_kp1[0] - new_p[0]) + abs(p_kp1[1] - new_p[1]) != 1:
                    continue
                    
                # Valid move
                new_chain = current_chain[:]
                new_chain[k] = new_p
                new_energy = _energy(sequence, new_chain)
                break
                
        if new_chain is None:
            continue
            
        delta = new_energy - current_energy
        
        # Simple Hill Climbing with occasional random acceptance to escape
        if delta > 0:
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy > best_energy:
                best_energy = current_energy
                best_chain = current_chain[:]
        else:
            # Accept with probability exp(delta / T) where T is small
            # Since delta is negative, exp(delta/T) is < 1
            # We want to minimize energy (maximize score), so delta > 0 is improvement.
            # Here delta < 0 is worsening.
            # Let's use a small temperature
            T = 0.5
            if random.random() < math.exp(delta / T):
                current_chain = new_chain
                current_energy = new_energy

    return best_chain

def fold(sequence, dimension):
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate initial chain
    current_chain = _generate_initial_chain(sequence)
    current_energy = _energy(sequence, current_chain)
    
    best_chain = current_chain[:]
    best_energy = current_energy
    
    # 2. Iterated Local Search
    while time.time() < deadline - 5.0:
        # Local search
        improved_chain = _local_search(sequence, current_chain, max_iterations=2000, time_limit=5.0)
        improved_energy = _energy(sequence, improved_chain)
        
        if improved_energy > best_energy:
            best_energy = improved_energy
            best_chain = improved_chain[:]
        
        # Perturb the best chain to escape local minimum
        # Use a random pivot move on the best chain
        if len(best_chain) > 2:
            k = random.randint(1, len(best_chain) - 2)
            rot = random.choice([1, 2, 3])
            pivot_x, pivot_y = best_chain[k]
            head = best_chain[:k+1]
            tail = best_chain[k+1:]
            head_set = set(head)
            
            new_tail = []
            valid = True
            for px, py in tail:
                dx = px - pivot_x
                dy = py - pivot_y
                if rot == 1:
                    nx, ny = -dy, dx
                elif rot == 2:
                    nx, ny = -dx, -dy
                else:
                    nx, ny = dy, -dx
                    
                new_pos = (pivot_x + nx, pivot_y + ny)
                if new_pos in head_set:
                    valid = False
                    break
                new_tail.append(new_pos)
                
            if valid:
                current_chain = head + new_tail
            else:
                # If pivot failed, try a random segment reversal
                i = random.randint(0, len(best_chain) - 2)
                j = random.randint(i + 1, len(best_chain) - 1)
                if i > 0 and j < len(best_chain) - 1:
                    p_im1 = best_chain[i-1]
                    p_j = best_chain[j]
                    p_i = best_chain[i]
                    p_jp1 = best_chain[j+1]
                    if abs(p_im1[0] - p_j[0]) + abs(p_im1[1] - p_j[1]) == 1 and \
                       abs(p_i[0] - p_jp1[0]) + abs(p_i[1] - p_jp1[1]) == 1:
                        current_chain = best_chain[:i] + best_chain[i:j+1][::-1] + best_chain[j+1:]
                    else:
                        current_chain = best_chain[:]
                else:
                    current_chain = best_chain[:]
        else:
            current_chain = best_chain[:]
            
    return [list(p) for p in best_chain]
