"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We aim for 50s max to be safe.
    deadline = start_time + 50

    def get_nz(poly):
        """Get list of indices of non-zero terms."""
        return [i for i in range(len(poly)) if poly[i] != 0]

    def init_square(poly):
        """Calculate initial square coefficients and count."""
        nz = get_nz(poly)
        n = len(poly)
        sq = [0] * (2 * n)
        for i in nz:
            for j in nz:
                sq[i + j] += poly[i] * poly[j]
        count = sum(1 for c in sq if c != 0)
        return sq, count

    def construct():
        # Local variables for speed
        poly = [0] * 41
        # We need exactly 12 non-zero terms.
        
        # Helper to update square incrementally
        # State: sq (list of 81), count (int)
        
        def try_move_coeff(idx, new_val, poly, sq, count):
            """Update sq and count if we change poly[idx] to new_val. Returns (new_sq, new_count, success)."""
            if new_val == poly[idx]:
                return sq, count, False
            
            old_val = poly[idx]
            delta_coeff = new_val - old_val
            
            # Update sq
            # Terms affected: idx + k for all k in nz
            nz = get_nz(poly) # This is O(41). We can optimize by keeping nz list, but let's see.
            # Actually, for incremental update, we need the current nz list.
            # Let's pass nz list or compute it. Computing is fast enough for N=41.
            
            changed_indices = set()
            
            # Diagonal term: idx + idx
            old_sq_val = sq[2 * idx]
            new_sq_val = old_sq_val + (new_val * new_val - old_val * old_val)
            sq[2 * idx] = new_sq_val
            changed_indices.add(2 * idx)
            
            # Off-diagonal terms: idx + k
            for k in nz:
                if k == idx: continue
                pos = idx + k
                # Contribution change: 2 * (new_val * poly[k] - old_val * poly[k]) = 2 * delta_coeff * poly[k]
                delta = 2 * delta_coeff * poly[k]
                old_sq_val = sq[pos]
                new_sq_val = old_sq_val + delta
                sq[pos] = new_sq_val
                changed_indices.add(pos)
                
            # Update count
            # For each changed index, check if it crossed zero
            for pos in changed_indices:
                if sq[pos] == 0:
                    # Was non-zero, now zero?
                    # We need to know the OLD value to know if it was non-zero.
                    # This incremental count is tricky without storing old sq values or recalculating count.
                    # Given N=12, O(N^2) full recalc is ~144 ops.
                    # Incremental update with count tracking is complex.
                    # Let's stick to full recalc for simplicity and correctness, but optimize the inner loop.
                    pass
            
            return sq, count, True # Placeholder, we will use full recalc in the main loop for safety

    # Since incremental count is error-prone to implement quickly and correctly under time pressure,
    # and O(12^2) = 144 multiplications is very fast in Python (microseconds),
    # we will use a highly optimized full recalculation function.
    
    def square_nonzero_count(poly):
        nz = [i for i in range(len(poly)) if poly[i] != 0]
        n = len(nz)
        if n == 0: return 0
        sq = [0] * (2 * len(poly))
        # Unroll or use local vars
        for ii in range(n):
            i = nz[ii]
            ai = poly[i]
            for jj in range(ii, n):
                j = nz[jj]
                aj = poly[j]
                prod = ai * aj
                if ii == jj:
                    sq[i + j] += prod
                else:
                    sq[i + j] += 2 * prod
        count = 0
        for c in sq:
            if c != 0:
                count += 1
        return count

    def simulate_annealing(poly, max_iter, t_start, t_end, deadline):
        current = poly[:]
        current_score = square_nonzero_count(current)
        best = current[:]
        best_score = current_score
        temp = t_start
        
        # Pre-compute zero positions? No, positions change.
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            nz = [i for i in range(len(current)) if current[i] != 0]
            if len(nz) != 12:
                # Repair if necessary (shouldn't happen if moves are valid)
                break
            
            move_type = random.random()
            old_state = None # For revert
            
            if move_type < 0.5:
                # Change coefficient
                idx = random.choice(nz)
                old_val = current[idx]
                new_val = random.randint(1, 64)
                if random.random() < 0.5:
                    new_val = -new_val
                if new_val == old_val:
                    continue
                current[idx] = new_val
                old_state = (idx, old_val)
                
            elif move_type < 0.8:
                # Move term
                idx = random.choice(nz)
                coeff = current[idx]
                # Find a random empty slot
                # List of empty slots is 41 - 12 = 29.
                # Instead of building list, pick random pos and check
                for _ in range(10): # Try a few times
                    new_pos = random.randint(0, 40)
                    if current[new_pos] == 0:
                        break
                else:
                    continue # No empty slot found in 10 tries (unlikely)
                
                current[idx] = 0
                current[new_pos] = coeff
                old_state = (idx, new_pos, coeff)
                
            else:
                # Swap two coefficients
                i1 = random.choice(nz)
                i2 = random.choice(nz)
                if i1 == i2: continue
                v1 = current[i1]
                v2 = current[i2]
                current[i1] = v2
                current[i2] = v1
                old_state = (i1, i2, v1, v2)
                
            new_score = square_nonzero_count(current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                delta = new_score - current_score
                if delta > 0 and random.random() < math.exp(-delta / max(temp, 1e-10)):
                    current_score = new_score
                else:
                    # Revert
                    if old_state:
                        if len(old_state) == 2: # Coeff change
                            idx, old_val = old_state
                            current[idx] = old_val
                        elif len(old_state) == 3: # Move
                            idx, new_pos, coeff = old_state
                            current[new_pos] = 0
                            current[idx] = coeff
                        elif len(old_state) == 4: # Swap
                            i1, i2, v1, v2 = old_state
                            current[i1] = v1
                            current[i2] = v2
                            
            temp *= 0.999
            if temp < t_end:
                temp = t_end
                
        return best, best_score

    # Initial Candidates
    best_poly = [0] * 41
    best_score = float('inf')
    
    # 1. Alternating AP (degree 11)
    p1 = [0] * 41
    for i in range(12):
        p1[i] = 1 if i % 2 == 0 else -1
    s1 = square_nonzero_count(p1)
    if s1 < best_score:
        best_score = s1
        best_poly = p1[:]
        
    # 2. Alternating AP with gap 2 (degree 22)
    p2 = [0] * 41
    for i in range(12):
        p2[2*i] = 1 if i % 2 == 0 else -1
    s2 = square_nonzero_count(p2)
    if s2 < best_score:
        best_score = s2
        best_poly = p2[:]
        
    # 3. Random starts
    restarts = 0
    while time.time() < deadline - 5:
        positions = random.sample(range(41), 12)
        poly = [0] * 41
        for pos in positions:
            c = random.randint(1, 64)
            if random.random() < 0.5:
                c = -c
            poly[pos] = c
            
        # Run SA
        poly, score = simulate_annealing(poly, max_iter=5000, t_start=1.0, t_end=0.01, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_poly = poly[:]
            
        restarts += 1
        if restarts > 100: # Cap restarts
            break
            
    # Final refinement
    if time.time() < deadline - 2:
        best_poly, best_score = simulate_annealing(best_poly, max_iter=10000, t_start=0.5, t_end=0.001, deadline=deadline)
        
    # Trim and Validate
    last_nz = -1
    for i in range(len(best_poly) - 1, -1, -1):
        if best_poly[i] != 0:
            last_nz = i
            break
            
    if last_nz == -1:
        return [1] + [0]*10 + [1] # Fallback
        
    result = best_poly[:last_nz + 1]
    # Ensure exactly 12 non-zero
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != 12:
        return [1] + [0]*10 + [1]
        
    return result

# EVOLVE-BLOCK-END
