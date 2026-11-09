import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def square_nz(poly):
        """Compute number of nonzero terms in the exact square."""
        p = np.array(poly, dtype=np.int64)
        sq = np.convolve(p, p)
        return np.count_nonzero(sq)
    
    def is_valid(poly):
        if len(poly) != 41:
            return False
        if poly[40] == 0:
            return False
        nz = sum(1 for x in poly if x != 0)
        if nz != 12:
            return False
        if any(abs(x) > 64 for x in poly):
            return False
        return True
    
    best_score = float('inf')
    best_poly = None
    
    # Strategy 1: Structured families with systematic sign search
    
    # Family A: Palindromic polynomials (p[i] = p[40-i] for all i)
    # With 12 nonzero terms and p[40] != 0, if palindromic then p[0] = p[40] != 0
    # So we have pairs (i, 40-i) plus possibly middle terms if 40 is even (degree 40, middle at 20)
    # 12 terms: if all come in pairs, we need 6 pairs. p[0] and p[40] are one pair.
    # Remaining 10 terms = 5 pairs from positions 1-19 (paired with 39-21)
    
    def try_palindromic():
        """Search palindromic polynomials."""
        global best_score, best_poly
        # Positions: 0 and 40 are fixed pair. Choose 5 more pairs from (1,39), (2,38), ..., (19,21)
        # That's 19 possible pairs, choose 5
        # Plus signs: each pair has 2 sign choices (both +1 or both -1, or both +2, etc.)
        # For simplicity, start with coefficients in {-1, 1}
        
        pair_positions = list(range(1, 20))  # positions 1..19, paired with 40-1=39 .. 40-19=21
        
        # Try random subsets of 5 pairs
        for trial in range(500):
            if time.time() > deadline:
                return
            chosen_pairs = random.sample(pair_positions, 5)
            # Build polynomial
            poly = [0] * 41
            # Position 0 and 40
            sign0 = random.choice([-1, 1])
            poly[0] = sign0
            poly[40] = sign0
            # Chosen pairs
            for pos in chosen_pairs:
                sign = random.choice([-1, 1])
                poly[pos] = sign
                poly[40 - pos] = sign
            
            if not is_valid(poly):
                continue
            
            score = square_nz(poly)
            if score < best_score:
                best_score = score
                best_poly = poly[:]
                # Local refinement: try flipping signs
                for flip_trial in range(100):
                    if time.time() > deadline:
                        return
                    neighbor = best_poly[:]
                    nz_idx = [i for i in range(41) if neighbor[i] != 0]
                    # Must maintain palindromic: flip both i and 40-i
                    i = random.choice(nz_idx)
                    j = 40 - i
                    if i != j:
                        neighbor[i] *= -1
                        neighbor[j] *= -1
                    else:
                        neighbor[i] *= -1
                    if is_valid(neighbor):
                        ns = square_nz(neighbor)
                        if ns < best_score:
                            best_score = ns
                            best_poly = neighbor[:]
    
    # Family B: Anti-palindromic (p[i] = -p[40-i])
    # But p[40] != 0, so p[0] = -p[40] != 0. 12 terms in pairs.
    def try_anti_palindromic():
        """Search anti-palindromic polynomials."""
        global best_score, best_poly
        pair_positions = list(range(1, 20))
        
        for trial in range(500):
            if time.time() > deadline:
                return
            chosen_pairs = random.sample(pair_positions, 5)
            poly = [0] * 41
            sign0 = random.choice([-1, 1])
            poly[0] = sign0
            poly[40] = -sign0  # anti-palindromic
            for pos in chosen_pairs:
                sign = random.choice([-1, 1])
                poly[pos] = sign
                poly[40 - pos] = -sign
            
            if not is_valid(poly):
                continue
            
            score = square_nz(poly)
            if score < best_score:
                best_score = score
                best_poly = poly[:]
                # Local refinement
                for flip_trial in range(100):
                    if time.time() > deadline:
                        return
                    neighbor = best_poly[:]
                    nz_idx = [i for i in range(41) if neighbor[i] != 0]
                    i = random.choice(nz_idx)
                    j = 40 - i
                    if i != j:
                        neighbor[i] *= -1
                        neighbor[j] *= -1
                    else:
                        neighbor[i] *= -1
                    if is_valid(neighbor):
                        ns = square_nz(neighbor)
                        if ns < best_score:
                            best_score = ns
                            best_poly = neighbor[:]
    
    # Family C: General sparse with structured positions
    # Use arithmetic progressions of positions
    def try_arithmetic_progression():
        """Try polynomials with terms at arithmetic progression positions."""
        global best_score, best_poly
        # 12 terms at positions: start, start+d, start+2d, ..., start+11d
        # Must fit in [0, 40] and include 40
        # So start + 11d = 40, start >= 0
        # d can be 1 to 3 (if d=4, 11*4=44 > 40)
        # d=1: start=29, positions 29..40
        # d=2: start=18, positions 18,20,...,40
        # d=3: start=7, positions 7,10,...,40
        
        for d in [1, 2, 3]:
            start = 40 - 11 * d
            if start < 0:
                continue
            positions = [start + i * d for i in range(12)]
            
            # Try all 2^12 sign combinations? That's 4096, feasible.
            # But let's be smarter: use random signs + local search
            for trial in range(200):
                if time.time() > deadline:
                    return
                poly = [0] * 41
                for pos in positions:
                    poly[pos] = random.choice([-1, 1])
                
                if not is_valid(poly):
                    continue
                
                score = square_nz(poly)
                if score < best_score:
                    best_score = score
                    best_poly = poly[:]
                    # Local search: flip individual signs
                    for flip in range(200):
                        if time.time() > deadline:
                            return
                        neighbor = best_poly[:]
                        idx = random.choice(positions)
                        neighbor[idx] *= -1
                        if is_valid(neighbor):
                            ns = square_nz(neighbor)
                            if ns < best_score:
                                best_score = ns
                                best_poly = neighbor[:]
    
    # Family D: Two interleaved arithmetic progressions
    def try_two_ap():
        """Try two interleaved APs of 6 terms each."""
        global best_score, best_poly
        # AP1: start1, start1+d, ..., start1+5d
        # AP2: start2, start2+d, ..., start2+5d
        # All 12 positions distinct, must include 40
        for d in [2, 3, 4, 5, 6, 7, 8]:
            for start1 in range(0, 41 - 5 * d):
                pos1 = [start1 + i * d for i in range(6)]
                if 40 in pos1:
                    # AP1 contains 40
                    for start2 in range(0, 41 - 5 * d):
                        pos2 = [start2 + i * d for i in range(6)]
                        if len(set(pos1) | set(pos2)) != 12:
                            continue
                        if 40 not in pos2:  # 40 already in pos1
                            pass
                        all_pos = sorted(set(pos1) | set(pos2))
                        if len(all_pos) != 12:
                            continue
                        if max(all_pos) != 40:
                            continue
                        # Try random signs
                        for trial in range(20):
                            if time.time() > deadline:
                                return
                            poly = [0] * 41
                            for pos in all_pos:
                                poly[pos] = random.choice([-1, 1])
                            if not is_valid(poly):
                                continue
                            score = square_nz(poly)
                            if score < best_score:
                                best_score = score
                                best_poly = poly[:]
                                for flip in range(50):
                                    if time.time() > deadline:
                                        return
                                    neighbor = best_poly[:]
                                    idx = random.choice(all_pos)
                                    neighbor[idx] *= -1
                                    if is_valid(neighbor):
                                        ns = square_nz(neighbor)
                                        if ns < best_score:
                                            best_score = ns
                                            best_poly = neighbor[:]
    
    # Run structured families
    try_palindromic()
    try_anti_palindromic()
    try_arithmetic_progression()
    
    if time.time() > deadline:
        return best_poly if best_poly else [0]*40 + [1]
    
    # Now do general local search from best found
    if best_poly is None:
        # Fallback: simple valid polynomial
        best_poly = [0] * 41
        best_poly[40] = 1
        for i in range(11):
            best_poly[i] = random.choice([-1, 1])
        best_score = square_nz(best_poly)
    
    current_poly = best_poly[:]
    current_score = best_score
    
    # Multi-start local search with diverse moves
    T = 2.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        # Generate diverse neighbors
        move_type = random.random()
        neighbor = current_poly[:]
        
        if move_type < 0.3:
            # Flip sign of one term
            nz = [i for i in range(41) if neighbor[i] != 0]
            idx = random.choice(nz)
            neighbor[idx] *= -1
        elif move_type < 0.5:
            # Move a term to a new position
            nz = [i for i in range(41) if neighbor[i] != 0]
            z = [i for i in range(41) if neighbor[i] == 0]
            if not z:
                continue
            src = random.choice(nz)
            if src == 40:
                # Can't move 40, but can swap
                dst = random.choice([i for i in nz if i != 40] or z)
                if dst == 40 or dst not in nz:
                    continue
                neighbor[src], neighbor[dst] = neighbor[dst], neighbor[src]
            else:
                dst = random.choice(z)
                neighbor[dst] = neighbor[src]
                neighbor[src] = 0
        elif move_type < 0.7:
            # Change value of a term
            nz = [i for i in range(41) if neighbor[i] != 0]
            idx = random.choice(nz)
            old = neighbor[idx]
            new = old + random.choice([-2, -1, 1, 2])
            if new == 0:
                new = old
            elif abs(new) > 64:
                new = 64 if new > 0 else -64
            neighbor[idx] = new
        elif move_type < 0.85:
            # Swap two terms' positions
            nz = [i for i in range(41) if neighbor[i] != 0]
            if len(nz) < 2:
                continue
            i, j = random.sample(nz, 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Palindromic constraint move: pick i, set p[40-i] = p[i] or -p[i]
            nz = [i for i in range(41) if neighbor[i] != 0]
            i = random.choice(nz)
            j = 40 - i
            if i != j:
                if random.random() < 0.5:
                    neighbor[j] = neighbor[i]
                else:
                    neighbor[j] = -neighbor[i]
                # Ensure 12 nonzero terms
                # This might change count, so validate later
        
        if not is_valid(neighbor):
            continue
        
        new_score = square_nz(neighbor)
        
        if new_score < current_score:
            current_poly = neighbor
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_poly = current_poly[:]
        else:
            delta = new_score - current_score
            if T > T_min:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_poly = neighbor
                    current_score = new_score
            T *= cooling
        
        # Periodic restart to escape local optima
        if random.random() < 0.001:
            # Generate a fresh structured candidate
            method = random.random()
            if method < 0.3:
                # Palindromic
                pair_pos = list(range(1, 20))
                chosen = random.sample(pair_pos, 5)
                current_poly = [0] * 41
                s0 = random.choice([-1, 1])
                current_poly[0] = s0
                current_poly[40] = s0
                for pos in chosen:
                    s = random.choice([-1, 1])
                    current_poly[pos] = s
                    current_poly[40 - pos] = s
            elif method < 0.6:
                # AP
                d = random.choice([1, 2, 3])
                start = 40 - 11 * d
                if start >= 0:
                    current_poly = [0] * 41
                    for i in range(12):
                        current_poly[start + i * d] = random.choice([-1, 1])
            else:
                # Random
                current_poly = [0] * 41
                idxs = random.sample(range(41), 12)
                if 40 not in idxs:
                    idxs[-1] = 40
                for idx in idxs:
                    current_poly[idx] = random.choice([-1, 1])
            
            if is_valid(current_poly):
                current_score = square_nz(current_poly)
                if current_score < best_score:
                    best_score = current_score
                    best_poly = current_poly[:]
            T = 2.0
    
    return best_poly
