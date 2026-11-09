"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for 150s limit
    
    # Helper: Generate Paley tournament for prime p
    def get_paley(p):
        if p < 3 or p % 4 != 3:
            return None
        squares = {x * x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return {'n': p, 'orientation': orientation}

    # Helper: Fast validity check using bitmasks
    def is_valid_tournament(n, orientation):
        out_mask = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    out_mask[i] |= (1 << j)
                else:
                    out_mask[j] |= (1 << i)
        
        # Check all 4-sets
        for a in range(n - 3):
            ma = out_mask[a]
            for b in range(a + 1, n - 2):
                mb = out_mask[b]
                a_beats_b = (ma >> b) & 1
                b_beats_a = (mb >> a) & 1
                for c in range(b + 1, n - 1):
                    mc = out_mask[c]
                    a_beats_c = (ma >> c) & 1
                    b_beats_c = (mb >> c) & 1
                    for d in range(c + 1, n):
                        md = out_mask[d]
                        
                        # a dominates
                        if a_beats_b and a_beats_c and ((ma >> d) & 1):
                            continue
                        # b dominates
                        if b_beats_a and b_beats_c and ((mb >> d) & 1):
                            continue
                        # c dominates
                        if ((mc >> a) & 1) and ((mc >> b) & 1) and ((mc >> d) & 1):
                            continue
                        # d dominates
                        if ((md >> a) & 1) and ((md >> b) & 1) and ((md >> c) & 1):
                            continue
                        
                        return False
        return True

    # 1. Check smaller Paley tournaments
    # Primes p ≡ 3 (mod 4) less than 67
    paley_primes = [7, 11, 19, 23, 31, 43, 47, 59]
    for p in paley_primes:
        res = get_paley(p)
        if res and is_valid_tournament(p, res['orientation']):
            return res

    # 2. Fallback: Paley 67
    best_result = get_paley(67)
    
    # 3. Try local search for n=66, 65 with improved heuristic
    def local_search(n, max_steps=20000):
        # Initialize with Paley 67 truncated/modified if possible, else random
        # For now, use random initialization but with a better flip strategy
        
        num_edges = n * (n - 1) // 2
        orientation = [random.randint(0, 1) for _ in range(num_edges)]
        
        # Precompute edge indices for faster access
        # idx[i][j] for i < j
        idx_table = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_table[i][j] = idx
                idx_table[j][i] = idx
                idx += 1
        
        steps = 0
        while steps < max_steps:
            if time.time() > deadline:
                return None
            steps += 1
            
            # Build out_mask
            out_mask = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    bit = orientation[idx]
                    idx += 1
                    if bit == 0:
                        out_mask[i] |= (1 << j)
                    else:
                        out_mask[j] |= (1 << i)
                        
            # Find violation
            violation = None
            for a in range(n - 3):
                ma = out_mask[a]
                for b in range(a + 1, n - 2):
                    mb = out_mask[b]
                    a_beats_b = (ma >> b) & 1
                    b_beats_a = (mb >> a) & 1
                    for c in range(b + 1, n - 1):
                        mc = out_mask[c]
                        a_beats_c = (ma >> c) & 1
                        b_beats_c = (mb >> c) & 1
                        for d in range(c + 1, n):
                            md = out_mask[d]
                            if not (
                                (a_beats_b and a_beats_c and ((ma >> d) & 1)) or
                                (b_beats_a and b_beats_c and ((mb >> d) & 1)) or
                                (((mc >> a) & 1) and ((mc >> b) & 1) and ((mc >> d) & 1)) or
                                (((md >> a) & 1) and ((md >> b) & 1) and ((md >> c) & 1))
                            ):
                                violation = (a, b, c, d)
                                break
                        if violation: break
                    if violation: break
                if violation: break
            
            if not violation:
                return {'n': n, 'orientation': orientation}
                
            # Flip an edge in the violating 4-set
            # Use a smarter flip strategy: flip the edge that is most likely to fix the violation
            a, b, c, d = violation
            pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            
            # Count how many 4-sets each edge is involved in violations
            # For simplicity, just pick the edge that currently points "wrong"
            # For a violation, one of the vertices should dominate all others
            # Try flipping each edge and see if it fixes the violation (approximate)
            
            # Simple heuristic: flip the edge with the highest "conflict" score
            # For now, just pick randomly from the 6 edges
            u, v = random.choice(pairs)
            idx = idx_table[u][v]
            orientation[idx] ^= 1

        return None

    # Try local search for n=66
    if time.time() < deadline - 20.0:
        res = local_search(66, max_steps=5000)
        if res:
            return res
            
    # Try local search for n=65
    if time.time() < deadline - 10.0:
        res = local_search(65, max_steps=5000)
        if res:
            return res
            
    # Fallback
    return best_result

# EVOLVE-BLOCK-END
