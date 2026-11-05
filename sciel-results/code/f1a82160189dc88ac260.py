import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We aim for n=34 first as it's the known valid size.
    # If we find valid for 34, we can try to extend to 35, 36, etc.
    # But extending is hard. Let's try to find valid for 34, then 35, 36...
    # Actually, the task says "Maximize n".
    # The seed returns n=9.
    # The published construction has n=34.
    # If we return n=34, score is 34/9 = 3.77.
    # If we return n=35, score is 35/9 = 3.88.
    
    # Let's try to find a valid coloring for n=34.
    # Then try n=35, 36, etc.
    
    best_result = {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
    
    # We will try n from 34 down to 10? Or up?
    # The search is easier for smaller n.
    # But score is higher for larger n.
    # Let's try 34, 35, 36.
    # If we find valid for 36, great. If not, 35. If not, 34.
    
    # Actually, let's try to find valid for 34 first.
    # If we have time, try 35.
    
    target_ns = [34, 35, 36]
    
    # Precompute GF(32) for initialization
    # Primitive poly: x^5 + x^2 + 1 (0b10101 = 21)
    poly32 = 21
    def gf_mul32(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x20:
                a ^= poly32
        return p
        
    def gf_pow32(a, n):
        result = 1
        while n:
            if n & 1:
                result = gf_mul32(result, a)
            a = gf_mul32(a, a)
            n >>= 1
        return result
        
    # Squares in GF(32)
    squares32 = set()
    for i in range(1, 32):
        squares32.add(gf_mul32(i, i))
        
    # Non-zero elements
    nz_elements = list(range(1, 32))
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # Map vertices 0..n-1 to GF(32) elements
        # If n > 31, we can't map to distinct non-zero elements of GF(32).
        # For n=34, we have 31 non-zero elements. We must repeat some or use 0?
        # 0 is not in multiplicative group.
        # Let's use 0..n-1 mapped to a permutation of GF(32) elements?
        # If n > 31, we have duplicates.
        # For n=34, we have 3 duplicates.
        # This might break symmetry.
        # Alternative: Use GF(256) for larger n?
        # Let's stick to GF(32) for n<=31. For n=32..36, use GF(256) or random.
        
        if n <= 31:
            vertices = nz_elements[:n]
        else:
            # Use GF(256) for larger n
            poly256 = 0x11D
            def gf_mul256(a, b):
                p = 0
                while b:
                    if b & 1:
                        p ^= a
                    b >>= 1
                    a <<= 1
                    if a & 0x100:
                        a ^= poly256
                return p
                
            squares256 = set()
            for i in range(1, 256):
                squares256.add(gf_mul256(i, i))
                
            # Pick n distinct non-zero elements from GF(256)
            # Just use 1..n
            vertices = list(range(1, n + 1))
            
        def get_init_color(a, b, c, n):
            if n <= 31:
                s = a ^ b ^ c
                if s == 0:
                    return 1
                return 0 if s in squares32 else 1
            else:
                s = a ^ b ^ c
                if s == 0:
                    return 1
                return 0 if s in squares256 else 1

        # Generate triples
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    triples.append((i, j, k))
                    
        num_triples = len(triples)
        
        # Map (a,b,c) to index
        idx_map = {}
        for idx, (i, j, k) in enumerate(triples):
            idx_map[(i, j, k)] = idx
            
        # Initial colors
        colors = [get_init_color(vertices[i], vertices[j], vertices[k], n) for i, j, k in triples]
        
        # Precompute for incremental update
        # For each triple t=(a,b,c), we need to know which 4-sets and 5-sets it belongs to.
        # But for block flips, we need to evaluate the change in penalty for a set of flips.
        # Penalty P = #RedK4 + #BlueK5.
        
        # To make this efficient, we can use a "conflict" graph or similar.
        # But let's try a simpler approach:
        # 1. Calculate initial penalty.
        # 2. Local search: pick a random set of k triples (k=1..5) and flip them.
        # 3. Calculate delta penalty.
        #    This is expensive if done from scratch.
        #    We need incremental update.
        
        # Let's maintain:
        # red_k4_count
        # blue_k5_count
        # And for each 4-set and 5-set, the count of Red/Blue triples?
        # No, just boolean: is it mono?
        
        # Let's precompute lists of 4-sets and 5-sets.
        # C(34,4) = 46376
        # C(34,5) = 278256
        # This is manageable.
        
        k4_sets = list(combinations(range(n), 4))
        k5_sets = list(combinations(range(n), 5))
        
        # For each set, store the list of triple indices.
        k4_triples = []
        for q in k4_sets:
            t1 = idx_map[(q[0], q[1], q[2])]
            t2 = idx_map[(q[0], q[1], q[3])]
            t3 = idx_map[(q[0], q[2], q[3])]
            t4 = idx_map[(q[1], q[2], q[3])]
            k4_triples.append([t1, t2, t3, t4])
            
        k5_triples = []
        for q in k5_sets:
            ts = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        ts.append(idx_map[(q[i], q[j], q[k])])
            k5_triples.append(ts)
            
        # Initial counts
        red_k4_count = 0
        blue_k5_count = 0
        
        for ts in k4_triples:
            c0 = colors[ts[0]]
            c1 = colors[ts[1]]
            c2 = colors[ts[2]]
            c3 = colors[ts[3]]
            if c0 == c1 == c2 == c3 == 0:
                red_k4_count += 1
                
        for ts in k5_triples:
            all_blue = True
            for t in ts:
                if colors[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                blue_k5_count += 1
                
        current_penalty = red_k4_count + blue_k5_count
        best_penalty = current_penalty
        best_colors = colors[:]
        
        # Local Search
        # We will use a simple simulated annealing with block flips.
        temp = 5.0
        cooling = 0.999
        steps = 0
        
        while time.time() < deadline - 5:
            # Pick block size
            # Mostly 1, sometimes 2 or 3
            r = random.random()
            if r < 0.6:
                block_size = 1
            elif r < 0.9:
                block_size = 2
            else:
                block_size = 3
                
            # Pick random triples
            flip_indices = set(random.sample(range(num_triples), block_size))
            
            # Calculate delta
            # This is the hard part.
            # We need to check all K4s and K5s that contain ANY of the flipped triples.
            
            # Precompute: for each triple index, which K4s and K5s does it belong to?
            # triple_to_k4s = [[] for _ in range(num_triples)]
            # triple_to_k5s = [[] for _ in range(num_triples)]
            # This memory might be high but C(34,3)=5984.
            # Each K4 has 4 triples. Total links 4*46376 = 185504.
            # Each K5 has 10 triples. Total links 10*278256 = 2782560.
            # This is ~3M integers. ~24MB. Acceptable.
            
            # Let's build these maps once per n.
            triple_to_k4s = [[] for _ in range(num_triples)]
            for i, ts in enumerate(k4_triples):
                for t in ts:
                    triple_to_k4s[t].append(i)
                    
            triple_to_k5s = [[] for _ in range(num_triples)]
            for i, ts in enumerate(k5_triples):
                for t in ts:
                    triple_to_k5s[t].append(i)
                    
            # Now, for a given set of flips, find all affected K4s and K5s.
            affected_k4s = set()
            affected_k5s = set()
            
            for t in flip_indices:
                for k4_idx in triple_to_k4s[t]:
                    affected_k4s.add(k4_idx)
                for k5_idx in triple_to_k5s[t]:
                    affected_k5s.add(k5_idx)
                    
            # Calculate current penalty for these sets
            old_penalty_part = 0
            for k4_idx in affected_k4s:
                ts = k4_triples[k4_idx]
                if all(colors[t] == 0 for t in ts):
                    old_penalty_part += 1
                    
            for k5_idx in affected_k5s:
                ts = k5_triples[k5_idx]
                if all(colors[t] == 1 for t in ts):
                    old_penalty_part += 1
                    
            # Apply flips
            for t in flip_indices:
                colors[t] = 1 - colors[t]
                
            # Calculate new penalty for these sets
            new_penalty_part = 0
            for k4_idx in affected_k4s:
                ts = k4_triples[k4_idx]
                if all(colors[t] == 0 for t in ts):
                    new_penalty_part += 1
                    
            for k5_idx in affected_k5s:
                ts = k5_triples[k5_idx]
                if all(colors[t] == 1 for t in ts):
                    new_penalty_part += 1
                    
            delta = new_penalty_part - old_penalty_part
            new_penalty = current_penalty + delta
            
            if new_penalty < best_penalty:
                best_penalty = new_penalty
                best_colors = colors[:]
                
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_penalty = new_penalty
            else:
                # Revert
                for t in flip_indices:
                    colors[t] = 1 - colors[t]
                    
            temp *= cooling
            steps += 1
            
            if best_penalty == 0:
                # Found valid coloring!
                return {'n': n, 'colors': best_colors}
                
        # If we didn't find valid for this n, keep best_result
        # But we should check if best_penalty is 0?
        # The loop returns immediately if best_penalty == 0.
        # So if we exit, best_penalty > 0.
        
    return best_result

# To speed up, we should precompute triple_to_k4s and triple_to_k5s outside the main loop if possible?
# No, they depend on n.
# The current code builds them inside the `for n in target_ns` loop.
# This is correct.

# One optimization: The construction of triple_to_k4s and triple_to_k5s is O(num_triples * avg_degree).
# For n=34, this is fast.

# Let's refine the code to ensure it's robust.
