import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Try n=22, 21, 20
    # Budgets
    budgets = {22: 45.0, 21: 40.0, 20: 20.0}
    
    for n in [22, 21, 20]:
        if time.time() > deadline:
            break
        
        # Precompute masks for sizes 6, 7, 8
        # This is done once per n
        masks_6 = []
        for c in combinations(range(n), 6):
            mask = 0
            for v in c:
                mask |= (1 << v)
            masks_6.append((mask, c))
            
        masks_7 = []
        if n >= 7:
            for c in combinations(range(n), 7):
                mask = 0
                for v in c:
                    mask |= (1 << v)
                masks_7.append((mask, c))
                
        masks_8 = []
        if n >= 8:
            for c in combinations(range(n), 8):
                mask = 0
                for v in c:
                    mask |= (1 << v)
                masks_8.append((mask, c))
        
        # Local Search
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            adj_bits = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        
            # Run search
            # We will run until time limit for this n
            n_deadline = min(deadline, start_time + budgets[n])
            
            while time.time() < n_deadline:
                # Check 6, 7, 8
                viol = find_violation_main(adj_bits, masks_6, masks_7, masks_8)
                
                if viol is None:
                    # Check 9..n
                    viol_large = find_violation_large(adj_bits, n)
                    if viol_large is None:
                        # Valid
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj_bits[i] & (1 << j):
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
                    else:
                        # Flip edge in viol_large
                        u, v = random.sample(viol_large, 2)
                        toggle_edge(adj_bits, u, v)
                else:
                    # Flip edge in viol
                    u, v = random.sample(viol, 2)
                    toggle_edge(adj_bits, u, v)
                    
    # Fallback
    return {'n': 5, 'edges': []}

def toggle_edge(adj_bits, u, v):
    bit_u = 1 << u
    bit_v = 1 << v
    if adj_bits[u] & bit_v:
        adj_bits[u] ^= bit_v
        adj_bits[v] ^= bit_u
    else:
        adj_bits[u] |= bit_v
        adj_bits[v] |= bit_u

def find_violation_main(adj_bits, masks_6, masks_7, masks_8):
    # Check 6
    for mask, subset in masks_6:
        d0 = (adj_bits[subset[0]] & mask).bit_count()
        if d0 != (adj_bits[subset[1]] & mask).bit_count(): continue
        if d0 != (adj_bits[subset[2]] & mask).bit_count(): continue
        if d0 != (adj_bits[subset[3]] & mask).bit_count(): continue
        if d0 != (adj_bits[subset[4]] & mask).bit_count(): continue
        if d0 != (adj_bits[subset[5]] & mask).bit_count(): continue
        return subset
        
    # Check 7
    if masks_7:
        for mask, subset in masks_7:
            d0 = (adj_bits[subset[0]] & mask).bit_count()
            if d0 != (adj_bits[subset[1]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[2]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[3]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[4]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[5]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[6]] & mask).bit_count(): continue
            return subset
            
    # Check 8
    if masks_8:
        for mask, subset in masks_8:
            d0 = (adj_bits[subset[0]] & mask).bit_count()
            if d0 != (adj_bits[subset[1]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[2]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[3]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[4]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[5]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[6]] & mask).bit_count(): continue
            if d0 != (adj_bits[subset[7]] & mask).bit_count(): continue
            return subset
            
    return None

def find_violation_large(adj_bits, n):
    # Check sizes 9 to n
    # This is expensive, so only called when 6-8 are clean
    for k in range(9, n + 1):
        for c in combinations(range(n), k):
            mask = 0
            for v in c:
                mask |= (1 << v)
            d0 = (adj_bits[c[0]] & mask).bit_count()
            valid = True
            for i in range(1, k):
                if (adj_bits[c[i]] & mask).bit_count() != d0:
                    valid = False
                    break
            if valid:
                return list(c)
    return None
