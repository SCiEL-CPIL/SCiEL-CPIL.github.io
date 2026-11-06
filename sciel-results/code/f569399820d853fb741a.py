"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from math import gcd
import sys

def construct():
    N = 512
    numbers = list(range(1, 512))
    
    # Precompute weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Build conflict masks
    # conflicts[a] is a bitmask where bit b is 1 if gcd(a, b) > 1
    conflicts = [0] * N
    
    for a in range(1, N):
        mask = 0
        for b in range(1, N):
            if a != b and gcd(a, b) > 1:
                mask |= (1 << b)
        conflicts[a] = mask
        
    # Mask for all valid vertices 1..511
    mask_all = ((1 << N) - 1) ^ 1
    
    # Greedy Initial Solution
    # Since higher value = higher weight, we just pick the largest available
    avail = mask_all
    current_weight = 0.0
    current_set_mask = 0
    
    while avail:
        v = avail.bit_length() - 1
        if v < 1:
            break
        # Include v
        current_weight += weights[v]
        current_set_mask |= (1 << v)
        # Remove v and conflicts
        avail &= ~(conflicts[v] | (1 << v))
        
    best_weight = current_weight
    best_set_mask = current_set_mask
    
    sys.setrecursionlimit(10000)
    
    def calc_bound(mask):
        s = 0.0
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            s += weights[v]
            m ^= lsb
        return s
        
    def search(avail, current_w, current_mask):
        nonlocal best_weight, best_set_mask
        
        # Prune
        if current_w + calc_bound(avail) <= best_weight:
            return
            
        if avail == 0:
            if current_w > best_weight:
                best_weight = current_w
                best_set_mask = current_mask
            return
            
        # Pick highest weight vertex (highest value)
        v = avail.bit_length() - 1
        
        # Branch 1: Include v
        new_avail = avail & ~(conflicts[v] | (1 << v))
        search(new_avail, current_w + weights[v], current_mask | (1 << v))
        
        # Branch 2: Exclude v
        new_avail = avail & ~(1 << v)
        search(new_avail, current_w, current_mask)
        
    search(mask_all, 0.0, 0)
    
    # Convert best_set_mask to list
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m ^= lsb
        
    return result
# EVOLVE-BLOCK-END
