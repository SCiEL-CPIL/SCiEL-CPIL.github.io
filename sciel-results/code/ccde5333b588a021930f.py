import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 55
    
    # 1. Generate all 4-subsets and masks
    all_quads = list(combinations(range(1, 17), 4))
    n_quads = len(all_quads)
    masks = [0] * n_quads
    for i, q in enumerate(all_quads):
        m = 0
        for x in q:
            m |= (1 << (x - 1))
        masks[i] = m
        
    # Map mask to index for quick lookup of conflicts
    mask_to_idx = {masks[i]: i for i in range(n_quads)}
    
    # Precompute popcount for masks 0..65535 for speed?
    # Or just use bin(x).count('1')
    
    def get_conflict_indices(i, j):
        """
        Returns a list of indices k such that {masks[i], masks[j], masks[k]} form a sunflower.
        """
        M_i = masks[i]
        M_j = masks[j]
        K = M_i & M_j
        U = (M_i | M_j) & ~K  # Petals
        V = (~U) & 0xFFFF     # Allowed region for k, excluding petals
        # k must contain K and be subset of V
        # Size of K is pop_k
        pop_k = bin(K).count('1')
        size_p = 4 - pop_k
        
        # If size_p < 0, no solution
        if size_p < 0:
            return []
            
        # Iterate subsets of V of size size_p
        # V has bits where k can have elements.
        # We need to generate all subsets of V of size size_p.
        # Since V is 16-bit, and size_p is small (0 to 4), this is fast.
        
        confs = []
        # Get list of bits in V
        bits_in_v = []
        v = V
        while v:
            lsb = v & -v
            bits_in_v.append(lsb)
            v -= lsb
            
        # If len(bits_in_v) < size_p, no solution
        if len(bits_in_v) < size_p:
            return []
            
        # Generate combinations of bits_in_v of size size_p
        # size_p is at most 4.
        for combo in combinations(bits_in_v, size_p):
            sub = 0
            for b in combo:
                sub |= b
            M_k = K | sub
            if M_k in mask_to_idx:
                k_idx = mask_to_idx[M_k]
                if k_idx != i and k_idx != j:
                    confs.append(k_idx)
                    
        return confs

    best_family = []
    best_size = 0
    
    # Multiple restarts
    # Estimate cost: ~1.4M ops per restart. 55s / 0.05s per restart = 1100 restarts.
    num_restarts = 2000 
    
    for trial in range(num_restarts):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        F = []
        in_F = [False] * n_quads
        Forbidden = set()
        
        # Candidate pool
        candidates = list(range(n_quads))
        random.shuffle(candidates)
        
        for idx in candidates:
            if time.time() > deadline:
                break
                
            if in_F[idx] or idx in Forbidden:
                continue
                
            # Add idx to family
            F.append(idx)
            in_F[idx] = True
            
            # Update Forbidden with conflicts of idx with all existing in F
            # Note: We only need to check against F before adding idx?
            # No, conflicts are symmetric.
            # If we add idx, it conflicts with pairs (idx, j) for j in F.
            # The third element k is forbidden.
            
            M_idx = masks[idx]
            for j in F:
                if j == idx:
                    continue
                # Get conflicts for pair (idx, j)
                confs = get_conflict_indices(idx, j)
                for k in confs:
                    if not in_F[k]:
                        Forbidden.add(k)
            
            if len(F) > best_size:
                best_size = len(F)
                best_family = F[:]
                
    result = [sorted(all_quads[i]) for i in best_family]
    return result
