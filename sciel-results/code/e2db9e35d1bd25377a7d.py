import sys
import time
import random

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    
    # Use Beam Search CSP as per directive
    L_target = 1000
    primes_asc = sorted(primes)
    
    beam_width = 50
    max_depth = len(primes_asc)
    
    # State: (index, residues_list, cov_array, score)
    beam = [(0, [], [0] * (L_target + 1), 0)]
    
    for i in range(max_depth):
        p = primes_asc[i]
        new_beam = []
        rem = max_depth - i - 1
        
        for state in beam:
            idx, res, cov, _ = state
            
            candidates = []
            for r in range(p):
                gain_1 = 0
                gain_0 = 0
                start = r if r > 0 else p
                for x in range(start, L_target + 1, p):
                    if cov[x] == 1:
                        gain_1 += 1
                    elif cov[x] == 0:
                        gain_0 += 1
                score = gain_1 * 10 - gain_0
                candidates.append((score, r))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            valid_candidates = []
            for score, r in candidates:
                is_valid = True
                if rem <= 2:
                    for x in range(1, L_target + 1):
                        c = cov[x]
                        if c + rem < 2:
                            if x % p != r:
                                is_valid = False
                                break
                if is_valid:
                    valid_candidates.append((score, r))
            
            if not valid_candidates:
                continue
                
            for score, r in valid_candidates:
                new_cov = cov[:]
                start = r if r > 0 else p
                for x in range(start, L_target + 1, p):
                    new_cov[x] += 1
                new_res = res + [r]
                new_beam.append((i+1, new_res, new_cov, score))
        
        if not new_beam:
            # Fallback to greedy if CSP fails
            return _greedy_fallback(primes)
        
        new_beam.sort(key=lambda x: x[3], reverse=True)
        beam = new_beam[:beam_width]
        
    best_state = beam[0]
    res = best_state[1]
    
    res_map = {}
    for p, r in zip(primes_asc, res):
        res_map[p] = r
        
    final_residues = []
    for p in sorted(res_map.keys()):
        final_residues.append(res_map[p])
        
    return final_residues

def _greedy_fallback(primes):
    L_max = 1000
    primes_desc = sorted(primes, reverse=True)
    cov = [0] * (L_max + 1)
    residues = [0] * len(primes_desc)
    
    for i, p in enumerate(primes_desc):
        best_r = 0
        best_gain = -1
        for r in range(p):
            gain = 0
            start = r if r > 0 else p
            for x in range(start, L_max + 1, p):
                if cov[x] < 2:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[i] = best_r
        start = best_r if best_r > 0 else p
        for x in range(start, L_max + 1, p):
            cov[x] += 1
            
    res_map = {}
    for p, r in zip(primes_desc, residues):
        res_map[p] = r
        
    final_residues = []
    for p in sorted(res_map.keys()):
        final_residues.append(res_map[p])
    return final_residues
