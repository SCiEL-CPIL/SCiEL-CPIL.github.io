import time
import random
import itertools
import numpy as np

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = N * N

# Precomputation
# iso_q[p][v] is a list of q such that (q, p, v) is isosceles
# Note: (q, p, v) is isosceles if d(q,p)==d(q,v) or d(q,p)==d(p,v) or d(q,v)==d(p,v)

def _precompute_iso():
    # d2[i][j] is squared distance between i and j
    d2 = [[0]*NUM_POINTS for _ in range(NUM_POINTS)]
    for i in range(NUM_POINTS):
        xi, yi = POINTS[i]
        for j in range(i+1, NUM_POINTS):
            xj, yj = POINTS[j]
            dist = (xi-xj)**2 + (yi-yj)**2
            d2[i][j] = dist
            d2[j][i] = dist
            
    iso_q = [[[] for _ in range(NUM_POINTS)] for __ in range(NUM_POINTS)]
    
    for p in range(NUM_POINTS):
        for v in range(NUM_POINTS):
            if p == v: continue
            dpv = d2[p][v]
            # Find q such that (q, p, v) is isosceles
            # 1. d(q,p) == d(q,v)
            # 2. d(q,p) == dpv
            # 3. d(q,v) == dpv
            
            qs = set()
            for q in range(NUM_POINTS):
                if q == p or q == v:
                    continue
                if d2[q][p] == d2[q][v] or d2[q][p] == dpv or d2[q][v] == dpv:
                    qs.add(q)
            iso_q[p][v] = list(qs)
            
    return iso_q

_iso_q = None

def _init():
    global _iso_q
    if _iso_q is None:
        _iso_q = _precompute_iso()

def construct():
    _init()
    start_time = time.time()
    deadline = start_time + 280
    
    best_set = None
    best_len = float('inf')
    
    # Run multiple restarts with different random seeds for the greedy order or tie-breaking
    for restart in range(100): # Cap restarts, rely on time check
        if time.time() > deadline:
            break
            
        S = set()
        block_count = [0] * NUM_POINTS
        
        # Greedy Phase
        # We want to add points to S to block all others.
        # Initially, all points are "unblocked" (block_count 0).
        # We need to add points to S such that eventually all q not in S have block_count > 0.
        
        # To make it random, we can shuffle the order of candidates or add random tie-breaking.
        # Here we pick the best candidate each step.
        
        while True:
            # Check if all points not in S are blocked
            all_blocked = True
            for q in range(NUM_POINTS):
                if q not in S and block_count[q] == 0:
                    all_blocked = False
                    break
            
            if all_blocked:
                break
                
            # Find best p to add
            best_p = -1
            best_gain = -1
            
            # Candidates are points not in S
            # We can iterate all q not in S
            for p in range(NUM_POINTS):
                if p in S:
                    continue
                
                # Calculate gain: how many currently unblocked points become blocked
                gain = 0
                # p itself is removed from unblocked pool, so we don't count it
                # For each v in S, check if p blocks any q via (q,p,v)
                for v in S:
                    for q in _iso_q[p][v]:
                        if q not in S and block_count[q] == 0:
                            gain += 1
                            # Note: a q might be blocked by multiple v's, but we just need count of distinct q?
                            # Actually, if block_count[q] becomes > 0, it's blocked.
                            # If it was 0, and we add 1, it becomes blocked.
                            # If it was already > 0, gain is 0 for that q.
                            # So checking block_count[q] == 0 is correct for "newly blocked".
                            
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    
            if best_p == -1:
                # Should not happen if not all_blocked
                break
                
            # Add best_p to S
            S.add(best_p)
            
            # Update block_count
            for v in S:
                if v == best_p:
                    continue
                for q in _iso_q[best_p][v]:
                    block_count[q] += 1
        
        # Now S is a maximal no-isosceles set
        current_len = len(S)
        if current_len < best_len:
            best_len = current_len
            best_set = list(S)
            
        # Local Search Phase
        # Try to remove points
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each point in S
            # We need to iterate over a copy or list because we modify S
            s_list = list(S)
            random.shuffle(s_list)
            
            for p in s_list:
                if p not in S:
                    continue
                    
                # Temporarily remove p
                S.remove(p)
                
                # Update block_count (decrement)
                for v in S:
                    for q in _iso_q[p][v]:
                        block_count[q] -= 1
                        
                # Check if still maximal
                valid = True
                for q in range(NUM_POINTS):
                    if q not in S and block_count[q] == 0:
                        valid = False
                        break
                        
                if valid:
                    # Keep p removed
                    improved = True
                    break # Restart the while loop with new S
                else:
                    # Restore p
                    S.add(p)
                    for v in S:
                        for q in _iso_q[p][v]:
                            block_count[q] += 1
                            
                    # Try to swap p with some r
                    # Find r not in S that blocks the unblocked points
                    unblocked = []
                    for q in range(NUM_POINTS):
                        if q not in S and block_count[q] == 0:
                            unblocked.append(q)
                            
                    if not unblocked:
                        continue
                        
                    best_r = -1
                    best_swap_gain = -1
                    
                    for r in range(NUM_POINTS):
                        if r in S or r == p:
                            continue
                            
                        # How many unblocked points does r block?
                        # r blocks q if exists v in S such that q in _iso_q[r][v]
                        # AND q is in unblocked
                        gain = 0
                        for v in S:
                            for q in _iso_q[r][v]:
                                if q in unblocked:
                                    gain += 1
                                    
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_r = r
                            
                    if best_r != -1 and best_swap_gain > 0:
                        # Perform swap: remove p, add r
                        # p is already removed in the "temporarily remove" logic above? 
                        # No, we restored p. So S contains p.
                        S.remove(p)
                        # Decrement block_count for p
                        for v in S:
                            for q in _iso_q[p][v]:
                                block_count[q] -= 1
                                
                        S.add(best_r)
                        # Increment block_count for best_r
                        for v in S:
                            if v == best_r: continue
                            for q in _iso_q[best_r][v]:
                                block_count[q] += 1
                                
                        # Check if valid after swap
                        # We know best_r blocks best_swap_gain points.
                        # Are all unblocked points blocked?
                        # If best_swap_gain == len(unblocked), then yes.
                        # If not, the swap might not result in a valid maximal set.
                        # We should verify.
                        valid_swap = True
                        for q in range(NUM_POINTS):
                            if q not in S and block_count[q] == 0:
                                valid_swap = False
                                break
                                
                        if valid_swap:
                            improved = True
                            break # Restart while loop
                        else:
                            # Revert swap
                            S.remove(best_r)
                            for v in S:
                                for q in _iso_q[best_r][v]:
                                    block_count[q] -= 1
                                    
                            S.add(p)
                            for v in S:
                                for q in _iso_q[p][v]:
                                    block_count[q] += 1

        if best_set is not None and len(best_set) < best_len:
             best_len = len(best_set)
             best_set = list(best_set)
             
    return best_set
