import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Tries direct construction for small n first, then falls back to Petersen lift.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3:
        # Try direct construction for n from 512 to 529
        # If found, it will be better than the Petersen lift (n=530)
        for n_target in range(512, 531, 2): # 3-regular graphs must have even n
            if time.time() > deadline - 10:
                break
            result = _construct_direct_3reg(g, n_target, deadline)
            if result is not None:
                return result
        
        # Fallback to Petersen lift
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_direct_3reg(g, n, deadline):
    """
    Tries to construct a 3-regular graph on n vertices with girth >= g.
    """
    # Try a few random instances
    max_attempts = 50
    
    for _ in range(max_attempts):
        if time.time() > deadline:
            return None
            
        # Generate random 3-regular graph
        edges = _random_3reg_graph(n)
        if not edges:
            continue
            
        # Check girth
        if _has_girth_at_least(n, edges, g):
            return n, edges
            
        # Try to repair
        # (Simple repair: just retry random generation for now, 
        # as implementing a robust local repair for girth is complex)
        # But we can try a few local moves
        # For now, just rely on random generation. 
        # With n=512, the probability of a short cycle is low enough 
        # that random search might hit a valid one.
        
    return None

def _random_3reg_graph(n):
    """
    Generates a random simple 3-regular graph on n vertices using configuration model.
    Returns list of edges or None if failed to make simple.
    """
    if n % 2 != 0 or n < 4:
        return None
        
    # Stubs
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
        
    random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    for i in range(0, n * 3, 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None # Loop
        if (u, v) in edges or (v, u) in edges:
            return None # Multi-edge
        edges.add((u, v))
        
    # Check degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 3 for d in deg):
        return None
        
    return list(edges)

def _construct_petersen_lift(g, deadline, start_time):
    # ... (keep existing implementation) ...
    pass

# ... (keep other existing functions) ...
