<<<<<<< SEARCH
import time
import random
from collections import defaultdict

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline:
            # Fallback
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        
        # 1. Try Optimal (Odd-Pairing)
        optimal_paths = _try_optimal_decomposition(edges, deadline)
        
        if optimal_paths is not None:
            results.append(optimal_paths)
        else:
            # 2. Fallback to Greedy
            greedy_paths = _greedy_decompose(edges, deadline)
            results.append(greedy_paths)
            
    return results
=======
import time
import random
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    deadline = time.time() + 140
    results = []
    
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > deadline - 20:
            # Fallback: just use individual edges as paths
            edges = _instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = _instance(*spec)
        
        # Try ILP-based decomposition
        ilp_paths = _ilp_decompose(edges, deadline)
        
        if ilp_paths is not None and len(ilp_paths) > 0:
            results.append(ilp_paths)
        else:
            # Fallback to greedy
            greedy_paths = _greedy_decompose(edges, deadline)
            results.append(greedy_paths)
            
    return results
>>>>>>> REPLACE

