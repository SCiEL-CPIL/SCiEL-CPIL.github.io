import time
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    """
    Returns an array of signs (-1 or +1) for each edge in the graph.
    Strategy:
    1. Exhaustive search for small m (m <= 6, i.e., edges <= 18) to guarantee global optimum.
    2. Simulated Annealing for larger m (m=7, 9) to escape local minima.
    3. Efficient connectivity check using the coboundary property:
       The 2-lift is connected iff the signing is NOT a coboundary.
       A signing is a coboundary if there exists f: V -> {0,1} such that s_e = (-1)^{f(u)+f(v)} for all edges e.
       This is equivalent to checking if the product of signs around every cycle is +1.
       For a graph with cyclomatic number k, we can check k independent cycles.
       Alternatively, we can perform a BFS on the "signed" graph to see if we can assign consistent labels.
       Actually, the lift is connected iff the signed graph is not "balanced" in the sense of having a vertex labeling f such that s_uv = (-1)^(f(u)+f(v)).
       If such f exists, the lift splits into two components. If no such f exists, the lift is connected.
       So, is_lift_connected(signs) is True iff NO such f exists.
       
       To check if a coboundary exists:
       Pick a root vertex 0. Set f(0)=0.
       BFS/DFS. For each edge (u,v) with sign s:
       f(v) must be f(u) XOR (1 if s==-1 else 0).
       If we reach a contradiction, no coboundary exists -> Lift is Connected.
       If we complete BFS without contradiction, a coboundary exists -> Lift is Disconnected.
       
       This check is O(N+E), which is very fast.
    """
    
    deadline = time.time() + 85.0  # Safety margin
    
    n = graph['n']
    edges = graph['edges']
    num_edges = len(edges)
    m = n // 2
    
    # Precompute edge indices for connectivity check
    # We need to check if the signing is a coboundary.
    # If it IS a coboundary, the lift is disconnected.
    # We want the lift to be CONNECTED, so we want the signing to be NOT a coboundary.
    
    def is_lift_connected(signs):
        """
        Returns True if the 2-lift is connected.
        This is equivalent to the signing NOT being a coboundary.
        """
        # f[v] is the label 0 or 1.
        # Constraint: s(u,v) = (-1)^(f(u) + f(v))
        # => f(v) = f(u) XOR (1 if s(u,v) == -1 else 0)
        
        f = [-1] * n
        f[0] = 0
        
        # Build adjacency list with edge indices for fast lookup? 
        # Or just iterate edges? 
        # BFS is better.
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        queue = deque([0])
        
        while queue:
            u = queue.popleft()
            for v, idx in adj[u]:
                s = signs[idx]
                # Required flip
                flip = 1 if s == -1 else 0
                required_f_v = f[u] ^ flip
                
                if f[v] == -1:
                    f[v] = required_f_v
                    queue.append(v)
                else:
                    if f[v] != required_f_v:
                        # Contradiction found: No coboundary exists.
                        # Therefore, the lift IS connected.
                        return True
        
        # No contradiction found: A coboundary exists.
        # Therefore, the lift is DISCONNECTED.
        return False

    def get_radius(signs):
        S = np.zeros((n, n), dtype=np.float64)
        for i, (u, v) in enumerate(edges):
            val = float(signs[i])
            S[u, v] = val
            S[v, u] = val
        eigvals = np.linalg.eigvalsh(S)
        return np.max(np.abs(eigvals))

    best_signs = None
    best_radius = np.inf
    
    rng = np.random.default_rng(3)
    
    # Strategy 1: Exhaustive Search for small m
    # m=4: 12 edges, 2^12 = 4096
    # m=5: 15 edges, 2^15 = 32768
    # m=6: 18 edges, 2^18 = 262144 (Doable in < 1s with vectorization? Maybe not fully vectorized eig, but let's try)
    # m=7: 21 edges, 2^21 = 2M (Too slow for full exhaustive with eigvalsh)
    
    if num_edges <= 18:
        # Exhaustive search
        # We can vectorize the radius calculation? 
        # No, eigvalsh is not easily vectorized for different matrices in a simple loop without overhead.
        # But 262k iterations * ~50us per eigvalsh = ~13s. Might be tight for m=6.
        # Let's optimize the loop.
        
        # Precompute edge endpoints
        u_arr = np.array([e[0] for e in edges])
        v_arr = np.array([e[1] for e in edges])
        
        # We will iterate through all 2^num_edges combinations.
        # To speed up, we can use a Gray code to update the matrix incrementally?
        # Or just brute force if m is small enough.
        
        # For m=4, 5: Brute force is very fast.
        # For m=6: 262k iterations. Let's see if it fits.
        
        if time.time() > deadline - 5.0:
             # Not enough time for exhaustive, fall back to SA
             pass
        else:
            # Generate all combinations
            # Using np.indices or itertools.product
            # For 18 bits, we can use integers 0 to 2^18-1 and extract bits.
            
            max_combo = 1 << num_edges
            
            # Optimization: Check connectivity first?
            # Connectivity check is O(N+E). Radius is O(N^3).
            # Most signings are connected.
            
            # We'll do a simple loop.
            for k in range(max_combo):
                if time.time() > deadline:
                    break
                
                # Extract signs from k
                # signs[i] = -1 if bit i is 0, +1 if bit i is 1?
                # Let's say bit 1 -> +1, bit 0 -> -1.
                signs = np.where((k >> np.arange(num_edges)) & 1, 1, -1).astype(np.int8)
                
                if not is_lift_connected(signs):
                    continue
                
                # Compute radius
                S = np.zeros((n, n), dtype=np.float64)
                S[u_arr, v_arr] = signs.astype(float)
                S[v_arr, u_arr] = signs.astype(float)
                
                # Symmetric matrix
                eigvals = np.linalg.eigvalsh(S)
                radius = np.max(np.abs(eigvals))
                
                if radius < best_radius:
                    best_radius = radius
                    best_signs = signs.copy()
                    
            if best_signs is not None:
                return best_signs

    # Strategy 2: Simulated Annealing for larger m or if exhaustive failed/was skipped
    
    # Initial state: Random
    signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
    
    # Ensure connectivity for start
    while not is_lift_connected(signs):
        # Flip random edge
        idx = rng.integers(0, num_edges)
        signs[idx] *= -1
        # Safety break
        if rng.integers(0, 100) == 0: 
            signs = rng.choice([-1, 1], size=num_edges).astype(np.int8)
            break

    current_radius = get_radius(signs)
    best_signs = signs.copy()
    best_radius = current_radius
    
    # SA Parameters
    T0 = 1.0
    T_end = 0.01
    cooling_rate = 0.999
    steps_per_temp = 50
    
    u_arr = np.array([e[0] for e in edges])
    v_arr = np.array([e[1] for e in edges])
    
    T = T0
    while T > T_end and time.time() < deadline:
        for _ in range(steps_per_temp):
            if time.time() > deadline:
                break
                
            # Pick random edge to flip
            idx = rng.integers(0, num_edges)
            old_sign = signs[idx]
            new_sign = -old_sign
            
            # Update signs
            signs[idx] = new_sign
            
            # Check connectivity
            if not is_lift_connected(signs):
                signs[idx] = old_sign
                continue
            
            # Compute new radius
            S = np.zeros((n, n), dtype=np.float64)
            S[u_arr, v_arr] = signs.astype(float)
            S[v_arr, u_arr] = signs.astype(float)
            new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
            
            delta = new_radius - current_radius
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                current_radius = new_radius
                signs[idx] = new_sign
                if new_radius < best_radius:
                    best_radius = new_radius
                    best_signs = signs.copy()
            else:
                # Reject
                signs[idx] = old_sign
        
        T *= cooling_rate
        
        # Optional: Local search pass to polish
        # Do a quick hill climb to escape shallow local minima
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(num_edges):
                old_sign = signs[i]
                signs[i] = -old_sign
                if not is_lift_connected(signs):
                    signs[i] = old_sign
                    continue
                S = np.zeros((n, n), dtype=np.float64)
                S[u_arr, v_arr] = signs.astype(float)
                S[v_arr, u_arr] = signs.astype(float)
                new_radius = np.max(np.abs(np.linalg.eigvalsh(S)))
                if new_radius < current_radius:
                    current_radius = new_radius
                    improved = True
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_signs = signs.copy()
                else:
                    signs[i] = old_sign
                    
    if best_signs is None:
        # Fallback
        best_signs = np.ones(num_edges, dtype=np.int8)
        if not is_lift_connected(best_signs):
            best_signs[0] = -1
            if not is_lift_connected(best_signs):
                best_signs[1] = -1

    return best_signs
