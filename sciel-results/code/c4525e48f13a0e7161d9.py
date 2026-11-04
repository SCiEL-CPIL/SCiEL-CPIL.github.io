"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    """
    Solve the QAP instance wil100 using Simulated Annealing with O(n) delta evaluation.
    
    Parameters:
    a: (n, n) int64 numpy array - flow matrix
    b: (n, n) int64 numpy array - distance matrix
    
    Returns:
    p: list of n integers, where p[i] is the location assigned to facility i
    """
    n = len(a)
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Convert to int64 for exact arithmetic
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Precompute row sums of a for delta computation
    # a_row_sums[i] = sum_j a[i,j]
    a_row_sums = a.sum(axis=1)  # shape (n,)
    
    # Initial solution: rank-matching heuristic
    # Heavy flow facilities go to central (low-distance) locations
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def calc_cost(p):
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))
    
    def delta_cost(p, i, j):
        """
        Compute the change in cost if we swap facilities i and j in permutation p.
        cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        When we swap p[i] and p[j], the terms involving i or j change.
        Let's derive:
        Original contribution of i and j to cost:
          terms where first index is i: sum_k a[i,k] * b[p[i], p[k]]
          terms where first index is j: sum_k a[j,k] * b[p[j], p[k]]
          terms where second index is i: sum_k a[k,i] * b[p[k], p[i]]
          terms where second index is j: sum_k a[k,j] * b[p[k], p[j]]
        
        After swapping p[i] <-> p[j]:
          new p[i] = old p[j], new p[j] = old p[i]
          
        New contribution:
          terms where first index is i: sum_k a[i,k] * b[p'[i], p'[k]]
          terms where first index is j: sum_k a[j,k] * b[p'[j], p'[k]]
          terms where second index is i: sum_k a[k,i] * b[p'[k], p'[i]]
          terms where second index is j: sum_k a[k,j] * b[p'[k], p'[j]]
        
        The delta is complex. Let me use a different approach.
        
        Actually, let's define:
        C(p) = sum_{i,j} a[i,j] * b[p[i], p[j]]
        
        When swapping positions i and j (i.e., swapping the values p[i] and p[j]):
        Let u = p[i], v = p[j] (the locations currently at positions i and j)
        After swap: p[i] = v, p[j] = u
        
        The change in cost comes from all terms where at least one of the indices is i or j.
        
        Terms affected:
        1. a[i,i]*b[u,u] -> a[i,i]*b[v,v]
        2. a[i,j]*b[u,v] -> a[i,j]*b[v,u]
        3. a[j,i]*b[v,u] -> a[j,i]*b[u,v]
        4. a[j,j]*b[v,v] -> a[j,j]*b[u,u]
        5. For k not in {i,j}:
           a[i,k]*b[u,p[k]] -> a[i,k]*b[v,p[k]]
           a[j,k]*b[v,p[k]] -> a[j,k]*b[u,p[k]]
           a[k,i]*b[p[k],u] -> a[k,i]*b[p[k],v]
           a[k,j]*b[p[k],v] -> a[k,j]*b[p[k],u]
        
        So delta = 
          a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
          + sum_{k!=i,j} [a[i,k]*(b[v,p[k]]-b[u,p[k]]) + a[j,k]*(b[u,p[k]]-b[v,p[k]]) 
                         + a[k,i]*(b[p[k],v]-b[p[k],u]) + a[k,j]*(b[p[k],u]-b[p[k],v])]
        
        Let me simplify. Note that b[v,u]-b[u,v] appears with a[i,j] and b[u,v]-b[v,u] appears with a[j,i].
        So those two combine to (a[i,j]-a[j,i])*(b[v,u]-b[u,v])
        
        Similarly a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v]) = (a[i,i]-a[j,j])*(b[v,v]-b[u,u])
        
        For the sum over k:
        = sum_{k!=i,j} [ (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]]) + (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u]) ]
        
        This is O(n) which is what we want.
        """
        u = p[i]
        v = p[j]
        
        # Diagonal terms
        delta = (a[i,i] - a[j,j]) * (b[v,v] - b[u,u])
        # Cross terms between i and j
        delta += (a[i,j] - a[j,i]) * (b[v,u] - b[u,v])
        
        # Terms with other facilities
        # We need to sum over k != i, j
        # (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        
        # Vectorized computation
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        k_indices = np.where(mask)[0]
        
        pk = p[k_indices]  # locations of other facilities
        
        # First part: (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        diff_a = a[i, k_indices] - a[j, k_indices]
        diff_b1 = b[v, pk] - b[u, pk]
        delta += np.sum(diff_a * diff_b1)
        
        # Second part: (a[k,i] - a[k,j]) * (b[p[k], v] - b[p[k], u])
        diff_a2 = a[k_indices, i] - a[k_indices, j]
        diff_b2 = b[pk, v] - b[pk, u]
        delta += np.sum(diff_a2 * diff_b2)
        
        return int(delta)
    
    # Simulated Annealing
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    T = 1000.0  # initial temperature
    cooling = 0.9995  # cooling rate
    min_T = 1.0
    
    np.random.seed(42)
    
    while time.time() < deadline and T > min_T:
        # Pick random pair to swap
        i, j = np.random.randint(0, n, size=2)
        if i == j:
            continue
        
        delta = delta_cost(p, i, j)
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            # Accept the move
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
    
    # Return the best solution found
    return best_p.tolist()
# EVOLVE-BLOCK-END
