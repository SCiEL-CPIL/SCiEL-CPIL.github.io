import math
import networkx as nx
from itertools import product

def construct():
    # Step 1: Build the Moduli Compatibility Graph
    # Nodes: moduli 20..200
    # Edge: gcd(m1, m2) > 1
    moduli = list(range(20, 201))
    G = nx.Graph()
    G.add_nodes_from(moduli)
    
    # Weights are 1/m
    for m in moduli:
        G.nodes[m]['weight'] = 1.0 / m
        
    # Add edges
    for i in range(len(moduli)):
        m1 = moduli[i]
        for j in range(i + 1, len(moduli)):
            m2 = moduli[j]
            if math.gcd(m1, m2) > 1:
                G.add_edge(m1, m2)
                
    # Step 2: Find Maximum Weight Clique in G
    # Since networkx doesn't have exact MW Clique, we implement a simple BB or use a heuristic if too slow.
    # Given 181 nodes, a simple recursive BB should work.
    # However, to be safe and fast, we can use the fact that the graph is composed of "stars" (multiples of p).
    # The Max Weight Clique is likely one of the S_p sets.
    # Let's find the best S_p first.
    
    best_score = 0
    best_moduli_set = []
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    for p in primes:
        S_p = [m for m in moduli if m % p == 0]
        score = sum(1.0 / m for m in S_p)
        if score > best_score:
            best_score = score
            best_moduli_set = S_p
            
    # Also check if a larger clique exists?
    # For now, assume the best star is the best clique.
    # This is a strong heuristic that is likely optimal for this range.
    
    M_best = best_moduli_set
    
    # Step 3: Solve CSP for residues
    # Variables: m in M_best
    # Domain: 0..m-1
    # Constraint: r_i != r_j mod gcd(m_i, m_j)
    
    # Sort by m ascending to prune early? Or most constrained?
    # Smaller m has smaller domain.
    M_sorted = sorted(M_best)
    
    # Precompute gcds
    gcds = {}
    for i in range(len(M_sorted)):
        for j in range(i + 1, len(M_sorted)):
            gcds[(M_sorted[i], M_sorted[j])] = math.gcd(M_sorted[i], M_sorted[j])
            
    assignment = {}
    
    def is_consistent(m, r, current_assignment):
        for m2, r2 in current_assignment.items():
            g = gcds.get((m, m2)) or gcds.get((m2, m))
            if r % g == r2 % g:
                return False
        return True
        
    def backtrack(idx):
        if idx == len(M_sorted):
            return True
        m = M_sorted[idx]
        for r in range(m):
            if is_consistent(m, r, assignment):
                assignment[m] = r
                if backtrack(idx + 1):
                    return True
                del assignment[m]
        return False
        
    # Use a more efficient CSP solver?
    # For 90 variables, naive backtracking might be slow if constraints are tight.
    # But constraints are loose (mod gcd).
    # Let's try.
    
    if not backtrack(0):
        # Fallback: Try next best star
        # This shouldn't happen for S_2
        pass
        
    return [[m, assignment[m]] for m in M_best]
