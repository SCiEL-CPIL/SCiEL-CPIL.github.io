import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    best_set = []
    best_size = 0
    
    # Helper to check validity and size
    def get_size(s):
        return len(s)
        
    # We iterate k from 9 down to 0
    # Max possible size for a given k is (k+1) + count_odds(512 // 2**k)
    # If this max is <= best_size, we can skip
    
    for k in range(9, -1, -1):
        if time.time() > deadline:
            break
            
        max_odd = 512 // (2 ** k)
        # Odd numbers up to max_odd
        odds = list(range(1, max_odd + 1, 2))
        n_odds = len(odds)
        
        # Theoretical max size for this k
        theoretical_max = (k + 1) + n_odds
        if theoretical_max <= best_size:
            continue
            
        # Build compatibility graph for odds
        # compatible[i][j] = True if lcm(odds[i], odds[j]) <= 512
        # We want Max Independent Set in conflict graph, or Max Clique in compatibility graph.
        # Let's find Max Clique in compatibility graph.
        
        # Optimization: If n_odds is small, exact search.
        # If n_odds is large, we might need heuristics or pruning.
        # But n_odds is at most 256.
        
        # Adjacency list for compatibility
        adj = [[] for _ in range(n_odds)]
        for i in range(n_odds):
            for j in range(i + 1, n_odds):
                a = odds[i]
                b = odds[j]
                # lcm(a,b) = a*b // gcd(a,b)
                g = gcd(a, b)
                # Check overflow? a,b <= 512. a*b <= 262144. Fits in int.
                if (a * b) // g <= 512:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        # Bron-Kerbosch with pivot for Max Clique
        # This finds the maximum clique in the compatibility graph.
        
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if time.time() > deadline:
                return
            
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            # Pivot selection: choose u in P U X with max degree in P
            # To simplify, just pick the first one or one with high degree
            # Standard pivot: u in P U X that maximizes |P \ N(u)|
            # Here we want to minimize |P \ N(u)| to reduce recursion?
            # Actually, standard Bron-Kerbosch with pivot:
            # Choose u in P U X.
            # For each v in P \ N(u):
            #   BronKerbosch(R U {v}, P \ (N(v) U {v}), X \ N(v))
            #   P = P \ {v}
            #   X = X U {v}
            
            if not (P or X):
                 if len(R) > len(best_clique):
                     best_clique = list(R)
                 return

            # Select pivot u
            u = -1
            max_deg = -1
            # We can just pick the first vertex in P if P is not empty, else first in X
            # Better: pick u that maximizes the number of non-neighbors in P
            # But for simplicity and speed in Python, picking a random or first vertex might be okay if graph is dense.
            # The compatibility graph is likely dense for small numbers.
            
            # Let's pick u from P if P else X
            if P:
                u = P[0]
            else:
                u = X[0]
                
            # P \ N(u)
            # N(u) are neighbors of u in the full graph
            # We need vertices in P that are NOT neighbors of u
            P_not_Nu = [v for v in P if v not in adj[u]]
            
            for v in P_not_Nu:
                # Recurse
                R.append(v)
                new_P = [x for x in P if x != v and x in adj[v]]
                new_X = [x for x in X if x in adj[v]]
                
                bron_kerbosch(R, new_P, new_X)
                
                R.pop()
                P.remove(v)
                X.append(v)
                
                if time.time() > deadline:
                    return

        # Initial call
        # P = all nodes, X = empty
        P_init = list(range(n_odds))
        X_init = []
        R_init = []
        
        # To handle potential slowness, we can add a simple bound:
        # If len(R) + len(P) <= len(best_clique), prune.
        # Let's modify bron_kerbosch to include this bound.
        
        def bron_kerbosch_bounded(R, P, X):
            nonlocal best_clique
            if time.time() > deadline:
                return
            
            if len(R) + len(P) <= len(best_clique):
                return
                
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            
            if P:
                u = P[0]
            else:
                u = X[0]
                
            P_not_Nu = [v for v in P if v not in adj[u]]
            
            for v in P_not_Nu:
                R.append(v)
                new_P = [x for x in P if x != v and x in adj[v]]
                new_X = [x for x in X if x in adj[v]]
                
                bron_kerbosch_bounded(R, new_P, new_X)
                
                R.pop()
                P.remove(v)
                X.append(v)
                
                if time.time() > deadline:
                    return

        bron_kerbosch_bounded(R_init, P_init, X_init)
        
        current_size = (k + 1) + len(best_clique)
        
        if current_size > best_size:
            best_size = current_size
            # Construct the set
            # Powers of 2: 1, 2, ..., 2^k
            powers_of_2 = [2 ** i for i in range(k + 1)]
            # Odd numbers from best_clique indices
            odd_subset = [odds[i] for i in best_clique]
            best_set = powers_of_2 + odd_subset
            
    # Fallback if no set found (should not happen)
    if not best_set:
        return list(range(1, 23))
        
    return best_set
