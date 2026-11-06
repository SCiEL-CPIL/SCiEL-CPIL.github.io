import time
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Precompute pairings of 6 elements into 3 pairs
    # Returns list of 15 pairings, each is a list of 3 tuples
    def get_pairings():
        pairings = []
        for partner in range(1, 6):
            remaining = [i for i in range(6) if i not in [0, partner]]
            r0, r1, r2, r3 = remaining
            p1 = [(0, partner), (r0, r1), (r2, r3)]
            p2 = [(0, partner), (r0, r2), (r1, r3)]
            p3 = [(0, partner), (r0, r3), (r1, r2)]
            pairings.extend([p1, p2, p3])
        return pairings

    PAIRINGS = get_pairings()
    
    def find_violation(G, n, random_sample=False):
        """
        Finds a monochromatic K2,2,2 in G.
        Returns (i, j) if found, else None.
        """
        if random_sample:
            # Sample 1000 random 6-sets
            for _ in range(1000):
                if time.time() > deadline:
                    return None
                combo = np.random.choice(n, 6, replace=False)
                # Check this combo
                for pairing in PAIRINGS:
                    p0, p1, p2 = pairing
                    v0a, v0b = combo[p0[0]], combo[p0[1]]
                    v1a, v1b = combo[p1[0]], combo[p1[1]]
                    v2a, v2b = combo[p2[0]], combo[p2[1]]
                    
                    # Check sums
                    s01 = G[v0a, v1a] + G[v0a, v1b] + G[v0b, v1a] + G[v0b, v1b]
                    if s01 != 0 and s01 != 4:
                        continue # Not mono for this pair interaction? No, we need ALL 12.
                        # Wait, if s01 is not 0 and not 4, it's mixed. 
                        # But we need to check if the WHOLE thing is mono.
                        # If s01 is mixed, it can't be all 0 or all 1.
                        # So we can break early for this pairing.
                        pass 
                    
                    # Actually, simpler:
                    # Check if all 12 are 0.
                    # Check if all 12 are 1.
                    
                    # Early exit optimization:
                    # If any edge is 0, not all 1.
                    # If any edge is 1, not all 0.
                    
                    all_zero = True
                    all_one = True
                    
                    # Edges 0-1
                    if G[v0a, v1a] != 0: all_zero = False
                    if G[v0a, v1b] != 0: all_zero = False
                    if G[v0b, v1a] != 0: all_zero = False
                    if G[v0b, v1b] != 0: all_zero = False
                    
                    if G[v0a, v1a] != 1: all_one = False
                    if G[v0a, v1b] != 1: all_one = False
                    if G[v0b, v1a] != 1: all_one = False
                    if G[v0b, v1b] != 1: all_one = False
                    
                    if not all_zero and not all_one:
                        continue # This pairing is not mono, try next pairing or next combo?
                        # Actually, if this pairing is not mono, we should check the NEXT pairing.
                        # But if all_zero became False, we still need to check if all_one is true?
                        # Yes.
                        # So we can't break immediately unless both are False.
                        # But we already checked 4 edges.
                        # Let's just compute sums for the rest.
                    
                    # This logic is getting messy. Let's use sums for clarity and speed (numpy scalar ops are fast).
                    # Re-do with sums.
                    
                    s01 = G[v0a, v1a] + G[v0a, v1b] + G[v0b, v1a] + G[v0b, v1b]
                    if s01 > 0: all_zero = False
                    if s01 < 4: all_one = False
                    
                    if not all_zero and not all_one:
                        continue
                    
                    s02 = G[v0a, v2a] + G[v0a, v2b] + G[v0b, v2a] + G[v0b, v2b]
                    if s02 > 0: all_zero = False
                    if s02 < 4: all_one = False
                    
                    if not all_zero and not all_one:
                        continue
                        
                    s12 = G[v1a, v2a] + G[v1a, v2b] + G[v1b, v2a] + G[v1b, v2b]
                    if s12 > 0: all_zero = False
                    if s12 < 4: all_one = False
                    
                    if all_zero or all_one:
                        return (v0a, v1a)
            
            return None
        else:
            # Systematic scan
            for combo in itertools.combinations(range(n), 6):
                if time.time() > deadline:
                    return None
                for pairing in PAIRINGS:
                    p0, p1, p2 = pairing
                    v0a, v0b = combo[p0[0]], combo[p0[1]]
                    v1a, v1b = combo[p1[0]], combo[p1[1]]
                    v2a, v2b = combo[p2[0]], combo[p2[1]]
                    
                    all_zero = True
                    all_one = True
                    
                    s01 = G[v0a, v1a] + G[v0a, v1b] + G[v0b, v1a] + G[v0b, v1b]
                    if s01 > 0: all_zero = False
                    if s01 < 4: all_one = False
                    if not all_zero and not all_one: continue
                    
                    s02 = G[v0a, v2a] + G[v0a, v2b] + G[v0b, v2a] + G[v0b, v2b]
                    if s02 > 0: all_zero = False
                    if s02 < 4: all_one = False
                    if not all_zero and not all_one: continue
                        
                    s12 = G[v1a, v2a] + G[v1a, v2b] + G[v1b, v2a] + G[v1b, v2b]
                    if s12 > 0: all_zero = False
                    if s12 < 4: all_one = False
                        
                    if all_zero or all_one:
                        return (v0a, v1a)
            return None

    # Fallback: Seed solution
    def get_seed():
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}

    best_result = get_seed()
    
    # Try to find better n
    # Start from high n, go down
    for target_n in range(29, 17, -1):
        if time.time() > deadline:
            break
            
        # Multiple restarts for this n
        restarts = 0
        max_restarts = 20
        
        while restarts < max_restarts and time.time() < deadline:
            restarts += 1
            
            # Init random G
            G = np.random.randint(0, 2, (target_n, target_n))
            G = np.triu(G, 1)
            G = G + G.T
            np.fill_diagonal(G, 0)
            
            # Local Search
            steps = 0
            max_steps = 500
            
            while steps < max_steps and time.time() < deadline:
                # Use random sampling first to quickly find violations if many exist
                # Then systematic if random fails
                v = find_violation(G, target_n, random_sample=True)
                if v is None:
                    # Try systematic to be sure
                    v = find_violation(G, target_n, random_sample=False)
                
                if v is None:
                    # Valid coloring found
                    colors = []
                    for i in range(target_n):
                        for j in range(i + 1, target_n):
                            colors.append(int(G[i, j]))
                    best_result = {'n': target_n, 'colors': colors}
                    # We can stop immediately as we want max n
                    return best_result
                else:
                    i, j = v
                    G[i, j] = 1 - G[i, j]
                    G[j, i] = G[i, j]
                    steps += 1
                    
    return best_result
