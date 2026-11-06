import math
import time
import random

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    moduli = list(range(2, 129))
    n = len(moduli)
    
    # Precompute GCDs
    gcds = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = _gcd(moduli[i], moduli[j])
            gcds[i][j] = g
            gcds[j][i] = g

    best_count = 0
    best_result = []
    
    # Helper to check if residue r for modulus index i is disjoint from selected list
    # selected: list of (idx, residue)
    def is_valid(i, r, selected):
        for (j, rj) in selected:
            g = gcds[i][j]
            if (r - rj) % g == 0:
                return False
        return True

    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Random order of moduli
        indices = list(range(n))
        random.shuffle(indices)
        
        selected = []
        
        for i in indices:
            m = moduli[i]
            # Try residues. 
            # Heuristic: Try random residues.
            # To speed up, we can limit tries per modulus? 
            # If we fail to find one, we skip this modulus.
            # Since we want max count, skipping is bad, but if no residue works, we must skip.
            
            # Optimization: 
            # The number of invalid residues for a given (i, selected) is at most sum of (m/gcd) ?
            # Actually, for each constraint, m/gcd residues are invalid.
            # If sum(m/gcd) > m, it's possible no solution exists? 
            # No, it's a union of sets.
            
            # Just try random residues. 
            # If m is large, random is good.
            
            found = False
            # Try a limited number of random residues first? 
            # If we don't find one in 100 tries, try systematic?
            # Systematic is m iterations. m <= 128. So 128 checks is fast.
            # Just iterate all residues in random order?
            # Or just iterate 0..m-1?
            # Random order might find a solution faster on average?
            # Let's just iterate 0..m-1. It's deterministic and fast enough.
            
            # Actually, if we shuffle residues, we might find a "good" one faster?
            # But 128 is small. Let's just iterate.
            
            for r in range(m):
                if is_valid(i, r, selected):
                    selected.append((i, r))
                    found = True
                    break
            
            # If not found, we skip this modulus.
            # This greedy might get stuck with a small set if early choices block later ones.
            # Randomizing the order of moduli helps.
            
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, moduli[i]] for i, r in selected]
            
            # Early stop if we found a very good solution?
            # Max possible is 127. 
            # If we get > 100, it's likely near optimal.
            if best_count > 100:
                # We can try a few more times to polish, but return soon
                pass

    if not best_result:
        # Fallback: return a small valid set
        # (1,2), (0,4) -> gcd(2,4)=2, 1-0=1, 1%2!=0. OK.
        # (1,2), (0,4), (1,6)? 
        # gcd(6,2)=2, 1-1=0, 0%2==0. Fail.
        # Let's just return the first few from a quick greedy run if empty?
        # The loop above should always find at least 1.
        return [[1, 2]]
        
    return best_result
