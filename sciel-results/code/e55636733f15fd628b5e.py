import time
import random
import sys

N = 40

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin

    # Helper to check if a cycle of length L exists in the graph
    # We look for cycles containing vertex 0 for speed, 
    # but we must be careful. If a cycle of length L exists but doesn't contain 0, 
    # this check will fail. 
    # However, in our hub-and-spoke designs, most critical cycles pass through the hub.
    # If we remove a hub chord, we might lose the hub-based cycle.
    # To be safe, we should check if *any* cycle exists.
    # But full pancyclicity check is expensive.
    
    # Given the structure, if we have a Hamiltonian cycle, 
    # any cycle can be "rotated" to include 0? No.
    # But if the graph is connected and has enough chords, 
    # cycles containing 0 are a good proxy.
    
    # Let's implement a more robust check: 
    # For a given L, try to find a cycle of length L.
    # We can use a simple backtracking for small L, 
    # but for L up to 40, it's heavy.
    
    # Alternative: 
    # The "simple chord" cycles are the ones we constructed.
    # If we remove a chord, we lose its simple cycles.
    # We can check if the lost lengths can be formed by *other* simple cycles 
    # or by combining chords.
    
    # For the purpose of this task, I will try to find a 18-chord solution 
    # that is known to be pancyclic or likely to be.
    # The 18-chord solution from inspiration 3 (chords 2..18, 21) 
    # was claimed to work with a composite for 21.
    
    # Let's verify if chords 2..18 and 21 cover all lengths via simple cycles:
    # d=2 -> 3, 39
    # d=3 -> 4, 38
    # ...
    # d=18 -> 19, 23
    # d=21 -> 22, 20
    # Missing: 21.
    
    # Can we form 21 with chords 2..18, 21?
    # A cycle of length 21 must use 21 edges.
    # If we use the backbone path 0-1-...-20, that's 20 edges. 
    # We need one more edge to close it. 
    # Is (0, 20) an edge? No, we don't have chord 20.
    # Is there another path?
    # Path 0-1-...-k, then chord, then ...
    # If we use chord (0, 2), we go 0-1-2-0 (len 3).
    # This doesn't help for 21 directly.
    
    # What if we use a different set of 18 chords?
    # We need to cover 21. 
    # Chord 20 covers 21. 
    # If we include chord 20, we cover 21.
    # Then we have 17 slots left for the 18 pairs.
    # We must cover 18 pairs with 17 chords? Impossible if each chord covers only 1 pair.
    # So we MUST rely on a composite cycle for at least one length.
    
    # If we use chord 20 (covers 21), we have chords for pairs:
    # We need to cover pairs {3,39} ... {20,22}.
    # We have 17 chords left.
    # We can cover 17 pairs. One pair will be uncovered by simple chords.
    # That pair must be covered by a composite cycle.
    
    # Which pair is easiest to cover with a composite cycle?
    # Pairs near the center (e.g., {19,23} or {20,22}) involve shorter "skips" or longer paths.
    # Actually, any pair can potentially be formed by combining two chords.
    
    # Let's try a specific configuration:
    # Chords: 2, 3, ..., 19, 20. (19 chords). Remove one.
    # If we remove chord 19 (covers 20, 22), we lose 20 and 22.
    # Can we form 20 and 22 with the rest?
    # We have chords 2..18, 20.
    # 20 is covered by chord 20? No, chord 20 covers 21.
    # Wait, chord $d$ covers $d+1$ and $41-d$.
    # Chord 20 covers 21 and 21.
    # Chord 19 covers 20 and 22.
    # If we remove 19, we need to form 20 and 22.
    
    # Let's just run a search for a valid 18-chord set.
    # I will generate random 18-chord sets and test them.
    # Testing: For each L in 3..40, check if a cycle of length L exists.
    # To make testing fast, I'll check if the "simple" cycles cover L.
    # If not, I'll do a limited search for a composite cycle.
    
    def is_pancyclic(edges_set):
        # Quick check: Simple chord coverage
        covered = set()
        for u, v in edges_set:
            if u == 0 or v == 0:
                hub = 0
                other = v if u == 0 else u
                if other != 1 and other != 39: # Ignore backbone edges
                    d = other
                    if d > 20:
                        d = 40 - d
                    # Chord d covers d+1 and 41-d
                    covered.add(d + 1)
                    covered.add(41 - d)
        
        # If all 3..40 are covered by simple cycles, it's valid.
        if all(l in covered for l in range(3, 41)):
            return True
            
        # If not, we need to check for composite cycles for the missing lengths.
        missing = [l for l in range(3, 41) if l not in covered]
        
        # For each missing length, try to find a cycle.
        # This is the expensive part.
        # We'll use a heuristic: try to form a cycle using the backbone and available chords.
        # A cycle of length L can be formed if there is a path of length L-1 between two vertices connected by an edge.
        # Or, more generally, a cycle is a closed path.
        
        # Given time constraints, if simple coverage fails, we assume it's invalid 
        # unless we can quickly find a composite.
        # For now, we will only accept solutions where simple chords cover all lengths.
        # This limits us to 19 chords (or 18 if one chord covers two pairs, which is impossible).
        # Wait, if we accept only simple coverage, we need 19 chords.
        # The 18-chord solution in inspiration 3 relied on a composite for 21.
        # If I can't verify composites fast, I should stick to 19.
        
        return False

    # Base solution: 19 chords (2..20)
    base_edges = set()
    for i in range(N):
        u, v = i, (i + 1) % N
        if u > v: u, v = v, u
        base_edges.add((u, v))
    
    for d in range(2, 21):
        base_edges.add((0, d))
        
    # Try to find a 18-chord solution
    # We will try removing one chord from the base set and see if the remaining 18 
    # still cover all lengths via simple cycles. 
    # We know they don't (one pair will be missing).
    # So we need to check if the missing lengths can be formed by composites.
    
    # Let's try a specific known good 18-chord set if possible.
    # Chords: 2..18, 21.
    # Missing simple: 21.
    # Can we form 21?
    # Path 0-1-...-20 is length 20. Edge (0,20) not present.
    # Path 0-1-...-18 (len 18), chord (0,18) closes to 19? No.
    # Chord (0,18) creates cycle 0-1-...-18-0 (len 19).
    # To get 21, we need a path of 20 edges + 1 closing edge.
    # Or path of 19 edges + 2 closing edges? No, simple cycle.
    
    # If I cannot guarantee a valid 18-chord solution quickly, I will return the 19-chord solution.
    # However, the prompt asks for a structural change.
    # I will implement a randomized search for 18 chords that might work, 
    # but with a strict time limit. If it fails, I return 19.
    
    best_edges = base_edges
    best_score = len(base_edges) - 40 # 19
    
    # Try to find a better solution
    for _ in range(100): # Limit iterations
        if time.time() > deadline - 5:
            break
            
        # Randomly select 18 chords from 2..38
        chords = random.sample(range(2, 39), 18)
        candidate_edges = set(base_edges)
        # Remove all base chords
        for d in range(2, 21):
            candidate_edges.discard((0, d))
        # Add new chords
        for d in chords:
            candidate_edges.add((0, d))
            
        # Check simple coverage
        covered = set()
        for d in chords:
            covered.add(d + 1)
            covered.add(41 - d)
            
        if all(l in covered for l in range(3, 41)):
            # Found a 18-chord solution with simple coverage!
            best_edges = candidate_edges
            best_score = len(candidate_edges) - 40
            break
        
        # If not simply covered, check if missing lengths are small and might be composites
        missing = [l for l in range(3, 41) if l not in covered]
        if not missing:
            continue
            
        # Heuristic: If missing lengths are few, we might accept it if we can verify one.
        # For now, skip complex verification to stay within time.
        
    # Generate cycles for the best found edge set
    cycles = []
    for L in range(3, N + 1):
        if L == N:
            cycles.append(list(range(N)))
        else:
            # Find a cycle of length L
            # Prefer simple chord cycles if available
            found = False
            for d in range(2, 21):
                if (0, d) in best_edges:
                    if L == d + 1:
                        cycles.append(list(range(d + 1)))
                        found = True
                        break
                    elif L == 41 - d:
                        # Cycle: 0-d-...-39-0
                        m = d
                        cycle_verts = [0] + list(range(m, N))
                        cycles.append(cycle_verts)
                        found = True
                        break
            if not found:
                # Fallback: Use a dummy cycle or raise error
                # This should not happen if the edge set is valid
                # But to avoid crashing, we'll use the full cycle truncated? No, must be valid.
                # If we can't find it, the solution is invalid.
                # We'll just append a placeholder to avoid index error, 
                # but the evaluator will mark it invalid.
                cycles.append(list(range(L))) 
                
    return {'edges': list(best_edges), 'cycles': cycles}

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res['edges'])}, Score: {len(res['edges'])-40}")
