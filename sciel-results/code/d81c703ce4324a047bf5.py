"""Improved program for R_4(C6) - 4-colouring of K_n without monochromatic C6."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, permutations

def generate_c6s(n):
    """Generate all C6s in K_n as tuples of 6 vertices."""
    c6s = []
    # A C6 is defined by 6 vertices. 
    # To avoid duplicates, we can fix the first vertex as the smallest?
    # No, a C6 is a cycle. The set of vertices is not enough, the order matters.
    # However, for checking monochromaticity, the order doesn't matter, only the set of edges.
    # But different orders can form different edge sets.
    # Actually, a set of 6 vertices can form multiple C6s.
    # We need to enumerate all distinct C6s (edge sets).
    
    # Number of C6s in K_n:
    # Choose 6 vertices: C(n, 6)
    # Number of distinct C6s on 6 labeled vertices: (6-1)! / 2 = 60.
    
    for verts in combinations(range(n), 6):
        # Generate all 60 distinct cycles on these 6 vertices
        # Fix v0 = verts[0] to reduce permutations?
        # A cycle is v0-v1-v2-v3-v4-v5-v0.
        # Permutations of v1..v5.
        # Total 5! = 120 permutations.
        # Each cycle is counted twice (direction), so 60 distinct cycles.
        for perm in permutations(verts[1:]):
            cycle = (verts[0],) + perm
            # Canonical form to avoid duplicates?
            # Actually, since we iterate all permutations of v1..v5, 
            # and fix v0, do we get duplicates?
            # Cycle (0,1,2,3,4,5) is same as (0,5,4,3,2,1).
            # So we will get each C6 twice.
            # We can just use a set of frozensets of edges?
            # Or just check all 120 and let the set handle duplicates?
            # Better: generate unique cycles.
            pass
            
    # Simpler: Just generate all permutations of 6 vertices, 
    # and use a set of frozensets of edges to store unique C6s.
    unique_c6s = set()
    for verts in combinations(range(n), 6):
        for perm in permutations(verts):
            # Normalize: start with smallest vertex
            # Find index of min
            min_val = min(perm)
            min_idx = perm.index(min_val)
            # Rotate so min is first
            rotated = perm[min_idx:] + perm[:min_idx]
            # To avoid reverse duplicates, ensure second vertex < last vertex?
            # Or just add frozenset of edges.
            edges = frozenset([
                (min(rotated[i], rotated[(i+1)%6]), max(rotated[i], rotated[(i+1)%6]))
                for i in range(6)
            ])
            unique_c6s.add(edges)
            
    return list(unique_c6s)

def construct():
    start_time = time.time()
    deadline = 140
    
    # Precompute C6s for max n=20
    # This might take a bit, but is done once.
    # C(20, 6) * 120 permutations = 38760 * 120 = 4.6M iterations.
    # This might be slow in Python. 
    # Let's optimize generation or just generate on the fly for smaller n?
    # For n=10, C(10,6)*120 = 210 * 120 = 25k. Fast.
    # For n=15, C(15,6)*120 = 5005 * 120 = 600k. OK.
    # For n=20, 4.6M. Might take 2-5s. Acceptable.
    
    all_c6s_20 = generate_c6s(20)
    # Map edge to list of C6 indices containing it
    # Edge is (u, v) with u < v.
    
    best_n = 10
    best_colors = [0] * (10 * 9 // 2)
    
    for n in range(10, 21):
        if time.time() > deadline:
            break
            
        num_edges = n * (n - 1) // 2
        # Filter C6s relevant to this n
        # A C6 is relevant if all its vertices are < n
        c6s = []
        edge_to_c6s = {}
        
        for c6 in all_c6s_20:
            # c6 is frozenset of edges
            # Check if all vertices in edges are < n
            # Max vertex in c6
            max_v = max(max(u, v) for u, v in c6)
            if max_v < n:
                c6s.append(c6)
        
        # Build edge_to_c6s map
        for idx, c6 in enumerate(c6s):
            for e in c6:
                if e not in edge_to_c6s:
                    edge_to_c6s[e] = []
                edge_to_c6s[e].append(idx)
        
        found = False
        # Try multiple restarts
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # Random init
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Initialize bad_c6s
            bad_c6s = set()
            for idx, c6 in enumerate(c6s):
                # Check if monochromatic
                c0 = colors[sum(1 for i in range(n) for j in range(i+1, n) if (i,j) == min(c6[0], max(c6[0])) or (i,j) == min(c6[1], max(c6[1])) or (i,j) == min(c6[2], max(c6[2])) or (i,j) == min(c6[3], max(c6[3])) or (i,j) == min(c6[4], max(c6[4])) or (i,j) == min(c6[5], max(c6[5])))]
                # This index calculation is slow. Let's precompute edge indices.
                pass
            
            # This initialization is too slow. 
            # Let's use a simpler check for the first step.
            
            # Actually, let's just run local search without maintaining bad_c6s set initially,
            # but use a fast validity check to start, then switch to incremental?
            # Or just use the incremental update from a random start, 
            # but we need the initial state.
            
            # Let's compute initial bad_c6s efficiently.
            # For each C6, check 6 edges.
            # Precompute edge indices for each C6.
            c6_edge_indices = []
            for c6 in c6s:
                idxs = []
                for e in c6:
                    u, v = e
                    # Calculate index: sum_{k=0}^{u-1} (n-k-1) + (v-u-1)
                    # Or just use a dictionary
                    idxs.append((u, v))
                c6_edge_indices.append(idxs)
                
            # Map (u,v) to index
            edge_idx_map = {}
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    edge_idx_map[(i,j)] = idx
                    idx += 1
                    
            c6_indices = []
            for c6 in c6s:
                idxs = [edge_idx_map[e] for e in c6]
                c6_indices.append(idxs)
            
            # Initial check
            bad_c6s = set()
            for i, idxs in enumerate(c6_indices):
                c0 = colors[idxs[0]]
                mono = True
                for j in range(1, 6):
                    if colors[idxs[j]] != c0:
                        mono = False
                        break
                if mono:
                    bad_c6s.add(i)
            
            # Local Search
            steps = 0
            max_steps = 5000
            while bad_c6s and steps < max_steps:
                if time.time() > deadline:
                    break
                    
                steps += 1
                # Pick a random bad C6
                bad_idx = random.choice(list(bad_c6s))
                idxs = c6_indices[bad_idx]
                
                # Pick a random edge in this C6
                e_idx = random.choice(idxs)
                
                # Flip color
                old_c = colors[e_idx]
                new_c = random.randint(0, 3)
                while new_c == old_c:
                    new_c = random.randint(0, 3)
                    
                colors[e_idx] = new_c
                
                # Update bad_c6s for all C6s containing this edge
                # Get list of C6 indices
                # We need edge_to_c6s map for this n
                # Rebuild edge_to_c6s for this n?
                # We did it above but didn't store it properly for the loop.
                # Let's assume we have it.
                
                # Actually, we need to update the status of all C6s that contain edge e_idx.
                # For each such C6, check if it is now monochromatic.
                # Remove from bad_c6s if not, add if yes.
                
                # We need the map: edge_idx -> list of c6_indices
                # Let's build it once per n.
                pass
                
            # This is getting messy. Let's simplify.
            # Just use the "find one C6" approach.
            
            # Restart with simpler logic
            break
            
        # Fallback: Simple local search
        for attempt in range(50):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Helper to get edge index
            def get_idx(u, v):
                if u > v: u, v = v, u
                # Precomputed map
                return edge_idx_map[(u,v)]
                
            # Precompute C6s for this n as list of edge indices
            local_c6s = []
            for c6 in c6s:
                idxs = tuple(edge_idx_map[e] for e in c6)
                local_c6s.append(idxs)
                
            # Local search
            for _ in range(1000):
                if time.time() > deadline:
                    break
                # Find a C6
                found_c6 = None
                # Random sample of C6s to find a bad one
                for _ in range(100):
                    c6 = random.choice(local_c6s)
                    c0 = colors[c6[0]]
                    if all(colors[x] == c0 for x in c6[1:]):
                        found_c6 = c6
                        break
                        
                if not found_c6:
                    # No C6 found in sample, might be valid
                    # Verify fully?
                    # For now, assume valid if no bad C6 found in 100 samples
                    # This is risky.
                    
                    # Let's do a full check for n <= 12, else trust sample?
                    # Full check:
                    valid = True
                    for c6 in local_c6s:
                        c0 = colors[c6[0]]
                        if all(colors[x] == c0 for x in c6[1:]):
                            valid = False
                            break
                            
                    if valid:
                        best_n = n
                        best_colors = colors[:]
                        found = True
                        break
                        
                # Flip random edge in found_c6
                e_idx = random.choice(found_c6)
                old_c = colors[e_idx]
                new_c = random.randint(0, 3)
                while new_c == old_c:
                    new_c = random.randint(0, 3)
                colors[e_idx] = new_c
                
        if found:
            continue
        else:
            break
            
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
