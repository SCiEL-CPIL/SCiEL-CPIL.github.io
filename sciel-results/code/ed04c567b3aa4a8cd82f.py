import time
import random
from itertools import combinations

# Precomputation
N = 16
ALL_TRIPLES = list(combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES) # 560

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Generate all K(2,2,2) configurations
# A K(2,2,2) is defined by 3 disjoint pairs.
# We iterate over all sets of 6 vertices.
# For each set of 6, there are 15 ways to partition into 3 pairs.
# We need to be careful not to duplicate.
# Actually, it's easier to iterate over pairs.
# But 120120 is small enough to just generate them.

def generate_k222s():
    k222s = []
    # Iterate over all combinations of 6 vertices
    for verts in combinations(range(N), 6):
        # Generate all 15 partitions of 6 items into 3 pairs
        # We can do this by fixing the partner of the first vertex, etc.
        # v0, v1, v2, v3, v4, v5
        # v0 pairs with v1..v5 (5 choices)
        # remaining 4 pair up in 3 ways
        # Total 5 * 3 = 15
        v = list(verts)
        # Pair v0 with v1, v2, v3, v4, v5
        for i in range(1, 6):
            pair0 = (v[0], v[i])
            remaining = [v[j] for j in range(6) if j != 0 and j != i]
            # Pair remaining 4 into 2 pairs
            # r0 pairs with r1, r2, r3 (3 choices)
            for j in range(1, 4):
                pair1 = (remaining[0], remaining[j])
                pair2 = (remaining[1], remaining[2]) if j == 1 else (
                    (remaining[1], remaining[3]) if j == 2 else (remaining[2], remaining[3])
                )
                # Wait, the logic for pair2 depends on which indices are left.
                # Let's just use a helper or simpler logic.
                # remaining is list of 4.
                # If j=1, remaining[0] pairs with remaining[1]. Left: remaining[2], remaining[3].
                # If j=2, remaining[0] pairs with remaining[2]. Left: remaining[1], remaining[3].
                # If j=3, remaining[0] pairs with remaining[3]. Left: remaining[1], remaining[2].
                
                # Let's re-verify indices.
                # remaining = [r0, r1, r2, r3]
                # j=1: pair1=(r0,r1), pair2=(r2,r3)
                # j=2: pair1=(r0,r2), pair2=(r1,r3)
                # j=3: pair1=(r0,r3), pair2=(r1,r2)
                
                if j == 1:
                    pair2 = (remaining[2], remaining[3])
                elif j == 2:
                    pair2 = (remaining[1], remaining[3])
                else:
                    pair2 = (remaining[1], remaining[2])
                
                # The 8 triples are formed by picking one from each pair
                p0 = pair0
                p1 = pair1
                p2 = pair2
                edges = set()
                for x in p0:
                    for y in p1:
                        for z in p2:
                            edges.add(tuple(sorted((x, y, z))))
                
                # Convert to sorted list of indices for consistency
                edge_indices = sorted([triple_to_idx[e] for e in edges])
                k222s.append(edge_indices)
    return k222s

K222S = generate_k222s()
NUM_K222 = len(K222S)

# Map each edge to the list of K222 indices it belongs to
edge_to_k222s = [[] for _ in range(NUM_TRIPLES)]
for k_idx, edges in enumerate(K222S):
    for e_idx in edges:
        edge_to_k222s[e_idx].append(k_idx)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_count = 0
    best_system = []
    
    # Run multiple random greedy attempts
    # Each attempt shuffles the edges and adds them if valid
    # We can also try to improve with local search if time permits
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        random.shuffle(ALL_TRIPLES)
        
        counts = [0] * NUM_K222
        present = [False] * NUM_TRIPLES
        current_system = []
        
        for t in ALL_TRIPLES:
            e_idx = triple_to_idx[t]
            # Check if adding this edge creates a K(2,2,2)
            valid = True
            for k_idx in edge_to_k222s[e_idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            
            if valid:
                present[e_idx] = True
                current_system.append(t)
                for k_idx in edge_to_k222s[e_idx]:
                    counts[k_idx] += 1
        
        if len(current_system) > best_count:
            best_count = len(current_system)
            best_system = current_system
            
        # If we have a lot of time, we could try to add more edges?
        # The greedy already tries all edges.
        # We could try to remove an edge and add two?
        # For now, just restart.
        
    return best_system
