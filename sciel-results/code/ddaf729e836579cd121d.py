"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    d = 7
    n = 2 ** d
    edges = []
    edge_to_idx = {}
    
    # 1. Generate all edges
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
    
    m = len(edges) # 448
    
    # 2. Pre-compute face structures
    # A face is a 4-cycle. We represent each face by the set of 4 edge indices it contains.
    # Constraint: No face may have all 4 edges selected.
    # This is equivalent to: For each face, at most 3 edges can be in the selected set.
    
    face_set = set()
    for x in range(n):
        for i in range(d):
            for j in range(i+1, d):
                y = x ^ (1 << i)
                z = x ^ (1 << j)
                w = x ^ (1 << i) ^ (1 << j)
                e1 = (min(x, y), max(x, y))
                e2 = (min(y, w), max(y, w))
                e3 = (min(w, z), max(w, z))
                e4 = (min(z, x), max(z, x))
                idxs = (edge_to_idx[e1], edge_to_idx[e2], edge_to_idx[e3], edge_to_idx[e4])
                face_set.add(frozenset(idxs))
                
    faces = list(face_set)
    num_faces = len(faces)
    face_edges = [list(f) for f in faces]
    
    # edge_faces[ei] = list of face indices containing edge ei
    edge_faces = [[] for _ in range(m)]
    for fi, f in enumerate(face_edges):
        for ei in f:
            edge_faces[ei].append(fi)
    
    # Pre-compute for each edge, the "conflict edges" in each face it belongs to?
    # Actually, for SA, we need to quickly check if adding edge ei creates a violation.
    # Violation occurs if there exists a face fi containing ei such that the other 3 edges in fi are already selected.
    # So for each edge ei and each face fi it belongs to, we need to know the other 3 edges.
    
    # Let's precompute for each (ei, fi) pair, the list of the other 3 edges in face fi.
    # Since each edge belongs to d-1 = 6 faces, and each face has 4 edges, this is small.
    
    # edge_face_other_edges[ei][fi] = [e1, e2, e3] where {e1,e2,e3,ei} is face fi
    # But fi is an index, not a key. Let's use a dict or just compute on the fly?
    # On the fly is O(4) per check, which is fast.
    
    # 3. Optimization Strategy:
    # The problem is a Max-SAT / Independent Set in a hypergraph variant.
    # We want to maximize |S| such that for all faces F, |S ∩ F| <= 3.
    
    # Strategy:
    # 1. Start with a random valid set (or empty).
    # 2. Use Simulated Annealing or Greedy with Local Search.
    #    Move: Add a random unselected edge if it doesn't violate constraints.
    #    Move: Remove a random selected edge.
    #    Move: Swap? (Remove one, add another)
    
    # A powerful heuristic for "Maximize edges s.t. no hyperedge is fully covered" is:
    # Start with all edges selected.
    # While there is a violated face (4 edges selected):
    #    Remove one edge from that face (randomly or heuristically).
    # This is a "Repair" approach.
    
    # Let's try the Repair approach first, then refine with SA.
    
    def repair(selected, face_counts):
        """
        selected: list of booleans
        face_counts: list of counts for each face
        Returns: repaired selected, face_counts, number of removals
        """
        # Find all violated faces (count == 4)
        # We can iterate until no violations.
        # To avoid infinite loops or slow convergence, we pick a random violated face and remove a random edge from it.
        
        # We need to maintain face_counts dynamically.
        
        # Let's implement a loop that continues as long as there are violations.
        # Since we only remove edges, it will terminate.
        
        # To speed up finding violations, we can keep a list of violated faces?
        # Or just scan? 112 faces is small.
        
        # However, removing an edge reduces counts of its 6 faces.
        
        # Let's just do a while loop.
        
        # Initial state: all True
        # face_counts initially: each face has 4 edges selected?
        # Wait, if we start with ALL edges selected, every face has 4 edges.
        # So we have 112 violations.
        
        # We remove edges until valid.
        
        # Let's implement a greedy removal:
        # While exists face with count 4:
        #   Pick a random such face.
        #   Pick a random edge from that face.
        #   Remove it.
        
        # This is O(V * Removals).
        
        # Better:
        # We can prioritize removing edges that are in many violated faces?
        # Or just random.
        
        # Let's use a queue of violated faces.
        
        # Actually, since the graph is small, a simple loop is fine.
        
        # But we want to MAXIMIZE edges. Removing randomly might remove too many.
        # We want to remove the "least useful" edge.
        # An edge is useful if it is in many faces that are currently NOT violated?
        # No, we are in a repair phase.
        
        # Let's try:
        # 1. Start with all edges.
        # 2. While violated:
        #    Pick random violated face.
        #    Pick random edge from it.
        #    Remove.
        
        # Then, try to ADD edges back?
        # Greedy Add:
        # While exists edge that can be added without violation:
        #    Add it.
        
        # This "Remove then Add" cycle is a good local search.
        
        # Let's implement this as a single function `optimize()`.
        
        pass # Placeholder, will implement in main logic below.

    # --- Implementation of the Optimization Loop ---
    
    # We will use a Simulated Annealing approach on the state of edges.
    # State: selected (list of bool)
    # Cost: Number of violated faces (should be 0).
    # Objective: Maximize count.
    
    # To keep it simple and robust within time limits:
    # 1. Initialize with all edges.
    # 2. Repair to valid state (remove edges from violated faces).
    # 3. Local Search: Try to add edges. If adding creates a violation, remove one edge from the violated face (swap).
    
    # Let's implement "Greedy Repair + Greedy Add" with randomization and restarts.
    
    best_count = 0
    best_selected = [False] * m
    
    # We can run multiple iterations of this process, keeping the best.
    
    # Helper: Check if adding edge ei is valid given current face_counts
    def can_add(ei, face_counts):
        for fi in edge_faces[ei]:
            if face_counts[fi] == 3:
                return False
        return True
    
    # Helper: Add edge ei, update face_counts
    def add_edge(ei, selected, face_counts):
        selected[ei] = True
        for fi in edge_faces[ei]:
            face_counts[fi] += 1
    
    # Helper: Remove edge ei, update face_counts
    def remove_edge(ei, selected, face_counts):
        selected[ei] = False
        for fi in edge_faces[ei]:
            face_counts[fi] -= 1
    
    # Main Loop: Multiple Randomized Greedy Runs
    # Each run:
    # 1. Start with all edges.
    # 2. Repair: While violated, remove random edge from random violated face.
    # 3. Improve: While possible, add random edge that can be added.
    #    To make it more effective, we can try to add edges in a random order.
    
    num_runs = 100 # We will adjust based on time
    
    for run in range(num_runs):
        if time.time() > deadline:
            break
            
        selected = [True] * m
        count = m
        
        # Compute initial face_counts
        # Since all edges are selected, every face has count 4.
        face_counts = [4] * num_faces
        
        # Repair Phase
        # We need a list of violated faces.
        # Initially all faces are violated.
        violated_faces = list(range(num_faces))
        
        # We remove edges until no violated faces.
        # To avoid O(N^2) scanning for violated faces, we maintain the list.
        # When we remove an edge, we update counts of its 6 faces.
        # If a face count drops from 4 to 3, it is no longer violated.
        # If a face count was 3 and we remove an edge from it, it becomes 2 (still not violated).
        
        # So we just need to check if a face becomes valid.
        
        # Let's use a while loop on the list of violated faces.
        # We pick a random index from the list, get the face, pick a random edge from it, remove it.
        # Then we check which of the 6 faces containing that edge became valid, and remove them from the list?
        # Or just filter the list? Filtering is O(N).
        
        # Given N=112, filtering is fast.
        
        while violated_faces:
            if time.time() > deadline:
                break # Abort this run if too slow, though repair is usually fast
            
            # Pick random violated face
            fi_idx = random.randint(0, len(violated_faces) - 1)
            fi = violated_faces[fi_idx]
            
            # Pick random edge from this face
            ei = random.choice(face_edges[fi])
            
            # Remove edge
            remove_edge(ei, selected, face_counts)
            count -= 1
            
            # Update violated_faces list
            # Check the 6 faces containing ei.
            # If face_counts[fi] < 4, it is no longer violated.
            new_violated = []
            for f in violated_faces:
                if face_counts[f] == 4:
                    new_violated.append(f)
            violated_faces = new_violated
            
            # Optimization: Instead of full scan, we can just check the neighbors of ei.
            # But full scan is O(112) which is negligible.
            
        # Now the set is valid.
        
        # Improve Phase: Try to add edges back.
        # We can try to add edges in a random order.
        # If an edge can be added, add it.
        # This is a greedy maximal independent set (in the hypergraph sense).
        
        # To get different results, we shuffle the order of edges.
        order = list(range(m))
        random.shuffle(order)
        
        for ei in order:
            if not selected[ei]:
                if can_add(ei, face_counts):
                    add_edge(ei, selected, face_counts)
                    count += 1
        
        # Update best
        if count > best_count:
            best_count = count
            best_selected = selected[:]
            
        # Optional: Local Search / SA on top of this?
        # Given time constraints, the randomized greedy repair+add might be sufficient to find good solutions.
        # The "repair" step removes edges to break cycles. The "add" step fills back in where possible.
        # Different random orders in repair and add lead to different local optima.
        
        # If we have time, we could do a few steps of SA here.
        # But let's stick to multiple randomized greedy runs first.
        
    # Return best found
    out = []
    for ei in range(m):
        if best_selected[ei]:
            u, v = edges[ei]
            out.append([u, v])
            
    return out
# EVOLVE-BLOCK-END
