import numpy as np
import time
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    d = 7
    n = 1 << d
    num_edges = d * (1 << (d - 1)) # 448
    
    # 1. Precompute Edge-Index Mapping
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_to_idx[(u, v)] = idx
                idx_to_edge.append((u, v))
                idx += 1
                
    assert idx == num_edges
    
    # 2. Precompute Face Constraints
    # A 2D face is defined by two distinct bit positions i, j and a base vertex u 
    # where bits i and j are 0 in u.
    # The 4 edges of the face are:
    # e1: (u, u^i), e2: (u, u^j), e3: (u^i, u^i^j), e4: (u^j, u^i^j)
    
    faces = []
    edge_to_faces = [[] for _ in range(num_edges)]
    
    for i in range(d):
        for j in range(i + 1, d):
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u
                    v2 = u ^ (1 << i)
                    v3 = u ^ (1 << j)
                    v4 = u ^ (1 << i) ^ (1 << j)
                    
                    e1 = (min(v1, v2), max(v1, v2))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v2, v4), max(v2, v4))
                    e4 = (min(v3, v4), max(v3, v4))
                    
                    ie1 = edge_to_idx[e1]
                    ie2 = edge_to_idx[e2]
                    ie3 = edge_to_idx[e3]
                    ie4 = edge_to_idx[e4]
                    
                    face_indices = (ie1, ie2, ie3, ie4)
                    faces.append(face_indices)
                    
                    for ie in face_indices:
                        edge_to_faces[ie].append(len(faces) - 1)
                        
    num_faces = len(faces)
    
    # 3. Initial Construction: Greedy Random
    # We use a greedy approach to get a large valid set quickly.
    # Then we use Max-Flow based swaps to improve it.
    
    def greedy_construct(seed):
        import random
        random.seed(seed)
        edges = list(range(num_edges))
        random.shuffle(edges)
        selected = np.zeros(num_edges, dtype=bool)
        
        # For fast validity check during greedy phase
        # We maintain a count of selected edges in each face
        face_counts = np.zeros(num_faces, dtype=np.int8)
        
        for e in edges:
            # Check if adding e violates any face
            # It violates if any face containing e already has 3 selected edges
            valid = True
            for f_idx in edge_to_faces[e]:
                if face_counts[f_idx] == 3:
                    valid = False
                    break
            if valid:
                selected[e] = True
                for f_idx in edge_to_faces[e]:
                    face_counts[f_idx] += 1
                    
        return selected, face_counts

    # 4. Max-Flow based Improvement
    # We model the problem of improving the set by swapping edges.
    # Let S be the current set of selected edges.
    # We want to find a set of edges R to remove and A to add such that:
    # 1. (S \ R) U A is valid.
    # 2. |A| > |R| (strictly increases count).
    # 3. The change is "small" (local swap).
    
    # This can be framed as finding a negative cost cycle in an exchange graph, 
    # or more simply, using a Max-Flow formulation where:
    # - Source connects to edges we want to ADD (capacity 1, cost -1? No, let's use standard flow).
    # - Actually, a common trick for "Max Independent Set in 4-uniform hypergraph" 
    #   with small swaps is to model it as a bipartite matching or flow problem 
    #   if we fix the "removals".
    
    # Alternative robust approach: 
    # We try to find a set of edges to remove such that the number of edges we 
    # can add back (and more) is greater.
    
    # Let's use a simpler, highly effective heuristic that is structurally different:
    # **Iterative Randomized Repair with Constraint Propagation**
    # But the prompt suggests Max-Flow from inspiration 2.
    
    # Let's implement a Max-Flow solver (Dinic) to solve the following:
    # We select a subset of "blocked" faces. We want to unblock them by removing edges.
    # Then we add all possible edges.
    
    # Actually, the most effective "different" strategy that works well for this specific 
    # problem (ex(Q7, C4)) is often a **Genetic Algorithm** or **Memetic Algorithm** 
    # combined with the **Max-Flow** local optimizer.
    
    # Given the time constraint and the need for a single file, I will implement 
    # a **Dinic's Max-Flow** based local search that looks for "swap" improvements 
    # of size k (remove k, add k+1).
    
    # For simplicity and speed in Python, I will implement a **Greedy Max-Flow** 
    # where we model the "conflict graph" of edges.
    # Two edges conflict if they are in a face with 2 other selected edges? No.
    
    # Let's stick to the **Max-Flow** idea from Inspiration 2 more concretely:
    # We construct a flow network where:
    # - Nodes represent edges.
    # - We want to find a set of edges to remove R and add A.
    # - This is equivalent to finding a path in the "exchange graph".
    
    # A simpler, very effective structural change: **Simulated Annealing with 
    # "Smart" Moves**. Instead of random flips, we generate moves by picking a 
    # random "blocked" edge (an edge not in S that cannot be added because it 
    # completes a face) and trying to remove one of the 3 blocking edges.
    
    # This is a **Guided Local Search** (GLS) / **Tabu Search** variant.
    
    def guided_local_search(selected, face_counts, seed):
        import random
        random.seed(seed)
        
        current_count = np.sum(selected)
        best_count = current_count
        best_selected = selected.copy()
        best_face_counts = face_counts.copy()
        
        T = 10.0
        cooling = 0.9995
        steps = 0
        max_steps = 5000
        
        # Precompute for each edge, the faces it is in
        # And for each face, the edges in it
        
        while steps < max_steps and time.time() < deadline:
            # Identify "blocked" edges: edges not in selected that are in at least one face 
            # where the other 3 edges are selected.
            # We pick one such edge randomly to try to "fix".
            
            # To speed up, we can maintain a list of blocked edges? 
            # Or just pick a random edge not in selected and check.
            
            e_add = random.randint(0, num_edges - 1)
            
            if selected[e_add]:
                # If it's already selected, we try to remove a random selected edge 
                # to create space? No, that decreases count. 
                # We only want to remove if it leads to a net gain.
                # In GLS, we usually penalize "bad" moves.
                # Let's just skip if we pick a selected edge, or try to remove it 
                # if it's "critical" for many faces?
                # For simplicity, we focus on adding.
                continue
            
            # Check if e_add is blocked
            blocking_faces = []
            for f_idx in edge_to_faces[e_add]:
                if face_counts[f_idx] == 3:
                    blocking_faces.append(f_idx)
            
            if not blocking_faces:
                # It's not blocked, add it!
                selected[e_add] = True
                for f_idx in edge_to_faces[e_add]:
                    face_counts[f_idx] += 1
                current_count += 1
                
                if current_count > best_count:
                    best_count = current_count
                    best_selected = selected.copy()
                    best_face_counts = face_counts.copy()
                continue
            
            # It is blocked. We must remove at least one edge from one of the blocking faces.
            # Pick a random blocking face
            f_target = random.choice(blocking_faces)
            f_edges = faces[f_target]
            
            # Pick a random edge from this face to remove
            # We must not pick e_add itself (it's not in the face count of 3? 
            # Wait, if face_counts[f] == 3, the 3 edges are selected. e_add is the 4th.
            # So the 3 selected edges are in f_edges and != e_add.
            
            candidates_to_remove = []
            for oe in f_edges:
                if oe != e_add and selected[oe]:
                    candidates_to_remove.append(oe)
            
            if not candidates_to_remove:
                continue
                
            e_remove = random.choice(candidates_to_remove)
            
            # Perform the move: Remove e_remove, Add e_add
            # This is a "swap". Count stays same.
            # But it might unblock other edges or allow future additions.
            # To make progress, we accept swaps with some probability (SA) 
            # or if they reduce "conflict potential".
            
            # Let's accept if it leads to a "better" state in terms of 
            # number of blocked edges? That's expensive.
            # Let's just use SA on the swap.
            # Delta = 0. Accept with prob 0.5? Or always accept to explore?
            # If we always accept swaps, we random walk. 
            # We need a bias.
            
            # Heuristic: Prefer removing edges that are in many "tight" faces?
            # For now, accept swap with probability 0.8 to explore.
            
            if random.random() < 0.8:
                # Apply swap
                selected[e_remove] = False
                for f_idx in edge_to_faces[e_remove]:
                    face_counts[f_idx] -= 1
                
                selected[e_add] = True
                for f_idx in edge_to_faces[e_add]:
                    face_counts[f_idx] += 1
                
            # After a swap, try to greedily add any newly unblocked edges
            # This is the "Repair" step in Memetic Algorithms.
            
            # Find all edges that can now be added
            # To do this efficiently, we can check the faces of the removed edge 
            # and the added edge? No, the state changed.
            # A full scan is O(E*F) which is 448 * ~1000 = 448,000 ops. 
            # In Python, this might be ~0.1s. 5000 steps * 0.1s = 500s. Too slow.
            
            # We need a faster repair.
            # Only edges in faces affected by e_remove or e_add might change status?
            # Actually, removing e_remove decreases face counts for faces containing e_remove.
            # Adding e_add increases face counts for faces containing e_add.
            
            # Edges that were blocked might become unblocked if one of their 
            # blocking faces had a count reduced to 2.
            # Faces affected: those containing e_remove.
            
            potentially_unblocked = set()
            for f_idx in edge_to_faces[e_remove]:
                if face_counts[f_idx] == 2:
                    # The 3rd edge in this face (other than the 2 selected) 
                    # is no longer blocked by THIS face.
                    # But it might be blocked by another face.
                    for oe in faces[f_idx]:
                        if not selected[oe]:
                            potentially_unblocked.add(oe)
            
            # Try to add these potentially unblocked edges
            for oe in potentially_unblocked:
                if selected[oe]:
                    continue
                # Check if oe is now valid
                valid = True
                for f_idx in edge_to_faces[oe]:
                    if face_counts[f_idx] == 3:
                        valid = False
                        break
                if valid:
                    selected[oe] = True
                    for f_idx in edge_to_faces[oe]:
                        face_counts[f_idx] += 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_selected = selected.copy()
                        best_face_counts = face_counts.copy()
            
            T *= cooling
            if T < 0.01:
                T = 0.01
            steps += 1
            
            if steps % 50 == 0 and time.time() > deadline:
                break
                
        return best_selected, best_face_counts, best_count

    # Main Loop: Multiple Restarts
    best_overall_count = 0
    best_overall_selected = np.zeros(num_edges, dtype=bool)
    
    seed = 0
    while time.time() < deadline:
        if time.time() > deadline:
            break
            
        # Greedy Start
        selected, face_counts = greedy_construct(seed)
        seed += 1
        current_count = np.sum(selected)
        
        if current_count > best_overall_count:
            best_overall_count = current_count
            best_overall_selected = selected.copy()
            
        # Guided Local Search
        selected, face_counts, current_count = guided_local_search(selected, face_counts, seed)
        seed += 1
        
        if current_count > best_overall_count:
            best_overall_count = current_count
            best_overall_selected = selected.copy()
            
        # If we have enough time, do another pass or restart
        # The GLS loop above has its own step limit.
        
    # Convert best_overall_selected to list of edges
    result = []
    for i in range(num_edges):
        if best_overall_selected[i]:
            result.append(list(idx_to_edge[i]))
            
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
