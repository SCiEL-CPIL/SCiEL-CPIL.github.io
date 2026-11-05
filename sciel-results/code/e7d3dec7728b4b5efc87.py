import random
import time
import heapq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    d = 7
    n = 2 ** d
    
    # 1. Precompute edges and faces
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)
    
    face_edges = []
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)
    
    # 2. Build conflict graph (edges conflict if they share a face)
    # For spectral initialization, we'll use the line graph of the conflict structure
    conflict_rows = []
    conflict_cols = []
    for f_idx, f_edges in enumerate(face_edges):
        for e1 in f_edges:
            for e2 in f_edges:
                if e1 < e2:
                    conflict_rows.append(e1)
                    conflict_cols.append(e2)
                    conflict_rows.append(e2)
                    conflict_cols.append(e1)
    
    # Build sparse adjacency matrix for conflict graph
    A = csr_matrix((np.ones(len(conflict_rows)), (conflict_rows, conflict_cols)), 
                   shape=(n_edges, n_edges))
    
    # Compute Fiedler vector (2nd smallest eigenvector) for spectral partitioning
    try:
        if time.time() - start_time < time_limit - 30:
            # Get the smallest eigenvalues and eigenvectors
            # We want the Fiedler vector (second smallest eigenvalue)
            eigvals, eigvecs = eigsh(A, k=2, which='SM')
            # Fiedler vector is the eigenvector with second smallest eigenvalue
            # eigsh returns in ascending order when using which='SM'
            fiedler_vec = eigvecs[:, 1]
            
            # Partition edges based on Fiedler vector sign
            # Edges with positive Fiedler values go to one partition, negative to other
            threshold = np.median(fiedler_vec)
            spectral_partition = (fiedler_vec > threshold).astype(int)
            
            # Initial hitting set: edges in the "conflict" partition
            # We want to minimize the hitting set, so start with edges that are "central"
            # Use the magnitude of Fiedler vector as a guide
            fiedler_magnitude = np.abs(fiedler_vec)
            sorted_indices = np.argsort(fiedler_magnitude)[::-1]
            
            # Start with edges having high Fiedler magnitude
            initial_R = set(sorted_indices[:n_edges // 2])
        else:
            initial_R = None
    except Exception:
        initial_R = None

    def greedy_hitting_set(rng, initial_R=None):
        benefit = [len(edge_faces[e]) for e in range(n_edges)]
        is_covered = [False] * n_faces
        R = set()
        
        # If we have a spectral initial partition, use it to guide the greedy choice
        if initial_R is not None:
            # Start with edges from the spectral partition that cover the most faces
            spectral_benefit = [(len(edge_faces[e]), e) for e in initial_R]
            spectral_benefit.sort(key=lambda x: -x[0])
            
            # Add top spectral edges first
            for b, e in spectral_benefit:
                if e in R:
                    continue
                R.add(e)
                for f in edge_faces[e]:
                    if not is_covered[f]:
                        is_covered[f] = True
                        for e_prime in face_edges[f]:
                            if e_prime != e:
                                benefit[e_prime] -= 1
                if time.time() - start_time > time_limit - 10:
                    break
        
        # Continue with standard greedy for remaining uncovered faces
        uncovered_count = sum(1 for f in range(n_faces) if not is_covered[f])
        
        heap = [(-benefit[e], e) for e in range(n_edges) if e not in R]
        heapq.heapify(heap)
        
        while uncovered_count > 0:
            if time.time() - start_time > time_limit - 10:
                break
                
            neg_b, e = heapq.heappop(heap)
            b = -neg_b
            
            if b != benefit[e]:
                continue
            if e in R:
                continue
            if b == 0:
                break
                
            R.add(e)
            
            for f in edge_faces[e]:
                if not is_covered[f]:
                    is_covered[f] = True
                    uncovered_count -= 1
                    
                    for e_prime in face_edges[f]:
                        if e_prime != e:
                            benefit[e_prime] -= 1
                            heapq.heappush(heap, (-benefit[e_prime], e_prime))
                            
        return R

    def minimize_hitting_set(R, rng):
        R = set(R)
        cover_count = [0] * n_faces
        for e in R:
            for f in edge_faces[e]:
                cover_count[f] += 1
        
        improved = True
        while improved:
            improved = False
            r_list = list(R)
            rng.shuffle(r_list)
            for e in r_list:
                if e not in R:
                    continue
                redundant = True
                for f in edge_faces[e]:
                    if cover_count[f] == 1:
                        redundant = False
                        break
                if redundant:
                    R.remove(e)
                    for f in edge_faces[e]:
                        cover_count[f] -= 1
                    improved = True
                    if time.time() - start_time > time_limit - 5:
                        return R
        return R

    # Use spectral initialization if available
    if initial_R is not None:
        best_R = greedy_hitting_set(random.Random(123), initial_R)
    else:
        best_R = greedy_hitting_set(random.Random(123))
    best_size = len(best_R)
    
    rng = random.Random(42)
    
    # Multi-phase local search with simulated annealing
    temperature = 50.0
    cooling_rate = 0.995
    
    while time.time() - start_time < time_limit - 10:
        current_R = set(best_R)
        
        # Random perturbation: remove some edges and re-add
        num_remove = rng.randint(1, 4)
        to_remove = rng.sample(list(current_R), min(num_remove, len(current_R)))
        for e in to_remove:
            current_R.remove(e)
            
        cover_count = [0] * n_faces
        for e in current_R:
            for f in edge_faces[e]:
                cover_count[f] += 1
                
        uncovered = [f for f in range(n_faces) if cover_count[f] == 0]
        while uncovered:
            f = rng.choice(uncovered)
            e_candidates = face_edges[f]
            e = rng.choice(e_candidates)
            
            if e not in current_R:
                current_R.add(e)
                for ff in edge_faces[e]:
                    cover_count[ff] += 1
                
            uncovered = [ff for ff in range(n_faces) if cover_count[ff] == 0]
            
        current_R = minimize_hitting_set(current_R, rng)
        
        # Simulated annealing acceptance
        if len(current_R) < best_size:
            best_size = len(current_R)
            best_R = current_R
        else:
            # Accept worse solutions with probability based on temperature
            delta = len(current_R) - best_size
            if delta > 0 and rng.random() < np.exp(-delta / temperature):
                best_R = current_R
                best_size = len(current_R)
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 1.0:
            temperature = 1.0

    result = []
    for e_id, (u, v) in enumerate(edges):
        if e_id not in best_R:
            result.append([u, v])
            
    return result
