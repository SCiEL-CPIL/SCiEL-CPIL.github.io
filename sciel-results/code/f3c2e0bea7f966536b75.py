"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    n = 40
    target_edges = 401
    total_possible = n * (n - 1) // 2  # 780

    def compute_max_book(A):
        """Compute the maximum book size over all edges."""
        A2 = A @ A
        mx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    b = int(A2[i, j]) - 1
                    if b > mx:
                        mx = b
        return mx

    def compute_book_vector(A):
        """Return array of book sizes for all possible edges (upper triangle)."""
        A2 = A @ A
        books = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                books[i, j] = int(A2[i, j]) - 1
        return books

    def get_edge_book(A, u, v):
        """Get book size of edge (u,v) assuming it exists."""
        return int((A @ A)[u, v]) - 1

    def random_initial_graph():
        """Generate a random graph with exactly target_edges edges."""
        A = np.zeros((n, n), dtype=np.int32)
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def structured_initial_graph():
        """Generate a structured graph close to K20,20 with balanced modifications."""
        A = np.zeros((n, n), dtype=np.int32)
        
        # Start with K20,20
        for i in range(20):
            for j in range(20, 40):
                A[i, j] = 1
                A[j, i] = 1
        
        # Remove some random cross edges and add intra edges
        # to create a more balanced structure
        cross_edges = [(i, j) for i in range(20) for j in range(20, 40)]
        random.shuffle(cross_edges)
        
        # Remove ~10 cross edges
        for _ in range(10):
            if cross_edges:
                u, v = cross_edges.pop()
                A[u, v] = 0
                A[v, u] = 0
        
        # Add 10 intra edges (5 in each partition)
        intra_added = 0
        attempts = 0
        while intra_added < 10 and attempts < 100:
            attempts += 1
            if random.random() < 0.5:
                a = random.randint(0, 19)
                b = random.randint(0, 19)
            else:
                a = random.randint(20, 39)
                b = random.randint(20, 39)
            if a != b and A[a, b] == 0:
                A[a, b] = 1
                A[b, a] = 1
                intra_added += 1
        
        # Now we have 400 - 10 + 10 = 400 edges. Need 1 more.
        # Find a non-edge that minimizes max book increase
        A2 = A @ A
        current_max = compute_max_book(A)
        
        best_add = None
        best_score = 999
        all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) if A[i, j] == 0]
        random.shuffle(all_nonedges)
        
        for a, b in all_nonedges[:100]:
            A[a, b] = 1
            A[b, a] = 1
            A2_t = A @ A
            new_max = 0
            for v in range(n):
                if v != a and A[a, v] == 1:
                    bk = int(A2_t[a, v]) - 1
                    if bk > new_max:
                        new_max = bk
                if v != b and A[b, v] == 1:
                    bk = int(A2_t[b, v]) - 1
                    if bk > new_max:
                        new_max = bk
            A[a, b] = 0
            A[b, a] = 0
            
            if new_max < best_score:
                best_score = new_max
                best_add = (a, b)
        
        if best_add:
            a, b = best_add
            A[a, b] = 1
            A[b, a] = 1
        
        return A

    def simulated_annealing(A, max_iters, start_temp, end_temp, start_time, deadline):
        """Run simulated annealing to minimize max book."""
        current_A = A.copy()
        current_max = compute_max_book(current_A)
        
        best_A = current_A.copy()
        best_max = current_max
        
        # Precompute edge lists
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) if current_A[i, j] == 0]
        
        for it in range(max_iters):
            if time.time() - start_time > deadline:
                break
            
            # Temperature schedule (geometric cooling)
            temp = start_temp * (end_temp / start_temp) ** (it / max_iters)
            
            # Choose a random edge to remove
            edge_to_remove = random.choice(all_edges)
            if current_A[edge_to_remove[0], edge_to_remove[1]] == 0:
                continue
            u1, v1 = edge_to_remove
            
            # Choose a random non-edge to add
            # Weight towards edges that might reduce max book
            if not all_nonedges:
                break
            
            # Sample a few candidates and pick the best
            candidates = random.sample(all_nonedges, min(len(all_nonedges), 20))
            best_candidate = None
            best_candidate_score = 999
            
            for u2, v2 in candidates:
                # Apply swap
                current_A[u1, v1] = 0
                current_A[v1, u1] = 0
                current_A[u2, v2] = 1
                current_A[v2, u2] = 1
                
                # Compute new max book (quick check)
                A2_t = current_A @ current_A
                new_max = 0
                # Check edges incident to affected vertices
                affected = {u1, v1, u2, v2}
                for v in affected:
                    for w in range(n):
                        if w != v and current_A[v, w] == 1:
                            bk = int(A2_t[v, w]) - 1
                            if bk > new_max:
                                new_max = bk
                
                # Revert
                current_A[u1, v1] = 1
                current_A[v1, u1] = 1
                current_A[u2, v2] = 0
                current_A[v2, u2] = 0
                
                if new_max < best_candidate_score:
                    best_candidate_score = new_max
                    best_candidate = (u2, v2)
            
            if best_candidate is None:
                continue
            
            u2, v2 = best_candidate
            
            # Apply the swap
            current_A[u1, v1] = 0
            current_A[v1, u1] = 0
            current_A[u2, v2] = 1
            current_A[v2, u2] = 1
            
            # Update edge lists
            all_edges.remove((u1, v1))
            all_edges.append((u2, v2))
            all_nonedges.remove((u2, v2))
            all_nonedges.append((u1, v1))
            
            # Compute new max book
            new_max = compute_max_book(current_A)
            
            # Accept if better, or with probability exp(-delta/temp)
            delta = new_max - current_max
            if delta <= 0:
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_A = current_A.copy()
            else:
                accept_prob = np.exp(-delta / temp)
                if random.random() < accept_prob:
                    current_max = new_max
        
        return best_A, best_max

    random.seed(42)
    
    # Try multiple initializations
    best_A = None
    best_max = 999
    
    # Strategy 1: Structured initial + annealing
    for trial in range(15):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        if trial % 2 == 0:
            A_init = structured_initial_graph()
        else:
            A_init = random_initial_graph()
        
        A_result, max_result = simulated_annealing(
            A_init, 
            max_iters=2000,
            start_temp=10.0,
            end_temp=0.1,
            start_time=start_time,
            deadline=DEADLINE - 5
        )
        
        if max_result < best_max:
            best_max = max_result
            best_A = A_result.copy()
        
        if best_max <= 7:
            break
    
    # If still not good, try more aggressive annealing from best
    if best_A is not None and best_max > 7:
        A_result, max_result = simulated_annealing(
            best_A,
            max_iters=5000,
            start_temp=5.0,
            end_temp=0.01,
            start_time=start_time,
            deadline=DEADLINE - 5
        )
        if max_result < best_max:
            best_max = max_result
            best_A = A_result.copy()
    
    # Fallback: random graph
    if best_A is None:
        best_A = random_initial_graph()
        best_max = compute_max_book(best_A)
    
    final_A = best_A
    
    # Extract edges
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_A[i, j] == 1:
                edge_list.append([i, j])
    
    # Ensure exactly 401 edges
    if len(edge_list) != target_edges:
        if len(edge_list) < target_edges:
            all_nonedges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                           if final_A[i, j] == 0]
            random.shuffle(all_nonedges)
            for u, v in all_nonedges:
                if len(edge_list) >= target_edges:
                    break
                final_A[u, v] = 1
                final_A[v, u] = 1
                edge_list.append([u, v])
        else:
            edge_list = edge_list[:target_edges]
            final_A = np.zeros((n, n), dtype=np.int32)
            for u, v in edge_list:
                final_A[u, v] = 1
                final_A[v, u] = 1
    
    assert len(edge_list) == target_edges, f"Edge count {len(edge_list)} != {target_edges}"
    
    return edge_list
# EVOLVE-BLOCK-END
