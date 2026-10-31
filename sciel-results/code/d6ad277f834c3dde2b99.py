import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    # 1. Generate all norm-2 integer vectors (squared norm 4)
    vectors = []
    
    # Type A: one +/- 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = 2 * s
            vectors.append(v)
            
    # Type B: four +/- 1
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, val in zip(combo, signs):
                v[pos] = val
            vectors.append(v)
            
    N = len(vectors)
    
    # 2. Build conflict graph
    vec_arr = np.array(vectors, dtype=np.int8)
    dots = vec_arr @ vec_arr.T
    
    # Conflict if dot > 2
    rows, cols = np.where(dots > 2)
    valid = rows < cols
    rows = rows[valid]
    cols = cols[valid]
    
    adj = [[] for _ in range(N)]
    for r, c in zip(rows, cols):
        adj[r].append(c)
        adj[c].append(r)
        
    degrees = np.array([len(a) for a in adj], dtype=np.int32)
    
    # 3. Spectral Partitioning
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    try:
        eigvals, eigvecs = eigsh(L, k=2, which='SM')
        idx = np.argsort(eigvals)
        fiedler_vec = eigvecs[:, idx[1]]
    except Exception:
        fiedler_vec = np.random.randn(N)
        
    def greedy_independent_set(vertices, adj, degrees):
        """Greedy MIS on a subset of vertices."""
        s = set()
        # Sort by degree descending to pick high-degree nodes first? 
        # Actually, for MIS, picking high-degree nodes first often leads to smaller sets.
        # Picking low-degree nodes first is often better for maximizing size.
        # However, standard greedy for Max Independent Set usually sorts by degree ascending (pick low degree) 
        # or uses a more complex heuristic. Let's try sorting by degree ascending.
        sorted_verts = sorted(vertices, key=lambda x: degrees[x])
        for v in sorted_verts:
            conflict = False
            for u in adj[v]:
                if u in s:
                    conflict = True
                    break
            if not conflict:
                s.add(v)
        return s

    def kempe_local_search(S, adj, degrees, max_iter=200):
        """
        Local search using Kempe chains.
        S: set of vertices in the independent set.
        """
        S = set(S)
        in_S = [False] * N
        for v in S:
            in_S[v] = True
            
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # Find candidates: vertices not in S
            # We prioritize vertices with many neighbors outside S? 
            # Or just iterate through all?
            # To speed up, we can shuffle the order.
            candidates = [v for v in range(N) if not in_S[v]]
            random.shuffle(candidates)
            
            for start_v in candidates:
                if in_S[start_v]:
                    continue
                
                # BFS to find connected component of start_v in V \ S
                comp = []
                queue = [start_v]
                visited_comp = {start_v}
                
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and v not in visited_comp:
                            visited_comp.add(v)
                            queue.append(v)
                            
                # Find neighbors of comp that are in S
                R = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R.add(u)
                            
                # Compute a good independent set I within comp
                # Use greedy MIS on comp
                I = greedy_independent_set(comp, adj, degrees)
                
                # Check if swap is beneficial
                if len(I) > len(R):
                    # Perform swap
                    for v in R:
                        in_S[v] = False
                        S.remove(v)
                    for v in I:
                        in_S[v] = True
                        S.add(v)
                    improved = True
                    # Early break? No, keep searching for more improvements
                    # But restarting the candidate list might be good? 
                    # Let's just continue to next candidate in current list
                    # Note: comp might have changed, but we are iterating over a static list of candidates
                    # This is a heuristic.
                    
        return S

    def solve():
        # Initial partition based on Fiedler vector
        P1 = [i for i in range(N) if fiedler_vec[i] > 0]
        P2 = [i for i in range(N) if fiedler_vec[i] <= 0]
        
        # Generate initial independent sets for both sides
        # We can mix them or just pick the best one
        S1_init = greedy_independent_set(P1, adj, degrees)
        S2_init = greedy_independent_set(P2, adj, degrees)
        
        # Also try a mixed initialization: Greedy on all vertices
        S3_init = greedy_independent_set(range(N), adj, degrees)
        
        candidates = [S1_init, S2_init, S3_init]
        
        best_set = set()
        best_size = 0
        
        for S_init in candidates:
            # Run Kempe local search
            S_final = kempe_local_search(S_init, adj, degrees, max_iter=100)
            
            if len(S_final) > best_size:
                best_size = len(S_final)
                best_set = S_final
                
        # Additional restarts with random perturbations
        for _ in range(5):
            # Perturb the best set: remove a random subset, add random vertices?
            # Or just restart from a random greedy set
            S_rand_init = greedy_independent_set(range(N), adj, degrees)
            # Add some random noise: remove a few, add a few
            S_perturbed = set(S_rand_init)
            if S_perturbed:
                # Remove 10% randomly
                to_remove = random.sample(list(S_perturbed), min(int(0.1*len(S_perturbed)), 10))
                for v in to_remove:
                    S_perturbed.remove(v)
            S_final = kempe_local_search(S_perturbed, adj, degrees, max_iter=100)
            
            if len(S_final) > best_size:
                best_size = len(S_final)
                best_set = S_final
                
        final_vectors = []
        for i in range(N):
            if i in best_set:
                final_vectors.append(vectors[i])
                
        return np.array(final_vectors, dtype=np.int64)

    return solve()

# EVOLVE-BLOCK-END
