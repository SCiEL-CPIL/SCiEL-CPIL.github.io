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
    
    # 3. Spectral Partitioning - compute multiple eigenvectors
    data = np.ones(len(rows), dtype=np.int32)
    row_sym = np.concatenate([rows, cols])
    col_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])
    A_sparse = csr_matrix((data_sym, (row_sym, col_sym)), shape=(N, N))
    
    D_sparse = csr_matrix(np.diag(degrees))
    L = D_sparse - A_sparse
    
    eigvecs = []
    try:
        eigvals, eigvecs_all = eigsh(L, k=5, which='SM')
        idx = np.argsort(eigvals)
        # Skip the trivial eigenvector (index 0), use indices 1-4
        for i in idx[1:5]:
            eigvecs.append(eigvecs_all[:, i])
    except Exception:
        eigvecs = [np.random.randn(N) for _ in range(4)]
    
    def add_to_set(S, in_S, candidates):
        """
        Try to add vertices from `candidates` to the independent set S.
        Returns the number of vertices added.
        """
        added_count = 0
        # Sort candidates by degree ascending (less constrained first)
        sorted_cands = sorted(candidates, key=lambda x: degrees[x])
        for v in sorted_cands:
            if in_S[v]:
                continue
            # Check if v conflicts with any vertex in S
            can_add = True
            for u in adj[v]:
                if in_S[u]:
                    can_add = False
                    break
            if can_add:
                in_S[v] = True
                S.add(v)
                added_count += 1
        return added_count

    def kempe_chain_local_search(S, in_S, max_iter=100):
        """
        Enhanced Kempe-chain based local search.
        """
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            
            # 1. Try to add any vertex that doesn't conflict (Maximal IS check)
            for v in range(N):
                if in_S[v]:
                    continue
                can_add = True
                for u in adj[v]:
                    if in_S[u]:
                        can_add = False
                        break
                if can_add:
                    in_S[v] = True
                    S.add(v)
                    improved = True
            
            if improved:
                continue
                
            # 2. Kempe Chain Swaps
            # Find components in the subgraph induced by V \ S
            visited = [False] * N
            
            best_gain = 0
            best_R = None
            best_I = None
            
            for start_v in range(N):
                if in_S[start_v] or visited[start_v]:
                    continue
                
                # BFS to find component
                comp = []
                queue = [start_v]
                visited[start_v] = True
                
                while queue:
                    u = queue.pop(0)
                    comp.append(u)
                    for v in adj[u]:
                        if not in_S[v] and not visited[v]:
                            visited[v] = True
                            queue.append(v)
                
                # Calculate R: neighbors of comp in S
                R_set = set()
                for v in comp:
                    for u in adj[v]:
                        if in_S[u]:
                            R_set.add(u)
                
                # Calculate I: Max Independent Set within comp
                # Use greedy on comp
                sorted_comp = sorted(comp, key=lambda x: degrees[x])
                I_set = set()
                in_I = [False] * N
                for v in sorted_comp:
                    conflict = False
                    for u in adj[v]:
                        if in_I[u]:
                            conflict = True
                            break
                    if not conflict:
                        I_set.add(v)
                        in_I[v] = True
                
                gain = len(I_set) - len(R_set)
                if gain > best_gain:
                    best_gain = gain
                    best_R = R_set
                    best_I = I_set
            
            if best_gain > 0 and best_R is not None:
                # Perform swap
                for v in best_R:
                    in_S[v] = False
                    S.discard(v)
                for v in best_I:
                    in_S[v] = True
                    S.add(v)
                improved = True
                
            # 3. Single Vertex Swaps (if Kempe didn't find a gain)
            if not improved:
                # Try removing one vertex from S and adding one from V\S
                for _ in range(20):
                    if not S: break
                    v_remove = random.choice(list(S))
                    # Find neighbors of v_remove in V\S
                    candidates = [u for u in adj[v_remove] if not in_S[u]]
                    if not candidates:
                        continue
                    # Try to add a maximal independent set from candidates
                    sorted_cands = sorted(candidates, key=lambda x: degrees[x])
                    I_temp = set()
                    in_temp = [False] * N
                    for u in sorted_cands:
                        conflict = False
                        for w in adj[u]:
                            if in_temp[w]:
                                conflict = True
                                break
                        if not conflict:
                            I_temp.add(u)
                            in_temp[u] = True
                    
                    # Check if I_temp is independent of S \ {v_remove}
                    valid_I = set()
                    for u in I_temp:
                        ok = True
                        for w in adj[u]:
                            if in_S[w] and w != v_remove:
                                ok = False
                                break
                        if ok:
                            valid_I.add(u)
                            
                    if len(valid_I) > 1: # Gain > 0 means len(valid_I) > 1 (since we remove 1)
                        # Perform swap
                        in_S[v_remove] = False
                        S.discard(v_remove)
                        for u in valid_I:
                            in_S[u] = True
                            S.add(u)
                        improved = True
                        break
                        
        return S

    def run_local_search_from_seed(seed_vertices):
        """
        Initialize S with a greedy IS from seed_vertices, then run local search.
        """
        S = set()
        in_S = [False] * N
        add_to_set(S, in_S, seed_vertices)
        
        # Run Kempe chain local search
        S = kempe_chain_local_search(S, in_S)
        
        # Final repair: add any remaining
        for v in range(N):
            if in_S[v]:
                continue
            can_add = True
            for u in adj[v]:
                if in_S[u]:
                    can_add = False
                    break
            if can_add:
                in_S[v] = True
                S.add(v)
                
        return S, in_S

    # 4. Generate multiple partitions and find the best
    best_set = set()
    best_size = 0
    
    all_seeds = []
    
    for evec in eigvecs:
        # 2-way partition
        P1 = [i for i in range(N) if evec[i] > 0]
        P2 = [i for i in range(N) if evec[i] <= 0]
        all_seeds.append(P1)
        all_seeds.append(P2)
        
        # Random perturbation
        for _ in range(2):
            noise = np.random.randn(N) * 0.1
            perturbed = evec + noise
            P1 = [i for i in range(N) if perturbed[i] > 0]
            P2 = [i for i in range(N) if perturbed[i] <= 0]
            all_seeds.append(P1)
            all_seeds.append(P2)

    # Strategy 2: Random subsets
    for _ in range(10):
        random_seed = random.sample(range(N), N // 4)
        all_seeds.append(random_seed)

    # Strategy 3: Degree-based
    low_degree = sorted(range(N), key=lambda x: degrees[x])[:N//4]
    all_seeds.append(low_degree)
    high_degree = sorted(range(N), key=lambda x: degrees[x])[-N//4:]
    all_seeds.append(high_degree)

    # Process seeds
    for seed in all_seeds:
        S, in_S = run_local_search_from_seed(seed)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    final_vectors = []
    for i in range(N):
        if i in best_set:
            final_vectors.append(vectors[i])
            
    return np.array(final_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
