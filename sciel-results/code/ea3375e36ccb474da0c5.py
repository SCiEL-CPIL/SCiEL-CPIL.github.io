import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_score = 12
    best_edges = None
    
    def has_independent_set_4(H):
        # H is 18x18 bool matrix
        # Check all 4-combinations
        for quad in combinations(range(18), 4):
            a, b, c, d = quad
            if not (H[a][b] or H[a][c] or H[a][d] or H[b][c] or H[b][d] or H[c][d]):
                return True
        return False

    def compute_score_greedy(H):
        # G is complement of H
        # We want max size of S such that G[S] is triangle-free
        # i.e., H[S] has no independent set of size 3
        S = set(range(18))
        changed = True
        while changed:
            changed = False
            # Find independent triple in H[S]
            # Iterate over triples in S
            # S is a set, convert to list for combinations
            s_list = list(S)
            for tri in combinations(s_list, 3):
                a, b, c = tri
                if not (H[a][b] or H[a][c] or H[b][c]):
                    # Remove one vertex
                    S.remove(a)
                    changed = True
                    break
        return len(S)

    # Initial best: disjoint triangles
    # H is complete multipartite with 6 parts of size 3
    H_init = [[False]*18 for _ in range(18)]
    for i in range(18):
        for j in range(18):
            if i != j and (i // 3) != (j // 3):
                H_init[i][j] = True
    # Score of this is 12
    best_edges = []
    for i in range(18):
        for j in range(i+1, 18):
            if not H_init[i][j]:
                best_edges.append([i, j])
                
    while time.time() < deadline:
        # Generate random H
        H = [[False]*18 for _ in range(18)]
        p = 0.5
        for i in range(18):
            for j in range(i+1, 18):
                if random.random() < p:
                    H[i][j] = True
                    H[j][i] = True
        
        if has_independent_set_4(H):
            continue
        
        score = compute_score_greedy(H)
        if score < best_score:
            best_score = score
            # Convert H to edges of G
            edges = []
            for i in range(18):
                for j in range(i+1, 18):
                    if not H[i][j]:
                        edges.append([i, j])
            best_edges = edges
            if best_score <= 9:
                break

    return best_edges
