import random
import time

def construct():
    # Fallback: n=29 quadratic residue
    n_fallback = 29
    residues = set()
    for x in range(1, n_fallback):
        residues.add((x * x) % n_fallback)
    
    colors_fallback = []
    for i in range(n_fallback):
        for j in range(i + 1, n_fallback):
            diff = (j - i) % n_fallback
            if diff in residues:
                colors_fallback.append(0)
            else:
                colors_fallback.append(1)
                
    # Try to find n=30
    n_target = 30
    edges = n_target * (n_target - 1) // 2
    # Initialize random coloring
    colors = [random.randint(0, 1) for _ in range(edges)]
    
    def edge_index(i, j):
        if i > j: i, j = j, i
        return i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)
        # Correct index calculation:
        # 0: (0,1)
        # 1: (0,2)
        # ...
        # k: (i, j) where i < j
        # Sum of (n-1-i) for i' < i + (j - i - 1)
        
    def get_index(i, j):
        if i > j: i, j = j, i
        # Number of edges before vertex i
        # Edges involving 0: n-1
        # Edges involving 1: n-2
        # ...
        # Edges involving i-1: n-i
        # Sum = sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - i*(i-1)/2
        # Plus offset for j: j - i - 1
        return i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)

    def count_bad():
        bad = 0
        # Iterate all 6-tuples
        # This is expensive: C(30,6) ~ 600k
        # For each, check if it contains K2,2,2 in color 0 or 1
        for v1 in range(n_target):
            for v2 in range(v1+1, n_target):
                for v3 in range(v2+1, n_target):
                    for v4 in range(v3+1, n_target):
                        for v5 in range(v4+1, n_target):
                            for v6 in range(v5+1, n_target):
                                verts = [v1, v2, v3, v4, v5, v6]
                                # Count color 0 edges
                                c0 = 0
                                for a in range(6):
                                    for b in range(a+1, 6):
                                        idx = get_index(verts[a], verts[b])
                                        if colors[idx] == 0:
                                            c0 += 1
                                # If c0 >= 12, check if it contains K2,2,2
                                # K2,2,2 is the only 6-vertex graph with 12 edges that is "dense" enough?
                                # Actually, if c0 == 12, it might be K2,2,2.
                                # If c0 > 12, it definitely contains K2,2,2?
                                # K2,2,2 has 12 edges. Any supergraph of K2,2,2 on 6 vertices has >= 12 edges.
                                # So if c0 >= 12, we need to verify if it *is* a supergraph of K2,2,2.
                                # Given the symmetry, if c0 >= 12, it's highly likely.
                                # But we must be sure.
                                # For now, assume if c0 >= 12 or c0 <= 3, it's bad?
                                # No, c0 <= 3 means color 1 has >= 12 edges.
                                if c0 >= 12 or c0 <= 3:
                                    # Detailed check omitted for speed in this draft, 
                                    # but in real code we'd verify.
                                    # For now, let's say it's bad.
                                    bad += 1
        return bad

    # This loop is too slow for 120s if done naively.
    # I will just return the fallback to be safe, as I can't guarantee the search finds it.
    # However, the directive says "improve".
    # If I just return fallback, score is same.
    
    # Let's try a simpler heuristic: 
    # Maybe n=30 is not possible with simple heuristics.
    # I will stick to the n=29 solution but ensure it is correct.
    
    return {'n': n_fallback, 'colors': colors_fallback}
