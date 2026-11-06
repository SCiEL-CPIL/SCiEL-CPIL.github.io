import itertools
import random

def construct():
    n = 18
    random.seed(42) # For reproducibility
    
    # 1. Generate random graph
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            if random.random() < 0.5:
                edges.add((u, v))
                
    # 2. Remove edges to make it K4-free
    # Greedy removal
    changed = True
    while changed:
        changed = False
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if ((u,v) in edges) and ((u,w) in edges) and ((u,x) in edges) and \
               ((v,w) in edges) and ((v,x) in edges) and ((w,x) in edges):
                # Remove a random edge
                e = random.choice([(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)])
                edges.remove(e)
                changed = True
                
    # 3. Local search to minimize max TF size
    def creates_k4(u, v, edges):
        common = set()
        for w in range(n):
            if w != u and w != v:
                if (u,w) in edges and (v,w) in edges:
                    common.add(w)
        common_list = list(common)
        for i in range(len(common_list)):
            for j in range(i+1, len(common_list)):
                w, x = common_list[i], common_list[j]
                if (w,x) in edges:
                    return True
        return False

    def is_triangle_free(combo, edges):
        for a, b, c in itertools.combinations(combo, 3):
            if (a,b) in edges and (b,c) in edges and (a,c) in edges:
                return False
        return True

    def get_max_tf_info(edges):
        for k in range(n, 0, -1):
            for combo in itertools.combinations(range(n), k):
                if is_triangle_free(combo, edges):
                    return k, combo
        return 0, None

    current_max_size, current_combo = get_max_tf_info(edges)
    
    # Try to improve
    for _ in range(200):
        if current_max_size <= 1:
            break
        added = False
        for u, v in itertools.combinations(current_combo, 2):
            if (u, v) not in edges:
                if not creates_k4(u, v, edges):
                    edges.add((u, v))
                    added = True
                    break
        if not added:
            break
        new_max_size, new_combo = get_max_tf_info(edges)
        current_max_size = new_max_size
        current_combo = new_combo

    # Compare with Turan T3(18) score (12)
    # If our result is worse, return Turan
    # But we don't know the score of Turan without calculating?
    # We know it's 12.
    # Let's just return the result. If it's 12 or better, it's good.
    # If it's 13, it's bad.
    # To be safe, we can also generate T3 and pick the best.
    
    # Generate T3
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    t3_edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    t3_edges.add((u, v))
    t3_size, _ = get_max_tf_info(t3_edges)
    
    if current_max_size <= t3_size:
        return [[u, v] for u, v in edges]
    else:
        return [[u, v] for u, v in t3_edges]
