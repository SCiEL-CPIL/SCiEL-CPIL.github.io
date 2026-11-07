"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    trees = instance_trees(20, 611)
    n = 20
    
    tree_edge_sets = []
    for i in range(n):
        es = set()
        for (u,v) in trees[i]:
            es.add((min(u,v), max(u,v)))
        tree_edge_sets.append(es)
    
    import sys
    sys.setrecursionlimit(10000)
    
    def try_pack(t20_hosts):
        used_edges = set()
        result = [None] * n
        
        # T20
        result[19] = t20_hosts
        for (u,v) in tree_edge_sets[19]:
            hu, hv = t20_hosts[u], t20_hosts[v]
            used_edges.add((min(hu,hv), max(hu,hv)))
        
        # T1
        result[0] = [t20_hosts[0]]  # or any, use first host vertex
        
        def backtrack(tree_idx):
            if tree_idx == 20:
                return len(used_edges) == 190
            
            k = tree_idx + 1
            if k == 1:
                result[tree_idx] = [t20_hosts[0]]
                return backtrack(tree_idx + 1)
            
            tes = tree_edge_sets[tree_idx]
            rng = Random(tree_idx * 7919 + 13)
            
            for _ in range(100000):
                hosts = rng.sample(range(n), k)
                valid = True
                new_edges = []
                for (u,v) in tes:
                    hu, hv = hosts[u], hosts[v]
                    he = (min(hu,hv), max(hu,hv))
                    if he in used_edges:
                        valid = False
                        break
                    new_edges.append(he)
                
                if valid:
                    for he in new_edges:
                        used_edges.add(he)
                    result[tree_idx] = hosts
                    
                    if backtrack(tree_idx + 1):
                        return True
                    
                    for he in new_edges:
                        used_edges.remove(he)
                    result[tree_idx] = None
            
            return False
        
        if backtrack(1):
            final = []
            for i in range(n):
                if result[i] is None:
                    final.append(None)
                else:
                    final.append([int(x) for x in result[i]])
            return final
        
        return None
    
    # Try identity first
    result = try_pack(list(range(20)))
    if result is not None:
        return result
    
    # Try random permutations for T20
    rng = Random(12345)
    for _ in range(100):
        perm = list(range(20))
        rng.shuffle(perm)
        result = try_pack(perm)
        if result is not None:
            return result
    
    return None

# EVOLVE-BLOCK-END
