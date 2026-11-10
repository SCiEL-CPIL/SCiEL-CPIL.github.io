"""Program for the no-isosceles set task using greedy with hill climbing."""
import random

N = 64

def construct():
    points = [(x, y) for y in range(N) for x in range(N)]
    
    def dist_sq(p1, p2):
        return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

    def can_add(p, S, distances_from):
        # Check if p is apex: any two points in S at same distance from p?
        seen_dists = set()
        for a in S:
            d = dist_sq(p, a)
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
        # Check if any a in S is apex: dist(p,a) == dist(a,b) for some b in S?
        for a in S:
            d = dist_sq(p, a)
            if d in distances_from[a]:
                return False
                
        return True

    def add_point(p, S, distances_from):
        S.append(p)
        distances_from[p] = set()
        for a in S[:-1]:
            d = dist_sq(p, a)
            distances_from[a].add(d)
            distances_from[p].add(d)

    def remove_point(r, S, distances_from):
        idx = S.index(r)
        S.pop(idx)
        for a in S:
            d = dist_sq(r, a)
            distances_from[a].discard(d)
        del distances_from[r]

    def greedy(order):
        S = []
        dist_from = {}
        for p in order:
            if can_add(p, S, dist_from):
                add_point(p, S, dist_from)
        return S, dist_from

    def hill_climb(S, dist_from, max_steps=100):
        best_S = S[:]
        best_dist = {k: v.copy() for k, v in dist_from.items()}
        best_size = len(best_S)
        
        for _ in range(max_steps):
            if not best_S:
                break
            r = random.choice(best_S)
            
            exp_S = best_S[:]
            exp_dist = {k: v.copy() for k, v in best_dist.items()}
            
            remove_point(r, exp_S, exp_dist)
            
            for p in points:
                if p in exp_S:
                    continue
                if can_add(p, exp_S, exp_dist):
                    add_point(p, exp_S, exp_dist)
            
            if len(exp_S) > best_size:
                best_S = exp_S
                best_dist = exp_dist
                best_size = len(exp_S)
                
        return best_S

    best_set = []
    best_size = 0
    
    for seed in range(30):
        order = points[:]
        random.seed(seed)
        random.shuffle(order)
        
        S, dist_from = greedy(order)
        S = hill_climb(S, dist_from, max_steps=50)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
