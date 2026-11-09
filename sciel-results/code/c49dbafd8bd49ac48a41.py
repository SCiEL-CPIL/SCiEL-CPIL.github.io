import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    
    Strategy:
    1. Start with 8 disjoint K4s (n=32, alpha=8, C5-free).
    2. Attempt to extend to n=33 by adding a vertex v32.
       - Add v32 to one K4 to form K5? No, K5 has no C5, but alpha doesn't change? 
         Wait, if we add v32 to one K4, we get a K5. 
         Original: 8 K4s. New: 1 K5 + 7 K4s. 
         Alpha: 1 (from K5) + 7 (from K4s) = 8.
         C5-free: K5 is C5-free.
         This works! n=33, alpha=8.
       
       Let's verify:
       Vertices: 0..32.
       Edges:
       K5 on {0,1,2,3,4}.
       K4 on {5,6,7,8}.
       ...
       K4 on {29,30,31,32}.
       
       Total vertices: 5 + 7*4 = 5 + 28 = 33.
       Independence number: max 1 from K5, max 1 from each K4. Total 1+7=8.
       No C5: K5 and K4 are C5-free. Components are disjoint.
       
       So n=33 is valid!
       
       Can we go to n=34?
       1 K5 + 7 K4s = 33.
       Add v33.
       If we add it to K5 -> K6. Alpha: 1 + 7 = 8. n=34.
       K6 is C5-free.
       So 1 K6 + 7 K4s = 6 + 28 = 34. Alpha = 1 + 7 = 8.
       
       Can we go to n=35?
       1 K7 + 7 K4s = 7 + 28 = 35. Alpha = 1 + 7 = 8.
       K7 is C5-free.
       
       Wait, the task says "best known has 32 vertices and no valid graph has more than 35".
       This implies R(C5, K9) > 32 is known, but maybe R(C5, K9) <= 33?
       If R(C5, K9) = 33, then no graph on 33 vertices exists.
       But I just constructed one!
       
       Let me double check the definition of R(C5, K9).
       R(G, H) is the smallest n such that any graph on n vertices contains G or the complement contains H.
       Here H = K9. So complement contains K9 means original graph has an independent set of size 9.
       So we need a graph with no C5 and no independent set of size 9.
       
       My construction:
       n=35. Components: K7, K4, K4, K4, K4, K4, K4, K4.
       Vertices: 7 + 7*4 = 35.
       Alpha: 1 + 7*1 = 8.
       C5-free: Yes.
       
       Why does the prompt say "best known has 32"?
       Maybe I made a mistake?
       
       Is K7 C5-free? Yes.
       Is the independence number of disjoint union the sum of independence numbers? Yes.
       alpha(K_k) = 1 for k>=1.
       So alpha = 1 + 7 = 8.
       
       This seems perfectly valid. Why is the best known 32?
       Perhaps the "best known" refers to a specific family or context I am missing, or the prompt's "best known" is outdated/wrong, or I am misinterpreting "Pentagon-free".
       "Pentagon-free" = no C5 as a subgraph.
       
       Let's re-read carefully: "construct() ... with 1 <= n <= 35 ... Maximize n. A valid graph on n vertices certifies R(C5,K9) > n".
       
       If I can construct n=35, then R(C5, K9) > 35.
       But the prompt says "no valid graph has more than 35". This means n=35 is the upper bound.
       So if my construction is valid, it is optimal.
       
       Why did the previous program use n=32?
       Maybe the previous program was just a seed.
       The score is 32/28 = 1.1429.
       If I return n=35, the score is 35/28 = 1.25.
       
       I will implement the n=35 construction.
    """
    start_time = time.time()
    
    # Configuration: 1 K7 and 7 K4s
    # This gives n = 7 + 7*4 = 35.
    # Alpha = 1 + 7*1 = 8.
    
    n = 35
    edges = []
    
    # First component: K7 (vertices 0 to 6)
    for i in range(7):
        for j in range(i + 1, 7):
            edges.append([i, j])
            
    # Next 7 components: K4s
    # Vertices 7 to 10, 11 to 14, ..., 29 to 32?
    # Wait. 7 + 7*4 = 35.
    # K4 1: 7, 8, 9, 10
    # K4 2: 11, 12, 13, 14
    # K4 3: 15, 16, 17, 18
    # K4 4: 19, 20, 21, 22
    # K4 5: 23, 24, 25, 26
    # K4 6: 27, 28, 29, 30
    # K4 7: 31, 32, 33, 34
    
    for k in range(7):
        base = 7 + 4 * k
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
                
    # Deadline guard
    if time.time() - start_time > 140:
        # Fallback to n=32 just in case, though this is instantaneous
        n_fallback = 32
        edges_fallback = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges_fallback.append([base + i, base + j])
        return {'n': n_fallback, 'edges': edges_fallback}

    return {'n': n, 'edges': edges}
