import time

N = 16

# Precomputed optimal colors g(n) for n=1..16
# g(1)=0, g(2)=1, g(3)=3, g(4)=4, g(5)=5, g(6)=7, g(7)=8, g(8)=9, 
# g(9)=11, g(10)=12, g(11)=13, g(12)=15, g(13)=16, g(14)=17, g(15)=19, g(16)=20
G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]

# Precomputed best split k for n=4..16
# n=4: 3,1 (or 1,3) -> G[3]+G[1]=3
# n=5: 3,2 (or 2,3) -> G[3]+G[2]=4 (Wait, G[3]=3, G[2]=1 -> 4. 1+4=5. Correct)
# n=6: 3,3 -> G[3]+G[3]=6. 1+6=7. Correct.
# n=7: 3,4 -> G[3]+G[4]=7. 1+7=8. Correct.
# n=8: 4,4 -> G[4]+G[4]=8. 1+8=9. Correct.
# n=9: 3,6 -> G[3]+G[6]=10. 1+10=11. Correct.
# n=10: 4,6 -> G[4]+G[6]=11. 1+11=12. Correct.
# n=11: 5,6 -> G[5]+G[6]=12. 1+12=13. Correct.
# n=12: 6,6 -> G[6]+G[6]=14. 1+14=15. Correct.
# n=13: 6,7 -> G[6]+G[7]=15. 1+15=16. Correct.
# n=14: 7,7 -> G[7]+G[7]=16. 1+16=17. Correct.
# n=15: 6,9 -> G[6]+G[9]=18. 1+18=19. Correct.
# n=16: 7,9 -> G[7]+G[9]=19. 1+19=20. Correct.

SPLIT = {
    4: 3,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 6,
    14: 7,
    15: 6,
    16: 7
}

def construct():
    # colors[i][j] for i < j
    colors = [[-1] * N for _ in range(N)]
    
    def solve(vertices, current_color):
        n = len(vertices)
        if n <= 1:
            return current_color
        if n == 2:
            u, v = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            return current_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            colors[v][w] = current_color + 1
            colors[w][v] = current_color + 1
            colors[u][w] = current_color + 2
            colors[w][u] = current_color + 2
            return current_color + 3
        
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        
        cross_color = current_color
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        
        next_color = current_color + 1
        next_color = solve(v1, next_color)
        next_color = solve(v2, next_color)
        return next_color
    
    vertices = list(range(N))
    solve(vertices, 0)
    
    # Flatten to list of 120 integers
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(colors[i][j])
            
    return result

# EVOLVE-BLOCK-END
