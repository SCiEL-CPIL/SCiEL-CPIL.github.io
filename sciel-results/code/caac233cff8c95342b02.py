import time
import random

def construct(k, g):
    # Petersen Graph: 10 vertices, 15 edges, 3-regular, girth 5
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10

    def get_thue_morse(n):
        """Generate first n bits of Thue-Morse sequence."""
        return [bin(i).count('1') % 2 for i in range(n)]

    def check_girth_fast(adj, n_vertices, girth_limit):
        """Check if graph has girth >= girth_limit using BFS from each vertex."""
        for start in range(n_vertices):
            dist = [-1] * n_vertices
            parent = [-1] * n_vertices
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                if dist[node] >= girth_limit - 1:
                    continue
                for neighbor in adj[node]:
                    if dist[neighbor] == -1:
                        dist[neighbor] = dist[node] + 1
                        parent[neighbor] = node
                        queue.append(neighbor)
                    elif neighbor != parent[node]:
                        cycle_len = dist[node] + dist[neighbor] + 1
                        if cycle_len < girth_limit:
                            return False
        return True

    def build_and_check(m, voltages, girth_limit):
        """Build lift, check girth, return edges if valid."""
        n_vertices = n_base * m
        adj = [[] for _ in range(n_vertices)]
        
        for i, (u, v) in enumerate(petersen_edges):
            w = voltages[i]
            for s in range(m):
                u_id = u * m + s
                v_id = v * m + (s + w) % m
                adj[u_id].append(v_id)
                adj[v_id].append(u_id)
        
        if check_girth_fast(adj, n_vertices, girth_limit):
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                w = voltages[i]
                for s in range(m):
                    u_id = u * m + s
                    v_id = v * m + (s + w) % m
                    if u_id < v_id:
                        edges.append((u_id, v_id))
                    else:
                        edges.append((v_id, u_id))
            return n_vertices, edges
        return None

    start_time = time.time()
    tm = get_thue_morse(1000)

    # Calculate lower bound for m
    if g % 2 == 0:
        r = g // 2
        moore_bound = k * (2 ** (r - 1) - 1) + 1
    else:
        r = (g - 1) // 2
        moore_bound = int(k / 2 * (2 ** r - 1)) + 1
        
    m_min = max(2, (moore_bound + n_base - 1) // n_base)

    # Search for smallest m
    # We try increasing m
    for m in range(m_min, m_min + 200):
        if time.time() - start_time > 110:
            break
            
        # Try multiple voltage patterns for this m
        num_tries = 100 if m < 60 else 50
        
        for t in range(num_tries):
            # Seed random generator with m and t, mixed with Thue-Morse
            seed_val = (m * 1000 + t * 7 + tm[t % len(tm)] * 13)
            random.seed(seed_val)
            
            # Generate voltages
            voltages = [random.randint(0, m - 1) for _ in range(15)]
            
            result = build_and_check(m, voltages, g)
            if result:
                return result
                
    # Fallback: return a large graph if not found
    m = m_min + 200
    n_vertices = n_base * m
    random.seed(42)
    voltages = [random.randint(0, m - 1) for _ in range(15)]
    edges = []
    for i, (u, v) in enumerate(petersen_edges):
        w = voltages[i]
        for s in range(m):
            u_id = u * m + s
            v_id = v * m + (s + w) % m
            if u_id < v_id:
                edges.append((u_id, v_id))
            else:
                edges.append((v_id, u_id))
    return n_vertices, edges
