import random
import time
import heapq

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin (150s kill, 5s buffer)

    d = 7
    n = 2 ** d
    
    # 1. Precompute edges and faces
    edges = []
    edge_id_map = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                edge_id = len(edges)
                edges.append((x, y))
                edge_id_map[(x, y)] = edge_id
                edge_id_map[(y, x)] = edge_id

    n_edges = len(edges)
    
    face_edges = []
    for v in range(n):
        for i in range(d):
            for j in range(i+1, d):
                v_i = v ^ (1 << i)
                v_j = v ^ (1 << j)
                v_ij = v ^ (1 << i) ^ (1 << j)
                
                e1 = edge_id_map[(min(v, v_i), max(v, v_i))]
                e2 = edge_id_map[(min(v_i, v_ij), max(v_i, v_ij))]
                e3 = edge_id_map[(min(v, v_j), max(v, v_j))]
                e4 = edge_id_map[(min(v_j, v_ij), max(v_j, v_ij))]
                
                face_edges.append([e1, e2, e3, e4])

    n_faces = len(face_edges)
    
    edge_faces = [[] for _ in range(n_edges)]
    for f_idx, f_edges in enumerate(face_edges):
        for e_idx in f_edges:
            edge_faces[e_idx].append(f_idx)

    # 2. Greedy Initialization via Hitting Set with priority queue
    def greedy_hitting_set(rng, time_budget):
        covered = [False] * n_faces
        R = set()
        cur_benefit = [len(edge_faces[e]) for e in range(n_edges)]
        
        heap = [(-cur_benefit[e], e) for e in range(n_edges)]
        heapq.heapify(heap)
        in_heap = [True] * n_edges
        
        n_covered = 0
        end_time = time.time() + time_budget
        
        while n_covered < n_faces and time.time() < end_time:
            best_e = -1
            best_ben = 0
            while heap:
                neg_b, e = heapq.heappop(heap)
                if not in_heap[e] or e in R:
                    in_heap[e] = False
                    continue
                cur_ben = -neg_b
                if cur_ben <= 0:
                    in_heap[e] = False
                    continue
                best_e = e
                best_ben = cur_ben
                break
            
            if best_e == -1:
                break
                
            R.add(best_e)
            in_heap[best_e] = False
            
            for f in edge_faces[best_e]:
                if not covered[f]:
                    covered[f] = True
                    n_covered += 1
                    for e in face_edges[f]:
                        if e != best_e and e not in R and in_heap[e]:
                            cur_benefit[e] -= 1
                            if cur_benefit[e] > 0:
                                heapq.heappush(heap, (-cur_benefit[e], e))
        
        return R

    def get_kept_set(R):
        return set(e for e in range(n_edges) if e not in R)

    # 3. Local Search with SA
    def local_search_kept(K_init, rng, time_budget):
        K = set(K_init)
        face_counts = [0] * n_faces
        for e in K:
            for f in edge_faces[e]:
                face_counts[f] += 1
        
        for f in range(n_faces):
            if face_counts[f] == 4:
                e_to_remove = face_edges[f][0]
                K.remove(e_to_remove)
                for ff in edge_faces[e_to_remove]:
                    face_counts[ff] -= 1
                break

        best_K = set(K)
        best_size = len(K)
        
        T_start = 1.0
        T_end = 0.05
        cooling = 0.999
        T = T_start
        
        end_time = time.time() + time_budget
        current_size = len(K)
        
        K_list = list(K)
        
        while time.time() < end_time:
            if T < T_end:
                T = T_start
                K_list = list(K)
            
            move_type = rng.random()
            if move_type < 0.6:
                e = rng.randint(0, n_edges - 1)
                if e in K:
                    continue
                
                valid = True
                for f in edge_faces[e]:
                    if face_counts[f] == 3:
                        valid = False
                        break
                
                if valid:
                    K.add(e)
                    for f in edge_faces[e]:
                        face_counts[f] += 1
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_K = set(K)
                        
            elif move_type < 0.8:
                if not K:
                    continue
                e = rng.choice(K_list)
                if e not in K:
                    continue
                
                delta = -1
                if rng.random() < pow(2.718281828, -delta / T):
                    K.remove(e)
                    for f in edge_faces[e]:
                        face_counts[f] -= 1
                    current_size -= 1
                    
            else:
                if not K:
                    continue
                e_out = rng.choice(K_list)
                if e_out not in K:
                    continue
                e_in = rng.randint(0, n_edges - 1)
                if e_in in K:
                    continue
                
                valid = True
                for f in edge_faces[e_in]:
                    e_out_in_face = False
                    for e_check in face_edges[f]:
                        if e_check == e_out:
                            e_out_in_face = True
                            break
                    
                    new_count = face_counts[f] - (1 if e_out_in_face else 0)
                    if new_count == 3:
                        valid = False
                        break
                
                if valid:
                    K.remove(e_out)
                    for f in edge_faces[e_out]:
                        face_counts[f] -= 1
                    
                    K.add(e_in)
                    for f in edge_faces[e_in]:
                        face_counts[f] += 1
                    if rng.random() >= 0.7:
                        K.remove(e_in)
                        for f in edge_faces[e_in]:
                            face_counts[f] -= 1
                        K.add(e_out)
                        for f in edge_faces[e_out]:
                            face_counts[f] += 1

            T *= cooling
            
        return best_K, best_size

    # 4. Final polish: deterministic local search
    def final_polish(K, time_budget):
        end_time = time.time() + time_budget
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for e in range(n_edges):
                if e in K:
                    continue
                valid = True
                for f in edge_faces[e]:
                    count = 0
                    for e2 in face_edges[f]:
                        if e2 in K:
                            count += 1
                    if count == 3:
                        valid = False
                        break
                if valid:
                    K.add(e)
                    improved = True
        return K

    # 5. Multi-Start
    best_K = set()
    best_size = 0
    
    rng_init = random.Random(42)
    init_time = min(8.0, time_limit - 30)
    R_init = greedy_hitting_set(rng_init, init_time)
    K_init = get_kept_set(R_init)
    
    remaining_time = time_limit - 5 - (time.time() - start_time)
    if remaining_time > 10:
        sa_time = remaining_time * 0.7
        K_best, size_best = local_search_kept(K_init, random.Random(123), sa_time)
        
        polish_time = min(5.0, time_limit - 10 - (time.time() - start_time))
        if polish_time > 1:
            K_best = final_polish(K_best, polish_time)
            size_best = len(K_best)
        
        best_K = K_best
        best_size = size_best
        
        remaining2 = time_limit - 15 - (time.time() - start_time)
        if remaining2 > 25:
            rng_init2 = random.Random(999)
            R_init2 = greedy_hitting_set(rng_init2, min(5.0, remaining2 * 0.1))
            K_init2 = get_kept_set(R_init2)
            
            sa_time2 = remaining2 * 0.65
            K_best2, size_best2 = local_search_kept(K_init2, random.Random(777), sa_time2)
            
            polish_time2 = min(3.0, time_limit - 10 - (time.time() - start_time))
            if polish_time2 > 0.5:
                K_best2 = final_polish(K_best2, polish_time2)
                size_best2 = len(K_best2)
            
            if size_best2 > best_size:
                best_K = K_best2
                best_size = size_best2

    result = []
    for e_id in best_K:
        u, v = edges[e_id]
        result.append([u, v])
        
    return result
