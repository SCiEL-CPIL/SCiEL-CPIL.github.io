import random
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill
    d = 7
    n = 1 << d  # 128 vertices
    best_count = 0
    best_chosen = None

    # All edges
    all_edges = []
    for u in range(n):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                all_edges.append((u, v, i))

    num_edges = len(all_edges)  # 448

    # Map edge tuple to index
    edge_tuple_to_idx = {}
    for idx, (u, v, i) in enumerate(all_edges):
        edge_tuple_to_idx[(u, v)] = idx

    # Precompute face edge indices
    faces = []
    for i in range(d):
        for j in range(i + 1, d):
            for mask in range(1 << d):
                if (mask & (1 << i)) or (mask & (1 << j)):
                    continue
                v1 = mask
                v2 = mask ^ (1 << i)
                v3 = mask ^ (1 << j)
                v4 = mask ^ (1 << i) ^ (1 << j)
                e1 = (min(v1, v2), max(v1, v2))
                e2 = (min(v2, v4), max(v2, v4))
                e3 = (min(v4, v3), max(v4, v3))
                e4 = (min(v3, v1), max(v3, v1))
                idx1 = edge_tuple_to_idx[e1]
                idx2 = edge_tuple_to_idx[e2]
                idx3 = edge_tuple_to_idx[e3]
                idx4 = edge_tuple_to_idx[e4]
                faces.append((idx1, idx2, idx3, idx4))

    num_faces = len(faces)  # 672

    # Map edge index to list of face indices it belongs to
    edge_to_faces = [[] for _ in range(num_edges)]
    for f_idx, face in enumerate(faces):
        for e_idx in face:
            edge_to_faces[e_idx].append(f_idx)

    # Numpy arrays for fast vectorized updates
    # edge_to_faces as numpy 2D array with -1 padding
    max_faces_per_edge = max(len(x) for x in edge_to_faces)
    edge_to_faces_np = np.full((num_edges, max_faces_per_edge), -1, dtype=np.int32)
    edge_to_faces_len = np.zeros(num_edges, dtype=np.int32)
    for e_idx, flist in enumerate(edge_to_faces):
        edge_to_faces_np[e_idx, :len(flist)] = flist
        edge_to_faces_len[e_idx] = len(flist)

    # faces as numpy 2D array (num_faces x 4)
    faces_np = np.array(faces, dtype=np.int32)

    # Precompute for each edge, the list of other edges sharing a face (sorted for fast lookup)
    edge_shared_edges = [set() for _ in range(num_edges)]
    for f_idx in range(num_faces):
        for e1 in faces_np[f_idx]:
            for e2 in faces_np[f_idx]:
                if e1 != e2:
                    edge_shared_edges[e1].add(e2)
    
    # Convert to sorted lists for faster membership testing
    edge_shared_edges_sorted = [sorted(s) for s in edge_shared_edges]
    
    # Build a 2D boolean adjacency matrix for O(1) compatibility checks
    # adj[e1][e2] = True if e1 and e2 share a face
    adj = np.zeros((num_edges, num_edges), dtype=bool)
    for e1 in range(num_edges):
        for e2 in edge_shared_edges_sorted[e1]:
            adj[e1, e2] = True

    def try_greedy(order_seed=None, weight_seed=None):
        rng = random.Random(order_seed)
        chosen = np.zeros(num_edges, dtype=bool)
        count = 0

        if weight_seed is not None:
            wrng = random.Random(weight_seed)
            weights = np.array([wrng.random() for _ in range(num_edges)])
            indices = np.argsort(-weights)
        else:
            indices = np.arange(num_edges)
            rng.shuffle(indices.tolist())
            indices = np.array(indices)

        face_counts = np.zeros(num_faces, dtype=np.int32)

        for idx in indices:
            f_idx_list = edge_to_faces[idx]
            can_add = True
            for f_idx in f_idx_list:
                if face_counts[f_idx] == 3:
                    can_add = False
                    break
            if can_add:
                chosen[idx] = True
                count += 1
                for f_idx in f_idx_list:
                    face_counts[f_idx] += 1

        return count, chosen, face_counts

    def kick_and_rebuild(chosen, face_counts, count, rng, kick_size, deadline):
        """Remove kick_size random chosen edges, then greedily re-add from neighborhood + all unchosen."""
        chosen_edges = np.where(chosen)[0]
        if len(chosen_edges) < kick_size:
            return count, chosen, face_counts

        # Select kick_size edges to remove
        kick_idx = rng.sample(list(chosen_edges), kick_size)

        # Remove them
        for e_idx in kick_idx:
            chosen[e_idx] = False
            count -= 1
            for f_idx in edge_to_faces[e_idx]:
                face_counts[f_idx] -= 1

        # Build candidate pool: all unchosen edges that share a face with any removed edge
        candidate_set = set()
        for e_idx in kick_idx:
            for f_idx in edge_to_faces[e_idx]:
                for e2 in faces_np[f_idx]:
                    if e2 != e_idx and not chosen[e2]:
                        candidate_set.add(e2)

        # Also add some random unchosen edges for diversity
        unchosen = np.where(~chosen)[0]
        if len(unchosen) > 0 and rng.random() < 0.5:
            extra = rng.sample(list(unchosen), min(5, len(unchosen)))
            candidate_set.update(extra)

        # Greedily add back
        candidates = list(candidate_set)
        rng.shuffle(candidates)

        for e_idx in candidates:
            if chosen[e_idx]:
                continue
            valid = True
            for f_idx in edge_to_faces[e_idx]:
                if face_counts[f_idx] == 3:
                    valid = False
                    break
            if valid:
                chosen[e_idx] = True
                count += 1
                for f_idx in edge_to_faces[e_idx]:
                    face_counts[f_idx] += 1

        return count, chosen, face_counts

    def local_search_1_2(chosen, face_counts, count, deadline):
        """Standard remove-1-add-2 local search with fast adjacency checks."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue

                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1

                # Collect candidates: unchosen edges sharing a face with r_idx
                candidate_set = set()
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces_np[f_idx]:
                        if e_idx != r_idx and not chosen[e_idx]:
                            candidate_set.add(e_idx)

                candidates = []
                for e_idx in candidate_set:
                    valid = True
                    for f_idx in edge_to_faces[e_idx]:
                        if face_counts[f_idx] == 3:
                            valid = False
                            break
                    if valid:
                        candidates.append(e_idx)

                # Try to find two compatible edges to add using precomputed adjacency
                added = []
                for i in range(len(candidates)):
                    e1 = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        e2 = candidates[j]
                        # Check if e1 and e2 are compatible (don't share a face)
                        if not adj[e1, e2]:
                            added.append(e1)
                            added.append(e2)
                            break
                    if len(added) >= 2:
                        break

                if len(added) >= 2:
                    for e_idx in added:
                        chosen[e_idx] = True
                        count += 1
                        for f_idx in edge_to_faces[e_idx]:
                            face_counts[f_idx] += 1
                    improved = True
                    break
                else:
                    # Restore r_idx
                    chosen[r_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] += 1

        return count, chosen, face_counts

    def local_search_1_1(chosen, face_counts, count, deadline):
        """Remove-1-add-1 swap local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for r_idx in range(num_edges):
                if not chosen[r_idx]:
                    continue

                # Tentatively remove r_idx
                chosen[r_idx] = False
                count -= 1
                for f_idx in edge_to_faces[r_idx]:
                    face_counts[f_idx] -= 1

                # Find an unchosen edge that can be added
                for f_idx in edge_to_faces[r_idx]:
                    for e_idx in faces_np[f_idx]:
                        if e_idx == r_idx or chosen[e_idx]:
                            continue
                        # Check if e_idx can be added
                        valid = True
                        for f2 in edge_to_faces[e_idx]:
                            if face_counts[f2] == 3:
                                valid = False
                                break
                        if valid:
                            # Perform the swap
                            chosen[e_idx] = True
                            count += 1
                            for f2 in edge_to_faces[e_idx]:
                                face_counts[f2] += 1
                            improved = True
                            break
                    if improved:
                        break

                if not improved:
                    # Restore r_idx
                    chosen[r_idx] = True
                    count += 1
                    for f_idx in edge_to_faces[r_idx]:
                        face_counts[f_idx] += 1

        return count, chosen, face_counts

    attempt = 0
    rng_master = random.Random(42)

    while time.time() < deadline:
        # Use a mix of random and weighted orders
        if attempt % 3 == 0:
            count, chosen, face_counts = try_greedy(attempt)
        elif attempt % 3 == 1:
            count, chosen, face_counts = try_greedy(attempt, weight_seed=attempt + 1000)
        else:
            count, chosen, face_counts = try_greedy(attempt, weight_seed=attempt + 2000)

        # Local search: remove-1-add-2
        count, chosen, face_counts = local_search_1_2(chosen, face_counts, count, deadline)

        # Also try 1-1 swaps to improve further
        if time.time() < deadline:
            count, chosen, face_counts = local_search_1_1(chosen, face_counts, count, deadline)

        # Kick and rebuild with varying kick sizes
        if time.time() < deadline:
            kick_sizes = [4, 8, 12, 16, 20]
            for ks in kick_sizes:
                if time.time() >= deadline:
                    break
                rng_kick = random.Random(attempt * 100 + ks)
                count, chosen, face_counts = kick_and_rebuild(
                    chosen, face_counts, count, rng_kick, ks, deadline
                )
                # Follow up with local search after kick
                count, chosen, face_counts = local_search_1_2(
                    chosen, face_counts, count, deadline
                )
                # Also try 1-1 swaps after kick
                if time.time() < deadline:
                    count, chosen, face_counts = local_search_1_1(
                        chosen, face_counts, count, deadline
                    )

        if count > best_count:
            best_count = count
            best_chosen = chosen.copy()
        attempt += 1

    out = []
    for idx in range(num_edges):
        if best_chosen[idx]:
            u, v, i = all_edges[idx]
            out.append([u, v])
    return out
