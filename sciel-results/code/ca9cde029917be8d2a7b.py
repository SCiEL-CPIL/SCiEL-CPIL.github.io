import time
import random
from itertools import combinations

# Precomputation
N = 16
ALL_TRIPLES = list(combinations(range(N), 3))
NUM_TRIPLES = len(ALL_TRIPLES)  # 560

# Map triple to index
triple_to_idx = {t: i for i, t in enumerate(ALL_TRIPLES)}

# Generate all K(2,2,2) configurations
def generate_k222s():
    k222s = []
    for verts in combinations(range(N), 6):
        v = list(verts)
        for i in range(1, 6):
            pair0 = (v[0], v[i])
            remaining = [v[j] for j in range(6) if j != 0 and j != i]
            
            # Option 1
            pair1 = (remaining[0], remaining[1])
            pair2 = (remaining[2], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 2
            pair1 = (remaining[0], remaining[2])
            pair2 = (remaining[1], remaining[3])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
            
            # Option 3
            pair1 = (remaining[0], remaining[3])
            pair2 = (remaining[1], remaining[2])
            edges = set()
            for x in pair0:
                for y in pair1:
                    for z in pair2:
                        edges.add(tuple(sorted((x, y, z))))
            k222s.append([triple_to_idx[e] for e in edges])
    return k222s

K222S = generate_k222s()
NUM_K222 = len(K222S)

# Map each edge to the list of K222 indices it belongs to
edge_to_k222s = [[] for _ in range(NUM_TRIPLES)]
for k_idx, edges in enumerate(K222S):
    for e_idx in edges:
        edge_to_k222s[e_idx].append(k_idx)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_count = 0
    best_system = []

    triple_list = ALL_TRIPLES

    while time.time() < deadline:
        # 1. Greedy Initialization
        shuffled_indices = list(range(NUM_TRIPLES))
        random.shuffle(shuffled_indices)

        present = [False] * NUM_TRIPLES
        counts = [0] * NUM_K222
        current_count = 0

        for idx in shuffled_indices:
            valid = True
            for k_idx in edge_to_k222s[idx]:
                if counts[k_idx] == 7:
                    valid = False
                    break
            if valid:
                present[idx] = True
                current_count += 1
                for k_idx in edge_to_k222s[idx]:
                    counts[k_idx] += 1

        if current_count > best_count:
            best_count = current_count
            best_system = [triple_list[i] for i in range(NUM_TRIPLES) if present[i]]

        # 2. Local Search with simulated annealing
        absent_set = set(i for i in range(NUM_TRIPLES) if not present[i])
        present_set = set(i for i in range(NUM_TRIPLES) if present[i])

        # Simulated annealing parameters
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.9995
        max_steps_per_restart = 5000

        step = 0
        while time.time() < deadline and step < max_steps_per_restart:
            step += 1

            move_type = random.random()

            if move_type < 0.6:
                # Swap Move: Pick an absent triple, try to make room for it
                if not absent_set:
                    break

                new_idx = random.choice(list(absent_set))

                # Check if new_idx is blocked
                blocking_k222s = []
                for k_idx in edge_to_k222s[new_idx]:
                    if counts[k_idx] == 7:
                        blocking_k222s.append(k_idx)

                if not blocking_k222s:
                    # Not blocked, just add it
                    present[new_idx] = True
                    absent_set.remove(new_idx)
                    present_set.add(new_idx)
                    current_count += 1
                    for k_idx in edge_to_k222s[new_idx]:
                        counts[k_idx] += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [triple_list[i] for i in present_set]
                else:
                    # It is blocked. We need to remove one edge from one of the blocking K222s.
                    k_idx = random.choice(blocking_k222s)
                    edges_in_k222 = K222S[k_idx]
                    present_edges_in_k222 = [e for e in edges_in_k222 if present[e]]

                    if present_edges_in_k222:
                        rem_idx = random.choice(present_edges_in_k222)

                        # Remove rem_idx
                        present[rem_idx] = False
                        absent_set.add(rem_idx)
                        present_set.remove(rem_idx)
                        current_count -= 1
                        for k_idx_2 in edge_to_k222s[rem_idx]:
                            counts[k_idx_2] -= 1

                        # Try to add new_idx
                        valid_add = True
                        for k_idx_3 in edge_to_k222s[new_idx]:
                            if counts[k_idx_3] == 7:
                                valid_add = False
                                break

                        if valid_add:
                            # Add new_idx
                            present[new_idx] = True
                            absent_set.remove(new_idx)
                            present_set.add(new_idx)
                            current_count += 1
                            for k_idx_4 in edge_to_k222s[new_idx]:
                                counts[k_idx_4] += 1
                            # Net change 0, but structure improved
                        else:
                            # Revert removal
                            present[rem_idx] = True
                            absent_set.remove(rem_idx)
                            present_set.add(rem_idx)
                            current_count += 1
                            for k_idx_5 in edge_to_k222s[rem_idx]:
                                counts[k_idx_5] += 1

            elif move_type < 0.8:
                # Add Move: Pick an absent triple, add if possible
                if not absent_set:
                    break
                new_idx = random.choice(list(absent_set))
                valid = True
                for k_idx in edge_to_k222s[new_idx]:
                    if counts[k_idx] == 7:
                        valid = False
                        break
                if valid:
                    present[new_idx] = True
                    absent_set.remove(new_idx)
                    present_set.add(new_idx)
                    current_count += 1
                    for k_idx in edge_to_k222s[new_idx]:
                        counts[k_idx] += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_system = [triple_list[i] for i in present_set]

            else:
                # Remove Move: Remove a random present triple (to allow future adds)
                if present_set:
                    rem_idx = random.choice(list(present_set))
                    present[rem_idx] = False
                    absent_set.add(rem_idx)
                    present_set.remove(rem_idx)
                    current_count -= 1
                    for k_idx in edge_to_k222s[rem_idx]:
                        counts[k_idx] -= 1

            # Cool down
            T = max(T * cooling_rate, T_min)

    return best_system
