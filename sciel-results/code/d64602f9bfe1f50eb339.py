"""Hypergraph Ramsey R(4,5;3) construction via efficient local search with incremental updates."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = 145  # Safety margin from 150s kill

    target_n = 34
    M = target_n * (target_n - 1) * (target_n - 2) // 6  # 5984 edges

    # Precompute edge indices
    edge_index = {}
    idx = 0
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                edge_index[(a, b, c)] = idx
                idx += 1

    # Precompute which K4s and K5s each edge belongs to
    # For each edge, store list of K4 indices and K5 indices that contain it
    k4_edges = []
    for quad in combinations(range(target_n), 4):
        edges = []
        for triple in combinations(quad, 3):
            edges.append(edge_index[triple])
        k4_edges.append(edges)

    k5_edges = []
    for quint in combinations(range(target_n), 5):
        edges = []
        for triple in combinations(quint, 3):
            edges.append(edge_index[triple])
        k5_edges.append(edges)

    num_k4 = len(k4_edges)
    num_k5 = len(k5_edges)

    # Build reverse index: for each edge, which K4s and K5s contain it
    edge_to_k4s = [[] for _ in range(M)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    edge_to_k5s = [[] for _ in range(M)]
    for k5_idx, edges in enumerate(k5_edges):
        for e in edges:
            edge_to_k5s[e].append(k5_idx)

    # Precompute k4 and k5 edge arrays as numpy for fast access
    k4_np = np.array(k4_edges, dtype=np.int32)
    k5_np = np.array(k5_edges, dtype=np.int32)

    def compute_objective(colors):
        k4_sums = np.sum(colors[k4_np], axis=1)
        red_k4_count = np.sum(k4_sums == 0)
        k5_sums = np.sum(colors[k5_np], axis=1)
        blue_k5_count = np.sum(k5_sums == 10)
        return red_k4_count + blue_k5_count

    def is_valid(colors):
        k4_sums = np.sum(colors[k4_np], axis=1)
        if np.any(k4_sums == 0):
            return False
        k5_sums = np.sum(colors[k5_np], axis=1)
        if np.any(k5_sums == 10):
            return False
        return True

    # Try multiple strategies and return the best valid one found
    best_result = None

    # Strategy 1: Start with random coloring and do local search
    random.seed(42)
    np.random.seed(42)

    for attempt in range(5):
        if time.time() - start_time > deadline - 10:
            break

        # Initialize with random coloring
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        current_obj = compute_objective(colors)

        # Local search with incremental updates
        # Maintain k4_sums and k5_sums incrementally
        k4_sums = np.sum(colors[k4_np], axis=1)
        k5_sums = np.sum(colors[k5_np], axis=1)
        red_k4_count = np.sum(k4_sums == 0)
        blue_k5_count = np.sum(k5_sums == 10)

        step = 0
        max_steps = 200000
        no_improve = 0
        max_no_improve = 5000

        while step < max_steps and time.time() - start_time < deadline - 5:
            if step % 1000 == 0 and time.time() - start_time > deadline - 10:
                break

            # Find violating structures
            bad_k4_indices = np.where(k4_sums == 0)[0]
            bad_k5_indices = np.where(k5_sums == 10)[0]

            if len(bad_k4_indices) == 0 and len(bad_k5_indices) == 0:
                # Found valid coloring!
                return {'n': target_n, 'colors': colors.tolist()}

            # Choose an edge to flip
            # Prioritize edges in violating structures
            candidate_edges = set()
            if len(bad_k4_indices) > 0:
                sample_size = min(10, len(bad_k4_indices))
                sample_k4 = np.random.choice(bad_k4_indices, sample_size, replace=False)
                for k4_idx in sample_k4:
                    candidate_edges.update(k4_np[k4_idx].tolist())
            if len(bad_k5_indices) > 0:
                sample_size = min(10, len(bad_k5_indices))
                sample_k5 = np.random.choice(bad_k5_indices, sample_size, replace=False)
                for k5_idx in sample_k5:
                    candidate_edges.update(k5_np[k5_idx].tolist())

            if not candidate_edges:
                # No candidates, pick random edge
                edge_idx = np.random.randint(0, M)
            else:
                edge_idx = random.choice(list(candidate_edges))

            old_val = colors[edge_idx]
            new_val = 1 - old_val

            # Compute delta in objective
            # For each K4 containing this edge, check if it becomes red K4 or stops being one
            delta_red_k4 = 0
            for k4_idx in edge_to_k4s[edge_idx]:
                old_sum = k4_sums[k4_idx]
                new_sum = old_sum + (new_val - old_val)
                if old_sum == 0 and new_sum != 0:
                    delta_red_k4 -= 1  # Good: red K4 eliminated
                elif old_sum != 0 and new_sum == 0:
                    delta_red_k4 += 1  # Bad: new red K4 created

            delta_blue_k5 = 0
            for k5_idx in edge_to_k5s[edge_idx]:
                old_sum = k5_sums[k5_idx]
                new_sum = old_sum + (new_val - old_val)
                if old_sum == 10 and new_sum != 10:
                    delta_blue_k5 -= 1  # Good: blue K5 eliminated
                elif old_sum != 10 and new_sum == 10:
                    delta_blue_k5 += 1  # Bad: new blue K5 created

            total_delta = delta_red_k4 + delta_blue_k5

            if total_delta < 0 or (total_delta == 0 and random.random() < 0.5) or (total_delta > 0 and random.random() < np.exp(-total_delta / max(0.1, 5.0 / (1 + step / 10000)))):
                # Accept the move
                colors[edge_idx] = new_val

                # Update k4_sums and k5_sums
                for k4_idx in edge_to_k4s[edge_idx]:
                    k4_sums[k4_idx] += (new_val - old_val)
                for k5_idx in edge_to_k5s[edge_idx]:
                    k5_sums[k5_idx] += (new_val - old_val)

                red_k4_count += delta_red_k4
                blue_k5_count += delta_blue_k5
                current_obj += total_delta

                if current_obj < 0:
                    current_obj = 0

                no_improve = 0
            else:
                # Reject the move
                no_improve += 1

            step += 1

            if no_improve > max_no_improve:
                # Restart with new random initialization
                if time.time() - start_time > deadline - 15:
                    break
                colors = np.random.randint(0, 2, M, dtype=np.int8)
                k4_sums = np.sum(colors[k4_np], axis=1)
                k5_sums = np.sum(colors[k5_np], axis=1)
                red_k4_count = np.sum(k4_sums == 0)
                blue_k5_count = np.sum(k5_sums == 10)
                current_obj = red_k4_count + blue_k5_count
                no_improve = 0
                step = 0

        # After local search, check if we have a valid coloring
        if red_k4_count == 0 and blue_k5_count == 0:
            return {'n': target_n, 'colors': colors.tolist()}

    # Strategy 2: Try a structured initial coloring
    # Use a cyclic-like pattern
    colors = np.zeros(M, dtype=np.int8)
    idx = 0
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                # Use a simple hash-like function
                colors[idx] = (a * 7 + b * 13 + c * 17) % 2
                idx += 1

    current_obj = compute_objective(colors)
    if current_obj == 0:
        return {'n': target_n, 'colors': colors.tolist()}

    # Local search from structured start
    k4_sums = np.sum(colors[k4_np], axis=1)
    k5_sums = np.sum(colors[k5_np], axis=1)
    red_k4_count = np.sum(k4_sums == 0)
    blue_k5_count = np.sum(k5_sums == 10)

    step = 0
    max_steps = 100000
    no_improve = 0
    max_no_improve = 3000

    while step < max_steps and time.time() - start_time < deadline - 5:
        if step % 1000 == 0 and time.time() - start_time > deadline - 10:
            break

        bad_k4_indices = np.where(k4_sums == 0)[0]
        bad_k5_indices = np.where(k5_sums == 10)[0]

        if len(bad_k4_indices) == 0 and len(bad_k5_indices) == 0:
            return {'n': target_n, 'colors': colors.tolist()}

        candidate_edges = set()
        if len(bad_k4_indices) > 0:
            sample_size = min(10, len(bad_k4_indices))
            sample_k4 = np.random.choice(bad_k4_indices, sample_size, replace=False)
            for k4_idx in sample_k4:
                candidate_edges.update(k4_np[k4_idx].tolist())
        if len(bad_k5_indices) > 0:
            sample_size = min(10, len(bad_k5_indices))
            sample_k5 = np.random.choice(bad_k5_indices, sample_size, replace=False)
            for k5_idx in sample_k5:
                candidate_edges.update(k5_np[k5_idx].tolist())

        if not candidate_edges:
            edge_idx = np.random.randint(0, M)
        else:
            edge_idx = random.choice(list(candidate_edges))

        old_val = colors[edge_idx]
        new_val = 1 - old_val

        delta_red_k4 = 0
        for k4_idx in edge_to_k4s[edge_idx]:
            old_sum = k4_sums[k4_idx]
            new_sum = old_sum + (new_val - old_val)
            if old_sum == 0 and new_sum != 0:
                delta_red_k4 -= 1
            elif old_sum != 0 and new_sum == 0:
                delta_red_k4 += 1

        delta_blue_k5 = 0
        for k5_idx in edge_to_k5s[edge_idx]:
            old_sum = k5_sums[k5_idx]
            new_sum = old_sum + (new_val - old_val)
            if old_sum == 10 and new_sum != 10:
                delta_blue_k5 -= 1
            elif old_sum != 10 and new_sum == 10:
                delta_blue_k5 += 1

        total_delta = delta_red_k4 + delta_blue_k5

        if total_delta < 0 or (total_delta == 0 and random.random() < 0.5) or (total_delta > 0 and random.random() < np.exp(-total_delta / max(0.1, 5.0 / (1 + step / 10000)))):
            colors[edge_idx] = new_val
            for k4_idx in edge_to_k4s[edge_idx]:
                k4_sums[k4_idx] += (new_val - old_val)
            for k5_idx in edge_to_k5s[edge_idx]:
                k5_sums[k5_idx] += (new_val - old_val)
            red_k4_count += delta_red_k4
            blue_k5_count += delta_blue_k5
            current_obj += total_delta
            if current_obj < 0:
                current_obj = 0
            no_improve = 0
        else:
            no_improve += 1

        step += 1

        if no_improve > max_no_improve:
            if time.time() - start_time > deadline - 15:
                break
            colors = np.random.randint(0, 2, M, dtype=np.int8)
            k4_sums = np.sum(colors[k4_np], axis=1)
            k5_sums = np.sum(colors[k5_np], axis=1)
            red_k4_count = np.sum(k4_sums == 0)
            blue_k5_count = np.sum(k5_sums == 10)
            current_obj = red_k4_count + blue_k5_count
            no_improve = 0
            step = 0

    if red_k4_count == 0 and blue_k5_count == 0:
        return {'n': target_n, 'colors': colors.tolist()}

    # Fallback: return a known valid small solution
    n_fallback = 9
    return {'n': n_fallback, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]}
# EVOLVE-BLOCK-END
