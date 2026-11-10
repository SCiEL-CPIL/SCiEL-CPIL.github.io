import time
import random
import math
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute all 4-term arithmetic progressions
    aps = []
    index_to_aps = [[] for _ in range(N)]

    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            idx1 = a - 1
            idx2 = a + d - 1
            idx3 = a + 2 * d - 1
            idx4 = a + 3 * d - 1

            ap = (idx1, idx2, idx3, idx4)
            ap_id = len(aps)
            aps.append(ap)

            index_to_aps[idx1].append((ap_id, 0))
            index_to_aps[idx2].append((ap_id, 1))
            index_to_aps[idx3].append((ap_id, 2))
            index_to_aps[idx4].append((ap_id, 3))

    total_aps = len(aps)

    # Convert to numpy for faster full-count operations
    aps_arr = np.array(aps, dtype=np.int32)

    def count_mono_numpy(colors_np):
        """Count monochromatic APs using numpy vectorization"""
        c0 = colors_np[aps_arr[:, 0]]
        c1 = colors_np[aps_arr[:, 1]]
        c2 = colors_np[aps_arr[:, 2]]
        c3 = colors_np[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))

    # Precompute for each index: list of (ap_id, other_indices_tuple)
    index_info = []
    for i in range(N):
        info = []
        for ap_id, pos in index_to_aps[i]:
            ap = aps[ap_id]
            others = tuple(j for j in ap if j != i)
            info.append((ap_id, others))
        index_info.append(info)

    # Generate diverse initializations
    def generate_initializations():
        inits = []

        # 1. Pure random
        inits.append([random.randint(0, 1) for _ in range(N)])

        # 2. Block pattern with period 2
        inits.append([(i // 2) % 2 for i in range(N)])

        # 3. Block pattern with period 4
        inits.append([(i // 4) % 2 for i in range(N)])

        # 4. Striped pattern
        inits.append([i % 2 for i in range(N)])

        # 5. Thue-Morse-like pattern
        tm = [0]
        for i in range(1, N):
            tm.append(bin(i).count('1') % 2)
        inits.append(tm[:N])

        # 6. Random with bias
        inits.append([random.choice([0, 0, 1]) for _ in range(N)])

        # 7. Another random
        inits.append([random.randint(0, 1) for _ in range(N)])

        # 8. More block patterns
        inits.append([(i // 8) % 2 for i in range(N)])
        inits.append([(i // 16) % 2 for i in range(N)])

        # 9. Block pattern with period 32
        inits.append([(i // 32) % 2 for i in range(N)])

        # 10. Random walk
        rw = [random.randint(0, 1)]
        for i in range(1, N):
            if random.random() < 0.7:
                rw.append(rw[-1])
            else:
                rw.append(1 - rw[-1])
        inits.append(rw)

        # 11. Periodic pattern with period 5
        inits.append([(i // 5) % 2 for i in range(N)])

        # 12. Periodic pattern with period 7
        inits.append([(i // 7) % 2 for i in range(N)])

        # 13. XOR-based pattern
        inits.append([(i >> 1) ^ (i >> 3) ^ (i >> 5) for i in range(N)])

        # 14. Another random
        inits.append([random.randint(0, 1) for _ in range(N)])

        return inits

    # Evaluate initializations and pick the best few
    init_scores = []
    for init in generate_initializations():
        init_np = np.array(init, dtype=np.int32)
        score = count_mono_numpy(init_np)
        init_scores.append((score, init))

    # Sort by score, take top 5 for multi-start
    init_scores.sort(key=lambda x: x[0])
    best_inits = [x[1] for x in init_scores[:5]]

    best_score = min(x[0] for x in init_scores)
    best_colors = best_inits[0][:]

    def compute_delta_1bit(colors, k):
        """Compute delta for flipping bit k"""
        delta = 0
        c_k = colors[k]
        for ap_id, others in index_info[k]:
            c_o1 = colors[others[0]]
            c_o2 = colors[others[1]]
            c_o3 = colors[others[2]]
            if c_o1 == c_o2 == c_o3:
                if c_k == c_o1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    # Local search: iterative single-bit flips until no improvement
    def local_search(colors, current_score, max_iterations=100):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            for k in range(N):
                delta = compute_delta_1bit(colors, k)
                if delta < 0:
                    colors[k] = 1 - colors[k]
                    current_score += delta
                    improved = True
        return colors, current_score

    # Genetic algorithm with simulated annealing
    population_size = 8
    max_generations = 50

    # Initialize population with diverse solutions
    population = []
    for init in best_inits:
        colors = init[:]
        score = count_mono_numpy(np.array(colors, dtype=np.int32))
        population.append((colors, score))

    # Add some random perturbations of best solutions
    for _ in range(population_size - len(population)):
        base = random.choice(population)[0][:]
        for _ in range(random.randint(5, 15)):
            k = random.randint(0, N - 1)
            base[k] = 1 - base[k]
        score = count_mono_numpy(np.array(base, dtype=np.int32))
        population.append((base, score))

    # Sort population by score
    population.sort(key=lambda x: x[1])

    # Track best overall
    if population[0][1] < best_score:
        best_score = population[0][1]
        best_colors = population[0][0][:]

    # Main evolutionary loop
    generation = 0
    while generation < max_generations and time.time() < deadline - 10.0:
        generation += 1

        # Selection: keep top half, replace bottom half with offspring
        elite_size = population_size // 2
        elites = population[:elite_size][:]

        # Generate offspring
        offspring = []
        for _ in range(population_size - elite_size):
            # Tournament selection
            candidates = random.sample(range(len(population)), min(3, len(population)))
            parent1 = min(candidates, key=lambda i: population[i][1])
            candidates = random.sample(range(len(population)), min(3, len(population)))
            parent2 = min(candidates, key=lambda i: population[i][1])

            # Crossover
            child = population[parent1][0][:]
            crossover_point = random.randint(0, N)
            child[crossover_point:] = population[parent2][0][crossover_point:]

            # Mutation
            mutation_rate = random.uniform(0.02, 0.08)
            for k in range(N):
                if random.random() < mutation_rate:
                    child[k] = 1 - child[k]

            # Local search on child
            score = count_mono_numpy(np.array(child, dtype=np.int32))
            child, score = local_search(child, score, max_iterations=30)

            offspring.append((child, score))

        # Combine elites and offspring
        population = elites + offspring
        population.sort(key=lambda x: x[1])

        # Update best
        if population[0][1] < best_score:
            best_score = population[0][1]
            best_colors = population[0][0][:]

        # Occasionally run SA on the best solution
        if generation % 5 == 0 and time.time() < deadline - 15.0:
            colors = best_colors[:]
            current_score = best_score

            # Simulated annealing
            T = 1.0
            cooling_rate = 0.9999
            min_T = 0.01
            steps_since_improvement = 0
            max_steps_no_improve = 3000

            while time.time() < deadline - 15.0 and T > min_T:
                k = random.randint(0, N - 1)
                delta = compute_delta_1bit(colors, k)

                if delta < 0:
                    colors[k] = 1 - colors[k]
                    current_score += delta
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors[:]
                        steps_since_improvement = 0
                    else:
                        steps_since_improvement += 1
                else:
                    if T > 0.01 and delta < 20 * T:
                        if random.random() < math.exp(-delta / T):
                            colors[k] = 1 - colors[k]
                            current_score += delta
                            if current_score < best_score:
                                best_score = current_score
                                best_colors = colors[:]
                                steps_since_improvement = 0
                            else:
                                steps_since_improvement += 1
                    else:
                        steps_since_improvement += 1

                T *= cooling_rate

                if steps_since_improvement > max_steps_no_improve:
                    break

    # Final polish: iterative single-bit flips on best solution
    if time.time() < deadline - 3.0:
        colors = best_colors[:]
        current_score = best_score
        colors, current_score = local_search(colors, current_score, max_iterations=200)
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]

    # Final verification
    final_score = count_mono_numpy(np.array(best_colors, dtype=np.int32))
    if final_score < best_score:
        best_score = final_score

    return best_colors
