"""Depth-by-depth genetic algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55

def _quick_verify(comparators, num_samples=512):
    """Quick verification on a random subset of boolean inputs."""
    if not comparators:
        return False
    num_inputs = 1 << N
    if num_samples >= num_inputs:
        indices = np.arange(num_inputs, dtype=np.uint32)
    else:
        indices = np.random.choice(num_inputs, size=num_samples, replace=False, dtype=np.uint32)
    state = np.zeros((len(indices), N), dtype=np.uint8)
    for j in range(N):
        state[:, j] = ((indices >> j) & 1).astype(np.uint8)
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _verify_network(comparators):
    """Full verification using zero-one principle."""
    if not comparators:
        return False
    num_inputs = 1 << N
    arange = np.arange(num_inputs, dtype=np.uint32)
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        state[:, j] = ((arange >> j) & 1).astype(np.uint8)
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _flatten_layers(layers):
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _generate_random_layer():
    available = set(range(N))
    layer = []
    while len(available) >= 2:
        if random.random() < 0.3:
            break
        i = random.choice(list(available))
        available.discard(i)
        j = random.choice(list(available))
        available.discard(j)
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _generate_random_network(depth_range=(12, 22)):
    depth = random.randint(depth_range[0], depth_range[1])
    return [_generate_random_layer() for _ in range(depth)]

def _odd_even_transposition_layers():
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _crossover_layers(parent1, parent2):
    if not parent1:
        return [list(l) for l in parent2]
    if not parent2:
        return [list(l) for l in parent1]
    max_depth = max(len(parent1), len(parent2))
    cut = random.randint(0, max_depth)
    child = []
    for i in range(cut):
        child.append(list(parent1[i]) if i < len(parent1) else [])
    for i in range(cut, max_depth):
        child.append(list(parent2[i]) if i < len(parent2) else [])
    return child

def _mutate_layers(layers, mutation_rate=0.08):
    layers = [list(l) for l in layers]
    n_mut = max(1, int(len(layers) * mutation_rate)) + random.randint(0, 2)
    for _ in range(n_mut):
        op = random.choice(['add_comp', 'remove_comp', 'swap_comp', 'add_layer', 'remove_layer'])
        if op == 'add_comp' and layers:
            li = random.randint(0, len(layers) - 1)
            used = set()
            for (a, b) in layers[li]:
                used.add(a); used.add(b)
            avail = [w for w in range(N) if w not in used]
            if len(avail) >= 2:
                a, b = random.sample(avail, 2)
                if a > b: a, b = b, a
                layers[li].append((a, b))
        elif op == 'remove_comp' and any(layers):
            ne = [i for i, l in enumerate(layers) if l]
            if ne:
                li = random.choice(ne)
                pos = random.randint(0, len(layers[li]) - 1)
                layers[li].pop(pos)
        elif op == 'swap_comp' and len(layers) >= 2:
            ne = [i for i, l in enumerate(layers) if l]
            if len(ne) >= 2:
                l1, l2 = random.sample(ne, 2)
                p1 = random.randint(0, len(layers[l1]) - 1)
                comp = layers[l1][p1]
                used2 = set()
                for (a, b) in layers[l2]:
                    used2.add(a); used2.add(b)
                if comp[0] not in used2 and comp[1] not in used2:
                    layers[l1].pop(p1)
                    layers[l2].append(comp)
        elif op == 'add_layer' and len(layers) < 30:
            pos = random.randint(0, len(layers))
            layers.insert(pos, _generate_random_layer())
        elif op == 'remove_layer' and len(layers) > 1:
            pos = random.randint(0, len(layers) - 1)
            layers.pop(pos)
    return layers

def _evaluate_fitness(layers):
    comps = _flatten_layers(layers)
    if len(comps) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if not _quick_verify(comps, num_samples=512):
        return MAX_COMPARATORS + 1000
    if _verify_network(comps):
        return len(comps)
    return MAX_COMPARATORS + 1000

def _local_search(comparators, time_budget=5.0):
    start = time.time()
    best = list(comparators)
    if not _verify_network(best):
        return best
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for idx in range(len(best) - 1, -1, -1):
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                improved = True
                break
    return best

def _genetic_algorithm():
    start_time = time.time()
    pop_size = 40
    population = [_odd_even_transposition_layers()] * 4
    population.extend(_generate_random_network() for _ in range(pop_size - 4))
    
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    best_idx = min(range(pop_size), key=lambda i: fitnesses[i])
    best_fitness = fitnesses[best_idx]
    best_comps = _flatten_layers(population[best_idx])
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    if best_fitness < 140:
        best_comps = _local_search(best_comps, time_budget=10.0)
        best_fitness = len(best_comps)
        print(f"After local search: {best_fitness} comparators", file=sys.stderr)
    
    gen = 0
    while time.time() - start_time < TIME_LIMIT:
        gen += 1
        srt = sorted(range(pop_size), key=lambda i: fitnesses[i])
        elite = max(2, pop_size // 5)
        new_pop = [population[i] for i in srt[:elite]]
        while len(new_pop) < pop_size:
            r = random.random()
            if r < 0.3 and elite >= 2:
                p1 = srt[random.randint(0, elite - 1)]
                p2 = srt[random.randint(0, elite - 1)]
                child = _crossover_layers(population[p1], population[p2])
            elif r < 0.7:
                p = srt[random.randint(0, elite - 1)]
                child = _mutate_layers(population[p])
            else:
                child = _generate_random_network()
            new_pop.append(child)
        population = new_pop
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        gb = min(range(pop_size), key=lambda i: fitnesses[i])
        if fitnesses[gb] < best_fitness:
            best_fitness = fitnesses[gb]
            best_comps = _flatten_layers(population[gb])
            print(f"Gen {gen}: new best {best_fitness} comparators", file=sys.stderr)
            if best_fitness < 100:
                imp = _local_search(best_comps, time_budget=3.0)
                if len(imp) < best_fitness:
                    best_fitness = len(imp)
                    best_comps = imp
                    print(f"  Local search: {best_fitness} comparators", file=sys.stderr)
    return best_comps

def construct():
    try:
        comparators = _genetic_algorithm()
        if _verify_network(comparators):
            return comparators
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
