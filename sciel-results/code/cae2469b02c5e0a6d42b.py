import time
import random
import copy

N = 18
TOTAL_INPUTS = 1 << N
DEADLINE = 52.0

def construct():
    start_time = time.time()

    def make_mask(i):
        L = 1 << (i + 1)
        num_blocks = TOTAL_INPUTS // L
        ones_val = (1 << (1 << i)) - 1
        block = ones_val << (1 << i)
        G = ((1 << TOTAL_INPUTS) - 1) // ((1 << L) - 1)
        mask = block * G
        return mask

    masks = [make_mask(i) for i in range(N)]

    def verify_network(network):
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return False
        return True

    def find_violation(network):
        wires = masks[:]
        for i, j in network:
            new_wi = wires[i] & wires[j]
            new_wj = wires[i] | wires[j]
            wires[i] = new_wi
            wires[j] = new_wj
        for i in range(N - 1):
            if (wires[i] & ~wires[i+1]) != 0:
                return (i, i + 1)
        return None

    # Seed: Odd-Even Transposition
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    best_network = network[:]
    best_size = len(network)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

    def update_best(net):
        nonlocal best_network, best_size
        if len(net) < best_size:
            best_network = net[:]
            best_size = len(net)

    # Main Search Loop
    while (time.time() - start_time) < DEADLINE - 2:
        
        # 1. Try Greedy Removal - scan multiple positions
        removed = False
        idx = 0
        while idx < len(network):
            if (time.time() - start_time) > DEADLINE - 2:
                break
            candidate = network[:]
            candidate.pop(idx)
            if verify_network(candidate):
                network = candidate
                update_best(network)
                removed = True
                # Don't increment idx, re-check same position
            else:
                idx += 1
        if removed:
            continue

        # 2. Try Random Swap - try multiple swaps
        if len(network) > 1:
            swapped = False
            for _ in range(5):
                i1 = random.randint(0, len(network) - 1)
                i2 = random.randint(0, len(network) - 1)
                if i1 == i2:
                    continue
                candidate = network[:]
                candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                if verify_network(candidate):
                    network = candidate
                    swapped = True
                    break
            if swapped:
                continue

        # 3. Try Remove + Add (Error Guided)
        idx = random.randint(0, len(network) - 1)
        candidate = network[:]
        removed_comp = candidate.pop(idx)
        
        if verify_network(candidate):
            network = candidate
            update_best(network)
            continue
            
        violation = find_violation(candidate)
        if not violation:
            continue
            
        vi, vj = violation
        
        repair_candidates = set()
        for w in range(N):
            if w != vi:
                a, b = (w, vi) if w < vi else (vi, w)
                repair_candidates.add((a, b))
            if w != vj:
                a, b = (w, vj) if w < vj else (vj, w)
                repair_candidates.add((a, b))
        
        # Add the removed comparator as a candidate
        repair_candidates.add(removed_comp)
        
        # Add more random candidates
        for _ in range(8):
            repair_candidates.add(random.choice(all_pairs))
            
        repair_list = list(repair_candidates)
        random.shuffle(repair_list)
        
        repaired = False
        for (ri, rj) in repair_list:
            if (time.time() - start_time) > DEADLINE - 2:
                break
                
            # Try inserting at multiple positions: near idx, near violation, 0, end, random
            positions = [idx, 0, len(candidate)]
            # Add positions near the violation wires
            positions.append(max(0, idx - 1))
            positions.append(min(len(candidate), idx + 1))
            # Add a few random positions
            for _ in range(3):
                positions.append(random.randint(0, len(candidate)))
                
            # Deduplicate and sort
            positions = sorted(set(max(0, min(len(candidate), p)) for p in positions))
            
            for pos in positions:
                temp = candidate[:]
                temp.insert(pos, (ri, rj))
                if verify_network(temp):
                    network = temp
                    repaired = True
                    break
            if repaired:
                break
        
        # 4. Occasional restart from fresh seed to escape local minima
        if not repaired and random.random() < 0.05:
            network = []
            for phase in range(N):
                for i in range(phase % 2, N - 1, 2):
                    network.append((i, i + 1))
            update_best(network)
            continue
            
    return best_network
