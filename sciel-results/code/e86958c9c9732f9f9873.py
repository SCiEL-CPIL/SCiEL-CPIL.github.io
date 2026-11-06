"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 55  # 55 second deadline for safety
    
    # Denominators are 2 to 48, indices 0 to 46
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]
    
    # Start with greedy solution
    selected, total = [], Fraction()
    for i, d in enumerate(range(2, 49)):
        if total + Fraction(1, d) < 1:
            selected.append(i)
            total += Fraction(1, d)
    
    best_indices = set(selected)
    best_sum = total
    best_deficit = Fraction(1, 1) - total
    
    # Local search: try swapping one selected index with one unselected index
    unselected = [i for i in range(n) if i not in best_indices]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_list = list(best_indices)
        
        for i in range(len(current_list)):
            for j in unselected:
                new_sum = best_sum - fracs[current_list[i]] + fracs[j]
                if new_sum < 1:
                    new_deficit = Fraction(1, 1) - new_sum
                    if new_deficit < best_deficit:
                        best_indices = best_indices - {current_list[i]} | {j}
                        best_sum = new_sum
                        best_deficit = new_deficit
                        improved = True
                        unselected = [x for x in unselected if x != j]
                        if current_list[i] not in unselected:
                            unselected.append(current_list[i])
                        current_list = list(best_indices)
                        break
            if improved:
                break
    
    return list(best_indices)
# EVOLVE-BLOCK-END
