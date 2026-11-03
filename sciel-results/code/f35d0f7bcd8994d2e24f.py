    def minimize_solution(solution, seed=None):
        if not solution:
            return solution
            
        if seed is not None:
            random.seed(seed)
            random.shuffle(solution)
            
        cover_count = [0] * num_subsets
        for b_idx in solution:
            for s_idx in block_covered_subsets[b_idx]:
                cover_count[s_idx] += 1
                
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_solution = []
            for b_idx in solution:
                essential = False
                for s_idx in block_covered_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        essential = True
                        break
                if not essential:
                    for s_idx in block_covered_subsets[b_idx]:
                        cover_count[s_idx] -= 1
                    changed = True
                else:
                    new_solution.append(b_idx)
            solution = new_solution
        return solution
