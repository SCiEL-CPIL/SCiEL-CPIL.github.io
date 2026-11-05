import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Specific to q=3, n=6, radius=1
    # General fallback for other params if needed, but task is specific.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to simple greedy or tetracode
        # Tetracode sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    # Map word tuple to index 0..728
    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    def to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute balls as numpy arrays for efficient operations
    balls = []
    for i in range(total_words):
        w = to_word(i)
        covered = {i}
        for pos in range(n):
            orig = w[pos]
            for new_val in range(3):
                if new_val == orig:
                    continue
                w[pos] = new_val
                covered.add(to_idx(w))
            w[pos] = orig
        balls.append(np.array(list(covered), dtype=np.int32))

    # Initial Code: Tetracode Sum
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # counts[word_idx] = number of codewords in current set covering word_idx
    counts = np.zeros(total_words, dtype=np.int32)
    for c in initial_code:
        counts[balls[c]] += 1
    
    best_code = list(initial_code)
    best_size = len(best_code)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    current_code = list(initial_code)
    current_counts = counts.copy()
    
    def try_remove(c):
        # Check if removing c is safe
        # We need to ensure all words in balls[c] have count > 1
        # If any word in balls[c] has count == 1, removing c makes it 0 -> invalid
        return np.all(current_counts[balls[c]] > 1)

    def do_remove(c):
        current_counts[balls[c]] -= 1

    def do_add(c):
        current_counts[balls[c]] += 1

    def fitness(code_list):
        # Fitness based on sum of squared coverage counts
        # Lower is better (more uniform coverage)
        if len(code_list) == 0:
            return float('inf')
        c_counts = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            c_counts[balls[c]] += 1
        return np.sum(c_counts ** 2)

    # Phase 1: Greedy Remove
    while True:
        removed_any = False
        # Randomize order for variety
        random.shuffle(current_code)
        for c in current_code[:]:
            if try_remove(c):
                do_remove(c)
                current_code.remove(c)
                removed_any = True
        if not removed_any:
            break
            
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = list(current_code)
        
    # Phase 2: Genetic Algorithm with Local Search
    # Population of valid codes
    population = [list(current_code)]
    
    # Add some diverse initial solutions
    for _ in range(5):
        pop = list(best_code)
        # Randomly remove some redundant words
        random.shuffle(pop)
        for c in pop[:]:
            if try_remove(c):
                do_remove(c)
                pop.remove(c)
        # Add some random words
        code_set = set(pop)
        for _ in range(10):
            rand_idx = random.randint(0, total_words - 1)
            if rand_idx not in code_set:
                pop.append(rand_idx)
                do_add(rand_idx)
                code_set.add(rand_idx)
        # Greedy remove again
        random.shuffle(pop)
        for c in pop[:]:
            if try_remove(c):
                do_remove(c)
                pop.remove(c)
        population.append(pop)
    
    # Restore original state
    current_counts = np.zeros(total_words, dtype=np.int32)
    for c in current_code:
        current_counts[balls[c]] += 1
    current_code = list(best_code)
    
    while time.time() < deadline:
        # Selection: keep best half
        population.sort(key=lambda x: len(x))
        population = population[:len(population)//2 + 1]
        
        # Crossover: combine two codes
        new_population = []
        for _ in range(len(population)//2):
            p1, p2 = random.sample(population, 2)
            # Take union, then reduce
            child = list(set(p1 + p2))
            # Greedy remove
            random.shuffle(child)
            child_counts = np.zeros(total_words, dtype=np.int32)
            for c in child:
                child_counts[balls[c]] += 1
            
            def try_remove_child(c):
                return np.all(child_counts[balls[c]] > 1)
            
            def do_remove_child(c):
                child_counts[balls[c]] -= 1
            
            for c in child[:]:
                if try_remove_child(c):
                    do_remove_child(c)
                    child.remove(c)
            
            new_population.append(child)
        
        # Mutation: add/remove random words
        for child in new_population:
            # Try adding a random word
            if random.random() < 0.5:
                code_set = set(child)
                for _ in range(5):
                    rand_idx = random.randint(0, total_words - 1)
                    if rand_idx not in code_set:
                        child.append(rand_idx)
                        code_set.add(rand_idx)
                        break
            
            # Greedy remove
            random.shuffle(child)
            child_counts = np.zeros(total_words, dtype=np.int32)
            for c in child:
                child_counts[balls[c]] += 1
            
            def try_remove_child(c):
                return np.all(child_counts[balls[c]] > 1)
            
            def do_remove_child(c):
                child_counts[balls[c]] -= 1
            
            for c in child[:]:
                if try_remove_child(c):
                    do_remove_child(c)
                    child.remove(c)
        
        # Combine and select
        population.extend(new_population)
        population.sort(key=lambda x: len(x))
        population = population[:10]  # Keep top 10
        
        # Update best
        if len(population[0]) < best_size:
            best_size = len(population[0])
            best_code = list(population[0])
    
    # Convert best_code indices back to lists of ints
    return [to_word(idx) for idx in best_code]
