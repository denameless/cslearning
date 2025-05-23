import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgTSP:
    def __init__(self, filename):
        self.city_ids = []
        self.coords = []
        with open(filename, 'r') as f:
            line = f.readline().strip()
            while line and not line.startswith('NODE_COORD_SECTION'):
                line = f.readline().strip()
            # Read city coordinates
            for line in f:
                line = line.strip()
                if line in ('EOF', ''):
                    break
                parts = line.split()
                city_id = int(parts[0])
                x, y = map(float, parts[1:3])
                self.city_ids.append(city_id)
                self.coords.append([x, y])
        self.coords = np.array(self.coords)
        self.pop_size = 100
        self.mutation_rate = 0.10
        self.tournament_size = 5
        # Initialize population
        n = len(self.city_ids)
        self.population = [random.sample(range(n), n) for _ in range(self.pop_size)]
        self.best_dist = []
    def distance(self, i, j):
        return np.linalg.norm(self.coords[i] - self.coords[j])

    def calculate_total_distance(self, path):
        total = 0.0
        n = len(path)
        for i in range(n):
            total += self.distance(path[i], path[(i+1) % n])
        return total

    def select_parents(self, fitness): # 锦标赛
        selected = []
        for _ in range(self.pop_size):
            candidates = random.sample(range(self.pop_size), self.tournament_size)
            best_idx = max(candidates, key=lambda idx: fitness[idx])
            selected.append(self.population[best_idx])
        return selected
    def select_parentst(self, fitness): # 轮盘赌
        total_fitness = sum(fitness)
        probs = [f/total_fitness for f in fitness]
        cum_probs = np.cumsum(probs)
        selected = []
        for _ in range(self.pop_size):
            r = random.random()
            for i, cp in enumerate(cum_probs):
                if r <= cp:
                    selected.append(self.population[i])
                    break
        return selected
    
    def crossover_ox(self, parent1, parent2):
        size = len(parent1)
        cx1 = random.randint(0, size - 1)
        cx2 = random.randint(cx1 + 1, size)
        # Initialize children with -1
        child1 = [-1] * size
        child2 = [-1] * size
        # Copy segments
        child1[cx1:cx2] = parent1[cx1:cx2]
        child2[cx1:cx2] = parent2[cx1:cx2]
        # Fill remaining positions for child1 using parent2
        ptr = cx2
        for city in parent2[cx2:] + parent2[:cx2]:
            if city not in child1[cx1:cx2]:
                while child1[ptr % size] != -1:
                    ptr += 1
                child1[ptr % size] = city
                ptr += 1
        # Fill remaining positions for child2 using parent1
        ptr = cx2
        for city in parent1[cx2:] + parent1[:cx2]:
            if city not in child2[cx1:cx2]:
                while child2[ptr % size] != -1:
                    ptr += 1
                child2[ptr % size] = city
                ptr += 1
        return child1, child2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            start = random.randint(0, len(individual) - 2)
            end = random.randint(start + 1, len(individual) - 1)
            individual[start:end] = reversed(individual[start:end])
        return individual

    def iterate(self, num_iterations):
        for _ in range(num_iterations):
            print("iterating")
            # Calculate fitness
            fitness = [1 / (self.calculate_total_distance(ind) + 1e-9) for ind in self.population]
            # Select parents
            parents = self.select_parents(fitness)
            # Generate offspring
            newpopulation = []
            for i in range(0, self.pop_size, 2):
                p1 = parents[i]
                p2 = parents[(i + 1) % self.pop_size]
                c1, c2 = self.crossover_ox(p1, p2)
                newpopulation.extend([self.mutate(c1), self.mutate(c2)])
                print(1)
            # Elitism: preserve the best individual
            best_idx = max(range(self.pop_size), key=lambda i: fitness[i])
            best_distance = self.calculate_total_distance(self.population[best_idx])
            self.best_dist.append(best_distance)
            print(f"Iteration {_ + 1}/{num_iterations} - Best Distance: {best_distance:.12f}")
            elite = self.population[best_idx]
            # Replace worst in new population
            new_fitness = [1 / (self.calculate_total_distance(ind)) for ind in newpopulation]
            worst_idx = min(range(len(new_fitness)), key=lambda i: new_fitness[i])
            newpopulation[worst_idx] = elite
            self.population = newpopulation[:self.pop_size]
        # Find best solution
        best_ind = min(self.population, key=lambda x: self.calculate_total_distance(x))
        total_distance = self.calculate_total_distance(best_ind)
        self.best_dist.append(total_distance)
        return [self.city_ids[i] for i in best_ind], total_distance
    
    def plot_path(self, path):
        plt.figure(figsize=(10, 6))
        
        # 获取坐标并按路径顺序排列
        ordered_coords = self.coords[path]
        
        # 绘制闭合路径（连接首尾）
        closed_path = np.vstack([ordered_coords, ordered_coords[0]])
        
        # 绘制连线
        plt.plot(closed_path[:, 0], closed_path[:, 1], 
                'b-', linewidth=1, alpha=0.6, label='Path')
        
        # 绘制城市节点
        plt.scatter(self.coords[:, 0], self.coords[:, 1],
                   c='red', s=50, edgecolors='black', label='Cities')
        
        # 标注城市ID
        for i, (x, y) in enumerate(self.coords):
            plt.text(x, y, str(self.city_ids[i]),
                    fontsize=8, ha='right', va='bottom')
        
        # 计算总距离用于标题
        total_dist = self.calculate_total_distance(path)
        
        plt.title(f'TSP Optimal Path (Length: {total_dist:.2f})')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.axis('equal')  # 保持坐标轴比例一致
        plt.tight_layout()
        plt.show()

    def plot_optimization_process(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_dist, 'b-', linewidth=1.5)
        plt.title('Optimization Process')
        plt.xlabel('Generation')
        plt.ylabel('Best Path Length')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

ga1 = GeneticAlgTSP("wi29.tsp")

best_path1 = ga1.iterate(100); 
print("最优路径：", best_path1)

path_indices = [ga1.city_ids.index(city_id) for city_id in best_path1[0]]  
ga1.plot_path(path_indices)
ga1.plot_optimization_process()

