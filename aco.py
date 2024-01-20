import numpy as np
import matplotlib.pyplot as plt


class ACO:

    def __init__(self, n_iterations, n_ants, points=np.random.rand(10, 2), alpha=1, beta=1, evaporation_rate=0.5, Q=None):
        self.points = points  # Точки в двовимірному просторі, які оброблює алгоритм
        self.n_points = len(points)  # Довжна масиву отриманих точок
        self.n_iterations = n_iterations  # Кількість ітерацій роботи алгоритму
        self.n_ants = n_ants  # Кількість комах, які будуть шукати наближений до оптимальлного шлях
        self.pheromones = np.ones((self.n_points, self.n_points))  # Масив значень ферамонів на графу
        self.alpha = alpha  # Параметр для керування впливу ферамонів на вибір шляху
        self.beta = beta  # Параметр для керування впливу вартості ребер на вибір шляху
        self.evaporation_rate = evaporation_rate  # Швидкість випаровування ферамонів
        # Константне значення, яке впливає на кількість ферамону, залишеного мурахою на шляху
        if Q is None:
            self.Q = self.n_points
        else:
            self.Q = Q

        self.best_path = None
        self.best_path_length = np.inf

    def run(self):
        for iteration in range(self.n_iterations):
            # На кожній ітерації оголощуємо список шляхів й довжин пройдений мурахами
            paths = []
            path_lengths = []

            for ant in range(self.n_ants):
                # Для кожної мурахи оголошуємо відвідані точки, пройдений шлях, довжину пройденого шляху
                visited = [False] * self.n_points
                current_point = np.random.randint(self.n_points)
                visited[current_point] = True
                path = [current_point]
                path_length = 0

                while False in visited:
                    # Для кожної невідвідної мурахою точки перераховуємо ймовірності
                    unvisited = np.where(np.logical_not(visited))[0]
                    probabilities = np.zeros(len(unvisited))

                    # Розрахунок ймовірності переходу мурахи з поточної точки до наступної невідвіданої
                    for index, unvisited_point in enumerate(unvisited):
                        probabilities[index] = self.pheromones[current_point][unvisited_point] ** self.alpha * (
                                1 / self.euclidian_distance(self.points[current_point],
                                                            self.points[unvisited_point]) ** self.beta)
                    probabilities /= np.sum(probabilities)

                    # Обираємо наступну точку згідно з отриманими ймовірностями і перераховуємо значення
                    next_point = np.random.choice(unvisited, p=probabilities)
                    path.append(next_point)
                    path_length += self.euclidian_distance(self.points[current_point], self.points[
                        next_point])  # Вираховуємо і збільшуємо відстань від поточної точки до наступної
                    visited[next_point] = True
                    current_point = next_point

                # Замикаємо пройдену відстань розраховуя відстань між останньою і першою відвіданою точкою
                path_length += self.euclidian_distance(self.points[path[0]], self.points[path[-1]])

                # Додаємо шлях і довжину шляху мурахи до пройдених мурахами шляхів і довжин
                paths.append(path)
                path_lengths.append(path_length)

                # Перевірка чи є довжина шляху мурахи найліпшою
                if path_length < self.best_path_length:
                    self.best_path_length = path_length
                    self.best_path = path

            # Оновлення феромонів на кожній ітерації
            self.update_pheromones(paths, path_lengths)

    def update_pheromones(self, paths, path_lengths):
        self.pheromones *= self.evaporation_rate  # Для феромонів застосовуємо швидкість випаровування

        # Оновлємо феромони відповідно до того скілки їх витратили мурахи
        for path, path_length in zip(paths, path_lengths):
            for i in range(self.n_points - 1):
                self.pheromones[path[i], path[i + 1]] += self.Q / path_length
            self.pheromones[path[-1], path[0]] += self.Q / path_length

    def show_results(self):
        if self.best_path is None:
            print('Call .run() method to get the results!')
            return

        plt.scatter(self.points[:, 0], self.points[:, 1], c='orange', marker='o')
        for index in range(len(self.best_path) - 1):
            plt.plot((self.points[self.best_path[index]][0], self.points[self.best_path[index + 1]][0]),
                     (self.points[self.best_path[index]][1], self.points[self.best_path[index + 1]][1]), color='gray',
                     linestyle='-', linewidth=2)

        plt.plot((self.points[self.best_path[-1]][0], self.points[self.best_path[0]][0]),
                 (self.points[self.best_path[-1]][1], self.points[self.best_path[0]][1]), color='gray', linestyle='-',
                 linewidth=2)

        plt.grid()
        plt.title(f'Iters:{self.n_iterations} | Ants:{self.n_ants} | Path length:{self.best_path_length}')
        plt.show()

        print('Best path found:', self.best_path)
        print('Best path length found', self.best_path_length)

    @staticmethod
    def euclidian_distance(point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 1 / 2


example_points = np.array([
    [47, 53],
    [25, 75],
    [88, 57],
    [77, 82],
    [19, 38],
    [21, 88],
    [95, 45],
    [42, 50],
    [81, 21],
    [68, 92]
])

aco = ACO(
    n_iterations=5,
    n_ants=5,
    points=example_points,
    beta=5
)
aco.run()
aco.show_results()
