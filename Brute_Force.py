import random
import csv
import math
import time

project = "."

def generate_cities_coords(cities_nbr: int, seed: int = None) -> list:
    if seed is not None:
        random.seed(seed)
    cities = []
    for _ in range(cities_nbr):
        x = round(random.randint(0,100) / 100, 2)
        y = round(random.randint(0,100) / 100, 2)
        cities.append((x, y))
    return cities 

def calculate_distance(first_city: tuple, second_city: tuple) -> float:
    distance = (((first_city[0] - second_city[0]) ** 2) + ((first_city[1] - second_city[1]) ** 2)) ** 0.5
    return distance

def euclidian_matrice(cities: list) -> list:
    matrice = []
    for i in range(len(cities)):
        matrice.append([])
        for j in range(len(cities)):
            # We ceil here instead of round or floor to respect triangle inequality
            matrice[i].append(math.ceil(calculate_distance(cities[i],cities[j]) * 100 ) / 100)
    return matrice

def random_matrice(cities_nbr: int, seed: int = None, complete_graph: bool = True) -> list:
    if seed is not None:
        random.seed(seed)
    random_matrice = []
    for i in range(cities_nbr):
        random_matrice.append([])
        for j in range(cities_nbr):
            if not complete_graph and random.random() < 0.2:
                random_matrice[i].append(None)
            elif j == i:
                random_matrice[i].append(0)
            elif j < i:
                random_matrice[i].append(random_matrice[j][i])
            else:
                random_matrice[i].append(random.randrange(0, 100))
    return random_matrice

def create_matrice(cities_nbr: int, seed: int = None, euclidian: bool = True, complete_graph: bool = True) -> list:
    if euclidian:
        return euclidian_matrice(generate_cities_coords(cities_nbr, seed))
    else:
        return random_matrice(cities_nbr, seed, complete_graph)

def matrice2listeadjacente(matrice: list) -> dict:
    liste_adjacente = {}
    for i in range(len(matrice)):
        liste_adjacente[i+1] = []
        for j in range(len(matrice[i])):
            if matrice[i][j] is not None:
                liste_adjacente[i+1].append(j+1)
    return liste_adjacente

def write_instance( instance: list, instance_name: str,) -> None:
    with open(f"{project}/instances/{instance_name}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerows(instance)

def read_instance(instance_name: str):
    instance = []
    with open(f"{project}/instances/{instance_name}.csv", 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            instance.append(row)
    return instance

# test_bruteforce = [[0.0, 0.1, 0.79, 0.52, 0.39], [0.1, 0.0, 0.34, 1.0, 0.71], [0.79, 0.34, 0.0, 0.66, 0.41], [0.52, 1.0, 0.66, 0.0, 0.5], [0.39, 0.71, 0.41, 0.5, 0.0]]
# test_bruteforce = [[0.0, 3.1, 0.79, 0.52, 0.39], [3.1, 0.0, 0.34, 1.0, 71], [0.79, 0.34, 0.0, 0.66, 0.41], [0.52, 1.0, 0.66, 0.0, 0.5], [0.39, 71, 0.41, 0.5, 0.0]]
# test_bruteforce = [
#         #  A       B        C      D      E
#     #  A [0.0,    1.1,    0.79,   0.52,   0.39], 
#     #  B [1.1,    0.0,    0.34,   1.0,    0.71], 
#     #  C [0.79,   0.34,   0.0,    0.66,   0.41], 
#     #  D [0.52,   1.0,    0.66,   0.0,    0.5], 
#     #  E [0.39,   0.71,   0.41,   0.5,    0.0]
#      ]


class Tsp_solver:

    def __init__(self, matrice):
        self.matrice = matrice
        self.n = len(matrice)  # nombre de ville ( n villes)
        self.smallest = None
        self.best = None

    def search(self, current, remaining_cities, coverd_cities,covered_path):
        if len(remaining_cities) == 0:
            last_to_first_dist = self.matrice[current][0] 
            total_dist = coverd_cities + last_to_first_dist
            full_path = [0] +covered_path + [0]


            if self.smallest is None or total_dist < self.smallest:
                self.smallest = total_dist
                self.best = full_path
            # print("trajet suivi:", full_path, "| distance parcouru:", round(total_dist, 2))
            return

        # if self.smallest is not None and self.smallest < coverd_cities:
        #     return

        i = 0
        while i < len(remaining_cities):
            next_city = remaining_cities[i]
            next_step_dist = self.matrice[current][next_city]
            new_remaining_cities = remaining_cities[:i] + remaining_cities[i+1:] 
            new_path =covered_path[:]
            new_path.append(next_city)
            self.search(next_city, new_remaining_cities, coverd_cities + next_step_dist, new_path)
            i += 1


    def bruteforce(self) -> tuple[float, list[int]]: 
        # start_time = time.perf_counter()
        start_time = time.perf_counter()


        self.smallest = None
        self.best = None
        if self.n == 0:
            # print("La matrice de ville est vide")
            return (0.0,[])
        if self.n == 1:
            # print("il n'y a qu'une ville dans la matrice fourni")
            return (0.0,[])

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        # construie la liste des villes à visiter (sans predndre en compte la ville de départ 0
        self.search(0, remaining_cities, 0.0, [])

        end_time = time.perf_counter()
        print(f"Brute force comput time: {end_time - start_time:.6f} seconds")

        # print("meilleur tour trouvé:", self.best, "| distance:", round(self.smallest, 6))
        return (self.best, self.smallest)


    def lower_bound(self, current, remaining_cities): 
        if len(remaining_cities) == 0:
            return self.matrice[current][0]
        
        bound = 0
        for i in range(0,self.n):
            lower = None
            second_lower = None 
            for cities in remaining_cities:
                if  i != cities and i in remaining_cities and cities in remaining_cities:
                    if lower is None or self.matrice[i][cities] < lower :
                        second_lower = lower
                        lower = self.matrice[i][cities]
            
            # check si lower et second_lower ne sont pas None avant de les utiliser pour les cas ou il y a des none dans la matrice (non testé)
            if lower is not None and second_lower is not None:
                bound += (lower + second_lower)/2
            elif lower is not None:
                bound += lower
        
        return bound
        

    def search_branch_and_bound(self, current, remaining_cities, coverd_cities,covered_path):

        if len(remaining_cities) == 0:
            last_to_first_dist = self.matrice[current][0] 
            total_dist = coverd_cities + last_to_first_dist
            full_path = [0] +covered_path + [0]


            if self.smallest is None or total_dist < self.smallest:
                self.smallest = total_dist
                self.best = full_path
            # print("trajet suivi:", full_path, "| distance parcouru:", round(total_dist, 2))
            return

        if self.smallest is not None and self.smallest < coverd_cities:
            return

        i = 0
        while i < len(remaining_cities):

            next_city = remaining_cities[i]
            next_step_dist = self.matrice[current][next_city]
            new_remaining_cities = remaining_cities[:i] + remaining_cities[i+1:]
            new_path = covered_path[:]
            new_path.append(next_city)

            if self.smallest is None or len(remaining_cities) <= 2 or self.lower_bound( next_city, new_remaining_cities) + coverd_cities + next_step_dist < self.smallest:
                self.search_branch_and_bound(next_city, new_remaining_cities, coverd_cities + next_step_dist, new_path)
            elif i == 0:
                self.search(next_city, new_remaining_cities, coverd_cities + next_step_dist, new_path)
            i += 1


    def branch_and_bound(self): 
        start_time = time.perf_counter()


        self.smallest = None
        self.best = None
        if self.n == 0:
            # print("La matrice de ville est vide")
            return (0.0,[])
        if self.n == 1:
            # print("il n'y a qu'une ville dans la matrice fourni")
            return (0.0,[])

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        self.search_branch_and_bound(0, remaining_cities, 0.0, [])

        end_time = time.perf_counter()
        print(f"B&B comput time: {end_time - start_time:.6f} seconds")
        # print("meilleur tour trouvé:", self.best, "| distance:", round(self.smallest, 6))
        return (self.best, self.smallest)


    def nearest_neighbourg(self):
        start_time = time.perf_counter()
        nearest = None
        current = 0
        total_dist = 0
        remaining_cities = []
        path = [current]

        i = 1
        while i < self.n:
            remaining_cities.append(i)
            i += 1
        
        while len(path) < self.n:
            nearest_dist = None
            for k in remaining_cities:
                if (nearest_dist is None or nearest_dist >= self.matrice[current][k]) and k not in path:
                    nearest_dist = self.matrice[current][k]
                    nearest = k

            total_dist += nearest_dist
            current = nearest
            path.append(current)
        total_dist += self.matrice[current][0]
        end_time = time.perf_counter()
        print(f"near comput time: {end_time - start_time:.6f} seconds")

        # print("trajet suivi:", path + [0], "| distance parcouru:", round(total_dist, 2))
        return {"path":path + [0],"dist": total_dist}


    
    def cheapest_insertion(self):
        start_time = time.perf_counter()
        if self.n == 0:
            print("La matrice de ville est vide")
            return ([], 0.0)
        if self.n == 1:
            return ([0, 0], 0.0)

        nearest_city = None
        nearest_dist = None

        for city in range(1, self.n):
            distance = self.matrice[0][city]
            if nearest_dist is None or distance < nearest_dist:
                nearest_dist = distance
                nearest_city = city

        if nearest_city is None:
            print("Impossible de trouver une ville atteignable depuis la ville 0")
            return (None,None)

        tour = [0, nearest_city, 0]
        visited = set(tour[:-1])

        while len(visited) < self.n:
            best_city = None
            best_position = None
            best_increase = None

            for city in range(self.n):
                if city not in visited:
                    for idx in range(len(tour) - 1):
                        i = tour[idx]
                        j = tour[idx + 1]
                        dist_ij = self.matrice[i][j]
                        dist_ic = self.matrice[i][city]
                        dist_cj = self.matrice[city][j]

                        if dist_ij is not None and dist_ic is not None and dist_cj is not None:
                            increase = dist_ic + dist_cj - dist_ij
                            if best_increase is None or increase < best_increase:
                                best_increase = increase
                                best_city = city
                                best_position = idx

            tour.insert(best_position + 1, best_city)
            visited.add(best_city)

        total_dist = 0.0
        for idx in range(len(tour) - 1):
            dist = self.matrice[tour[idx]][tour[idx + 1]]
            total_dist += dist

        end_time = time.perf_counter()
        print(f"cheap comput time: {end_time - start_time:.6} seconds")
        return {"path":tour, "total_dist":total_dist}



    def calculate_total_distance(self, path):
        total = 0.0
        for k in range(len(path) - 1):
            total += self.matrice[path[k]][path[k + 1]]
        return total

    def two_opt_solver(self, path):
        start_time = time.perf_counter()
        best_distance = self.calculate_total_distance(path)
        opti = True
        while opti is True:
            opti = False
            for i in range (1,len(path)-2):
                for j in range (i+2,len(path)-1):
                    # if j != i-1 and j != i and j != i+1:
                        a,b = path[i],path[i+1]
                        c,d = path[j],path[j+1]

                        old = self.matrice[a][b] + self.matrice[c][d]
                        # print(old)
                        new = self.matrice[a][c] + self.matrice[b][d]
                        # print(new)
                        if old > new:
        
                            path[i+1:j+1] = path[i+1:j+1][::-1]
                            best_distance += new - old
                            opti = True
        end_time = time.perf_counter()
        print(f"2opt comput time: {end_time - start_time:.6} seconds")
        return (path, best_distance)

tsp = Tsp_solver(create_matrice(9, seed=1))


# input_matrix = [
#         [0.0, 1.03, 0.59, 0.53, 0.42, 0.71, 0.95, 0.12],
#          [1.03, 0.0, 0.66, 0.96, 0.66, 0.43, 0.19, 1.01],
#          [0.59, 0.66, 0.0, 0.88, 0.52, 0.61, 0.69, 0.52],
#          [0.53, 0.96, 0.88, 0.0, 0.38, 0.53, 0.81, 0.62],
#          [0.42, 0.66, 0.52, 0.38, 0.0, 0.29, 0.55, 0.46],
#          [0.71, 0.43, 0.61, 0.53, 0.29, 0.0, 0.28, 0.73],
#          [0.95, 0.19, 0.69, 0.81, 0.55, 0.28, 0.0, 0.96],
#          [0.12, 1.01, 0.52, 0.62, 0.46, 0.73, 0.96, 0.0]
#     ]
# tsp = Tsp_solver(input_matrix)

# print("matrice",tsp.matrice)
# print("B&B :", tsp.branch_and_bound())
# print("Brute force :", tsp.bruteforce())
# print("NEAREST :", tsp.nearest_neighbourg())
# print("CHEAPEST :", tsp.cheapest_insertion())
# print("2opt :", tsp.two_opt_solver(tsp.cheapest_insertion()["path"]))

# B&B  1.87s user 0.03s system 97% cpu 1.945 total
#BrteForce 1.87s user 0.02s system 99% cpu 1.905 total
