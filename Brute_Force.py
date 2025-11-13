import random
import csv

project = "."

def generate_cities_coords(cities_nbr: int, seed: int = None) -> dict:
    if seed is not None:
        random.seed(seed)
    cities = {}
    for city in range(cities_nbr):
        x = round(random.randint(0,100) / 100, 2)
        y = round(random.randint(0,100) / 100, 2)
        cities[city] = (x, y)
    return cities 

def calculate_distance(first_city: tuple, second_city: tuple) -> float:
    distance = round((((first_city[0] - second_city[0]) * (first_city[0] - second_city[0])) + ((first_city[1] - second_city[1]) * (first_city[1] - second_city[1]))) ** 0.5, 2)
    return distance

def euclidian_matrice (cities_with_coords: dict) -> list:
    matrice = []
    for i in cities_with_coords:
        matrice.append([])
        for j in cities_with_coords:
            matrice[i].append(calculate_distance(cities_with_coords[i],cities_with_coords[j]))
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
        return euclidian_matrice(cities_nbr, seed)
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
        self.smallest = None
        self.best = None
        if self.n == 0:
            print("La matrice de ville est vide")
            return (0.0,[])
        if self.n == 1:
            print("il n'y a qu'une ville dans la matrice fourni")
            return (0.0,[])

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        # construie la liste des villes à visiter (sans predndre en compte la ville de départ 0
        self.search(0, remaining_cities, 0.0, [])

        # print("meilleur tour trouvé:", self.best, "| distance:", round(self.smallest, 6))
        return (self.best, self.smallest)


    def lower_bound(self, current, remaining_cities): 
        if len(remaining_cities) == 0:
            return self.matrice[current][0]
        
        bound = 0
        for i in range(0,self.n):
            lower = None
            second_lower = None 
            for j in remaining_cities:
                if  i != j and i in remaining_cities and j in remaining_cities:
                    if lower is None or self.matrice[i][j] < lower :
                        second_lower = lower
                        lower = self.matrice[i][j]
            
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
            new_path =covered_path[:]
            new_path.append(next_city)
            if self.smallest is None or len(remaining_cities) <= 2 or self.lower_bound( next_city, new_remaining_cities) + coverd_cities + next_step_dist < self.smallest:
                self.search_branch_and_bound(next_city, new_remaining_cities, coverd_cities + next_step_dist, new_path)
            elif i == 0:
                self.search(next_city, new_remaining_cities, coverd_cities + next_step_dist, new_path)
            i += 1


    def branch_and_bound(self): 
        self.smallest = None
        self.best = None
        if self.n == 0:
            print("La matrice de ville est vide")
            return (0.0,[])
        if self.n == 1:
            print("il n'y a qu'une ville dans la matrice fourni")
            return (0.0,[])

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        self.search_branch_and_bound(0, remaining_cities, 0.0, [])

        # print("meilleur tour trouvé:", self.best, "| distance:", round(self.smallest, 6))
        return (self.best, self.smallest)


    def nearest_neighbourg(self):
    
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
                if nearest_dist is None or nearest_dist < self.matrice[current][k] and k not in path:
                    nearest_dist = self.matrice[current][k]
                    nearest = k

            total_dist += nearest_dist
            current = nearest
            path.append(current)

            
        # print("trajet suivi:", path + [0], "| distance parcouru:", round(total_dist, 2))
        return (path + [0], total_dist)

    
    def cheapest_insertion(self):
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

        return (tour, total_dist)

    



tsp = Tsp_solver(create_matrice(14, seed=89898989, euclidian=False))


# print("matrice",tsp.matrice)
# print("Brute force :", tsp.bruteforce())
# print("B&B :", tsp.branch_and_bound())
# print("NEAREST :", tsp.nearest_neighbourg())
print("CHEAPEST :", tsp.cheapest_insertion())
# print("TLS CHEAPEST :", tsp.tls_cheapest_neighbor(0, 1))
# B&B  1.87s user 0.03s system 97% cpu 1.945 total
#BrteForce 1.87s user 0.02s system 99% cpu 1.905 total
