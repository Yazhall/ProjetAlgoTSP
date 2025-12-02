import random
import csv
import math
import time
import os
import json

def generate_cities_coords(cities_nbr, seed=None):
    """Generates random coordinates in [0,1]^2 for n cities."""
    if seed is not None:
        random.seed(seed)
    cities = []
    for _ in range(cities_nbr):
        x = round(random.randint(0, 100) / 100, 2)
        y = round(random.randint(0, 100) / 100, 2)
        cities.append((x, y))
    return cities

def calculate_distance(first_city, second_city):
    """return distance between two points."""
    distance = (((first_city[0] - second_city[0]) ** 2) + ((first_city[1] - second_city[1]) ** 2)) ** 0.5
    return distance
    
def euclidian_matrice(cities):
    """Builds the complete Euclidean matrice from a list of coordinates."""
    matrice = []
    for i in range(len(cities)):
        matrice.append([])
        for j in range(len(cities)):
            # We ceil here instead of round or floor to respect triangle inequality
            matrice[i].append(math.ceil(calculate_distance(cities[i], cities[j]) * 100) / 100)
    return matrice

def random_matrice(cities_nbr, seed=None, complete_graph=True):
    """Builds a symmetric weighted random matrice (with or without missing edges)."""
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
                # poids strictement positifs, bornés à 100 inclus
                random_matrice[i].append(random.randint(1, 100))
    return random_matrice

def force_triangle_inequality(matrice):
    """
    Forces the triangle inequality on a given distance matrix using Floyd-Warshall.

    Args:
        matrice (list[list]): The distance matrix to modify.
    Returns:
        list[list]: The modified matrix with the triangle inequality forced.
    """
    n = len(matrice)
    for k in range(n):
        for i in range(n):
            if matrice[i][k] is None:
                continue
            for j in range(n):
                if matrice[k][j] is None:
                    continue
                potential_distance = matrice[i][k] + matrice[k][j]
                if matrice[i][j] is None or potential_distance < matrice[i][j]:
                    matrice[i][j] = potential_distance
                    matrice[j][i] = potential_distance
    return matrice

def create_matrice(cities_nbr, seed=None, euclidian=True, complete_graph=True, force_triangle=False):
    """
    Generates random matrice, with an option to enforce the triangle inequality.
    Note: If 'euclidian' is True, the parameters 'complete_graph' and 'force_triangle' are obsolete.
    Args:
        cities_nbr (int): Number of cities.
        seed (int, optional): Seed for random number generator.
        euclidian (bool, optional): If True, generates Euclidean distances.
        complete_graph (bool, optional): If True, generates a complete graph.
        force_triangle (bool, optional): If True, enforces the triangle inequality.
    Returns:
        list[list]: The generated distance matrix.
    """
    
    if euclidian:
        return euclidian_matrice(generate_cities_coords(cities_nbr, seed))
    else:
        matrice = random_matrice(cities_nbr, seed, complete_graph)
        if force_triangle:
            matrice = force_triangle_inequality(matrice)
        return matrice

def matrice2listeadjacente(matrice):
    """Retourne une liste d'adjacence annotée [(voisin, poids), ...] pour chaque sommet."""
    liste_adjacente = {}
    for i in range(len(matrice)):
        liste_adjacente[i] = []
        for j in range(len(matrice[i])):
            if matrice[i][j] is not None and i != j:
                liste_adjacente[i].append((j, matrice[i][j]))
    return liste_adjacente

def write_instance(instance, instance_name):
    save_path = os.path.join(os.path.dirname(__file__), "saved")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    with open(os.path.join(save_path, f"{instance_name}.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(instance)
    print(f"Exported instance to: {os.path.join(save_path, f'{instance_name}.csv')}")

def read_instance(instance_name):
    save_path = os.path.join(os.path.dirname(__file__), "saved")
    instance = []
    with open(os.path.join(save_path, f"{instance_name}.csv"), 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            instance.append([float(value) for value in row])
    return instance

def write_solution_json(path, cost, filename="solution.json"):
    """Saves a solution (path + cost) in JSON format in the saved/ directory."""
    save_path = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_path, exist_ok=True)
    payload = {"path": path, "cost": cost}
    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    print(f"Solution sauvegardée dans saved/{filename}")


class Tsp_solver:

    def __init__(self, matrice):
        self.matrice = matrice
        self.n = len(matrice)  # nombre de ville ( n villes)
        self.smallest = None
        self.best = None
        self.nodes_bruteforce = 0
        self.nodes_bb = 0
    
    def search(self, current, remaining_cities, covered_cities, covered_path, verbose=False):
        """
        Recursive search for brute force TSP solution.

        Args:
            current (int): The current city index.
            remaining_cities (list): List of indices of cities not yet visited.
            covered_cities (float): The total distance covered.
            covered_path (list): The path of cities visited.
            verbose (bool, optional): Active mode verbose, prints details of the computation.

        Return:
            None: Updates the smallest distance and best path attributes of the instance.
        """
        # if verbose:
        self.nodes_bruteforce += 1
        if len(remaining_cities) == 0:
            last_to_first_dist = self.matrice[current][0]
            if last_to_first_dist is None:
                return
            total_dist = covered_cities + last_to_first_dist
            full_path = [0] + covered_path + [0]

            if verbose:
                print("trajet suivi:", full_path, "| distance parcourue:", round(total_dist, 2))

            if self.smallest is None or total_dist < self.smallest:
                self.smallest = total_dist
                self.best = full_path
            return

        # if self.smallest is not None and self.smallest < covered_cities:
        #     return

        i = 0
        while i < len(remaining_cities):
            next_city = remaining_cities[i]
            next_step_dist = self.matrice[current][next_city]
            if next_step_dist is None:
                i += 1
                continue
            new_remaining_cities = remaining_cities[:i] + remaining_cities[i+1:] 
            new_path = covered_path[:]
            new_path.append(next_city)
            self.search(next_city, new_remaining_cities, covered_cities + next_step_dist, new_path, verbose)
            i += 1

    def bruteforce(self, verbose=False):
        """
        Solve the TSP using brute force.

        Args:
            verbose (bool, optional): Active mode verbose, prints details of the computation.

        Returns:
            dict: {"path": list, "total_dist": float or None}.
        """
        self.nodes_bruteforce = 0
        # if verbose:
        start_time = time.perf_counter()
        self.smallest = None
        self.best = None
        if self.n == 0:
            return {"path": [], "total_dist": 0.0}
        if self.n == 1:
            return {"path": [0, 0], "total_dist": 0.0}

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        # Construire la liste des villes à visiter (sans prendre en compte la ville de départ 0)
        self.search(0, remaining_cities, 0.0, [], verbose)

        end_time = time.perf_counter()
        print(f"Brute force nodes explored: {self.nodes_bruteforce}")
        if verbose:
            print(f"Brute force comput time: {end_time - start_time:.6f} seconds")
            print("meilleur tour trouvé:", self.best, "| distance:", round(self.smallest, 2))
        return {"path": self.best, "total_dist": round(self.smallest, 2)}

    def lower_bound(self, current, remaining_cities):
        """
        Bound: for each node (current + remaining), add the sum of its two lightest incident
        edges in the entire graph, divided by 2. This remains admissible même en non-euclidien.
        Args:
            current (int): The current city index.
            remaining_cities (list): List of indices of cities not yet visited.
        Returns:
            float: The lower bound estimate.
        """
        nodes = [current] + remaining_cities
        n = self.n

        bound = 0
        for i in nodes:
            lower = None
            second_lower = None
            for j in range(n):
                if i == j:
                    continue
                dist = self.matrice[i][j]
                if dist is None:
                    continue
                if lower is None or dist < lower:
                    second_lower = lower
                    lower = dist
                elif second_lower is None or dist < second_lower:
                    second_lower = dist

            if lower is not None and second_lower is not None:
                bound += (lower + second_lower) / 2
            elif lower is not None:
                bound += lower

        return bound
        
    def search_branch_and_bound(self, current, remaining_cities, covered_cities, covered_path, verbose=False):
        """
        Solve the TSP using the branch and bound

        Args:
            current (int): The current city index.
            remaining_cities (list): List of indices of cities not yet visited.
            covered_cities (float): The total distance covered.
            covered_path (list): The path of cities visited.
            verbose (bool, optional): Active mode verbose, prints details of the computation.

        Returns:
            None: Updates the smallest distance and best path of the instance.
        """
        self.nodes_bb += 1
        # if verbose:
        if len(remaining_cities) == 0:
            last_to_first_dist = self.matrice[current][0]
            if last_to_first_dist is None:
                return
            total_dist = covered_cities + last_to_first_dist
            full_path = [0] + covered_path + [0]

            if self.smallest is None or total_dist < self.smallest:
                self.smallest = total_dist
                self.best = full_path
            return

        if self.smallest is not None and self.smallest < covered_cities:
            if verbose:
                print(f"Branch cut: current path {covered_path}, covered distance {covered_cities}")
            return

        i = 0
        while i < len(remaining_cities):
            next_city = remaining_cities[i]
            next_step_dist = self.matrice[current][next_city]
            if next_step_dist is None:
                i += 1
                continue
            new_remaining_cities = remaining_cities[:i] + remaining_cities[i+1:]
            new_path = covered_path[:]
            new_path.append(next_city)

            if self.smallest is None or len(remaining_cities) <= 2 or self.lower_bound(next_city, new_remaining_cities) + covered_cities + next_step_dist < self.smallest:
                self.search_branch_and_bound(next_city, new_remaining_cities, covered_cities + next_step_dist, new_path, verbose)
            else:
                if verbose:
                    print(f"Branch cut: current path {new_path}")
            i += 1

    def branch_and_bound(self, verbose=False):
        """
        Solves the TSP using the branch and bound method.

        Args:
            verbose (bool, optional): Active mode verbose, prints details of the computation.

        Returns:
            dict: {"path": list, "total_dist": float or None}.
        """
        start_time = time.perf_counter()
        self.nodes_bb = 0

        self.smallest = None
        self.best = None
        if self.n == 0:
            return {"path": [], "total_dist": 0.0}
        if self.n == 1:
            return {"path": [0, 0], "total_dist": 0.0}

        remaining_cities = []
        j = 1
        while j < self.n:
            remaining_cities.append(j)
            j += 1

        self.search_branch_and_bound(0, remaining_cities, 0.0, [], verbose)

        end_time = time.perf_counter()
        if verbose:
            print(f"B&B comput time: {end_time - start_time:.6f} seconds")
        print(f"B&B nodes explored: {self.nodes_bb}")
        return {"path": self.best, "total_dist": round(self.smallest, 2)}

    def nearest_neighbourg(self, verbose=False):
        """
        Solve the TSP using the nearest neighbor heuristic (saute les arêtes manquantes).
        Args:
            verbose (bool, optional): If True, prints details of the computation.

        Returns:
            dict: {"path": list, "total_dist": float or None}
        """
        iterations = 0
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
            iterations += 1
            nearest_dist = None
            for k in remaining_cities:
                w = self.matrice[current][k]
                if w is None:
                    continue
                if (nearest_dist is None or nearest_dist >= w) and k not in path:
                    nearest_dist = w
                    nearest = k

            if nearest_dist is None:
                if verbose:
                    print("Nearest neighbor: graphe incomplet, aucune arête disponible.")
                return {"path": [], "total_dist": None}

            total_dist += nearest_dist
            current = nearest
            path.append(current)
            if verbose:
                print(f"Visited city: {current}, Total distance: {total_dist}")

        if self.matrice[current][0] is None:
            if verbose:
                print("Nearest neighbor: impossible de revenir à 0.")
            return {"path": [], "total_dist": None}

        total_dist += self.matrice[current][0]
        end_time = time.perf_counter()
        if verbose:
            print(f"Nearest neighbor computation time: {end_time - start_time:.6f} seconds")
        print("nearest",iterations)
        return {"path": path + [0], "total_dist": round(total_dist, 2)}

    def cheapest_insertion(self, verbose=False):
        """
        Solve the TSP using the cheapest insertion heuristic.

        Args:
            verbose (bool, optional): If True, prints details of the computation.

        Returns:
            dict: A dictionary containing the path and the total distance.
        """
        iteration = 0
        start_time = time.perf_counter()
        if self.n == 0:
            print("The city matrix is empty")
            return {"path": [], "total_dist": 0.0}
        if self.n == 1:
            return {"path": [0, 0], "total_dist": 0.0}

        nearest_city = None
        nearest_dist = None

        for city in range(1, self.n):
            distance = self.matrice[0][city]
            if distance is None:
                continue
            if nearest_dist is None or distance < nearest_dist:
                nearest_dist = distance
                nearest_city = city

        if nearest_city is None:
            print("Unable to find a reachable city from city 0")
            return {"path": None, "total_dist": None}

        tour = [0, nearest_city, 0]
        visited = set(tour[:-1])

        while len(visited) < self.n:
            best_city = None
            best_position = None
            best_increase = None
            iteration += 1
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

            if best_city is None:
                print("Cheapest insertion: aucune insertion faisable (graph incomplet).")
                return {"path": None, "total_dist": None}

            tour.insert(best_position + 1, best_city)
            visited.add(best_city)
            if verbose:
                print(f"Inserted city: {best_city} at position {best_position + 1}, Current tour: {tour}")

        total_dist = 0.0
        for idx in range(len(tour) - 1):
            dist = self.matrice[tour[idx]][tour[idx + 1]]
            if dist is None:
                print("Cheapest insertion: arête manquante dans le tour final.")
                return {"path": tour, "total_dist": None}
            total_dist += dist

        end_time = time.perf_counter()
        if verbose:
            print(f"Cheapest insertion computation time: {end_time - start_time:.6f} seconds")
        print("cheapest",iteration)
        return {"path": tour, "total_dist": round(total_dist, 2)}

    def calculate_total_distance(self, path):
        """Calculates the length of a path; returns inf if an edge is missing."""
        total = 0.0
        for k in range(len(path) - 1):
            dist = self.matrice[path[k]][path[k + 1]]
            if dist is None:
                return math.inf
            total += dist
        return total

    def two_opt_solver(self, path, verbose=False, max_iterations=10000):
        """
        Improve a given TSP path using the 2-opt algorithm.
        Args:
            path (list): Initial TSP path to improve.
            verbose (bool, optional): If True, prints details of the computation.
            max_iterations (int, optional): Maximum number of iterations to perform.
        Returns:
            dict: {"path": list, "total_dist": float or None}
        """

        start_time = time.perf_counter()
        best_distance = self.calculate_total_distance(path)
        initial_dist = best_distance
        opti = True
        iterations = 0
        while opti is True and iterations < max_iterations:
            iterations += 1
            opti = False
            for i in range(1, len(path) - 2):
                for j in range(i + 2, len(path) - 1):
                    a, b = path[i], path[i + 1]
                    c, d = path[j], path[j + 1]

                    old_ab = self.matrice[a][b]
                    old_cd = self.matrice[c][d]
                    new_ac = self.matrice[a][c]
                    new_bd = self.matrice[b][d]
                    if None in (old_ab, old_cd, new_ac, new_bd):
                        continue
                    old = old_ab + old_cd
                    new = new_ac + new_bd
                    if old > new:
                        path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                        best_distance += new - old
                        opti = True
                if opti:
                    break
        end_time = time.perf_counter()
        # if verbose:
        gain = initial_dist - best_distance
        rel = (gain / initial_dist * 100) if initial_dist != 0 else 0
        print(f"2opt initial: {initial_dist:.4f}, final: {best_distance:.4f}, gain: {gain:.4f} ({rel:.2f}%), iters: {iterations}")
        print(f"2opt comput time: {end_time - start_time:.6f} seconds")
        return {"path": path, "total_dist": best_distance}


tsp = Tsp_solver(create_matrice(16, seed=1, euclidian=False, complete_graph=True, force_triangle=False))



# eucli_complet = create_matrice(10, seed=1, euclidian=True, complete_graph=True, force_triangle=False)
# non_eucli_complet = create_matrice(10, None, False, True, True)
# non_eucli_non_complet = create_matrice(10, None, False, False)
# eucli_non_complet = create_matrice(10, None, True)

# write_instance(tsp.matrice,"test non complet")
# write_instance(non_eucli_complet,"non_eucli_complet")
# write_instance(eucli_non_complet,"eucli_non_complet")
# write_instance(non_eucli_non_complet,"non_eucli_non_complet")

# print("liste eucli_complet :", matrice2listeadjacente(eucli_complet))
# print("liste non_eucli_complet :", matrice2listeadjacente(non_eucli_complet))
# print("liste eucli_non_complet :", matrice2listeadjacente(eucli_non_complet))
# print("liste non_eucli_non_complet :", matrice2listeadjacente(non_eucli_non_complet))


# print("matrice",tsp.matrice)
# print("Brute force :", tsp.bruteforce())
# print("B&B :", tsp.branch_and_bound())
# print("NEAREST :", tsp.nearest_neighbourg())
# print("CHEAPEST :", tsp.cheapest_insertion())
# print("2opt :", tsp.two_opt_solver(tsp.nearest_neighbourg()["path"]))
# print("2opt :", tsp.two_opt_solver(tsp.cheapest_insertion()["path"]))

# print("Nodes explored Bruteforce:", tsp.nodes_bruteforce)
# print("Nodes explored B&B:", tsp.nodes_bb)


# --- Script de test des méthodes pour différentes tailles ---
def benchmark_sizes(repetitions=1, methods=None):
    """
    Bench des méthodes sur différentes tailles, avec moyenne sur plusieurs répétitions.
    Args:
        repetitions (int): nombre de répétitions pour la moyenne des temps.
        methods (list[str] | None): sous-ensemble de méthodes à tester. None -> toutes.
    """
    sizes = [8, 9, 10, 11, 20, 30, 40, 50, 60]
    all_methods = ["bruteforce", "b&b", "nearest", "cheapest", "twoopt_nearest", "twoopt_cheapest"]
    methods = methods or all_methods
    results = []
    for n in sizes:
        for method in methods:
            if method in ("bruteforce", "b&b") and n > 11:
                results.append({"n": n, "method": method, "status": "skipped (>11)"})
                continue
            total_time = 0.0
            last_res = None
            for _ in range(repetitions):
                matrice = create_matrice(n, seed=1, euclidian=False, complete_graph=True, force_triangle=False)
                solver = Tsp_solver(matrice)
                start = time.perf_counter()
                if method == "bruteforce":
                    last_res = solver.bruteforce(verbose=False)
                elif method == "b&b":
                    last_res = solver.branch_and_bound(verbose=False)
                elif method == "nearest":
                    last_res = solver.nearest_neighbourg(verbose=False)
                elif method == "cheapest":
                    last_res = solver.cheapest_insertion(verbose=False)
                elif method == "twoopt_nearest":
                    base = solver.nearest_neighbourg(verbose=False)
                    if not base or not base.get("path"):
                        last_res = base
                    else:
                        last_res = solver.two_opt_solver(base["path"], verbose=False, max_iterations=1000)
                elif method == "twoopt_cheapest":
                    base = solver.cheapest_insertion(verbose=False)
                    if not base or not base.get("path"):
                        last_res = base
                    else:
                        last_res = solver.two_opt_solver(base["path"], verbose=False, max_iterations=1000)
                else:
                    last_res = None
                total_time += time.perf_counter() - start
            if last_res is None:
                results.append({"n": n, "method": method, "status": "failed"})
            else:
                avg_time = total_time / repetitions
                results.append({
                    "n": n,
                    "method": method,
                    "cost": last_res.get("total_dist"),
                    "path_len": len(last_res.get("path") or []),
                    "time": avg_time,
                    "repetitions": repetitions,
                })
    # Affichage simple
    for r in results:
        if "status" in r:
            print(f"n={r['n']:>2} | {r['method']:>15} | {r['status']}")
        else:
            print(f"n={r['n']:>2} | {r['method']:>15} | coût= {round(r['cost'],2)} | len= {r['path_len']} | t_moy= {r['time']*10**6:.0f} µs sur {r['repetitions']} runs")
    return results

# Décommenter pour lancer rapidement le benchmark
# benchmark_sizes(20,["bruteforce","b&b","nearest","cheapest","twoopt_nearest","twoopt_cheapest"])
