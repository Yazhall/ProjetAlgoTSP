import random
import math
import time
import csv
from pathlib import Path

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





class TspInstanceGenerator:
    """
    Classe utilitaire pour générer et manipuler des instances de matrices pour le TSP
    """
    def __init__(self, project_path: str = "."):
        self.project_path = project_path

    def generate_cities_coords(self, cities_nbr: int, seed: int = None) -> list:
        """Génère une liste de coordonnées pour chaque ville."""
        if seed is not None:
            random.seed(seed)
        cities = []
        for _ in range(cities_nbr):
            x = round(random.randint(0,100) / 100, 2)
            y = round(random.randint(0,100) / 100, 2)
            cities.append((x, y))
        return cities

    def calculate_distance(self, first_city: tuple, second_city: tuple) -> float:
        """Calcule la distance euclidienne entre deux villes."""
        distance = (((first_city[0] - second_city[0]) ** 2) + ((first_city[1] - second_city[1]) ** 2)) ** 0.5
        return distance

    def euclidian_matrix(self, cities: list) -> list:
        """Crée une matrice de distances euclidiennes entre les villes."""
        matrix = []
        for i in range(len(cities)):
            matrix.append([])
            for j in range(len(cities)):
                matrix[i].append(math.ceil(self.calculate_distance(cities[i],cities[j]) * 100 ) / 100)
        return matrix

    def random_matrix(self, cities_nbr: int, seed: int = None, complete_graph: bool = True) -> list:
        """Crée une matrice de distances aléatoires."""
        if seed is not None:
            random.seed(seed)
        matrix = []
        for i in range(cities_nbr):
            matrix.append([])
            for j in range(cities_nbr):
                if not complete_graph and random.random() < 0.2:
                    matrix[i].append(None)
                elif j == i:
                    matrix[i].append(0)
                elif j < i:
                    matrix[i].append(matrix[j][i])
                else:
                    matrix[i].append(random.randrange(0, 100))
        return matrix

    def create_matrix(self, cities_nbr: int, seed: int = None, euclidian: bool = True, complete_graph: bool = True) -> list:
        """Crée une matrice de distances selon le mode choisi."""
        if euclidian:
            return self.euclidian_matrix(self.generate_cities_coords(cities_nbr, seed))
        else:
            return self.random_matrix(cities_nbr, seed, complete_graph)

    def matrix_to_adjacency_list(self, matrix: list) -> dict:
        """Transforme une matrice en liste d'adjacence."""
        adjacency_list = {}
        for i in range(len(matrix)):
            adjacency_list[i+1] = []
            for j in range(len(matrix[i])):
                if matrix[i][j] is not None:
                    adjacency_list[i+1].append(j+1)
        return adjacency_list

    def write_instance(self, instance: list, instance_name: str) -> None:
        """Écrit une instance dans un fichier CSV."""
        with open(f"{self.project_path}/instances/{instance_name}.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='|')
            writer.writerows(instance)

    def read_instance(self, instance_name: str) -> list:
        """Lit une instance depuis un fichier CSV."""
        instance = []
        with open(f"{self.project_path}/instances/{instance_name}.csv", 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            for row in reader:
                instance.append(row)
        return instance
