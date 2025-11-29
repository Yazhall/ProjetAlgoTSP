import random
import math
import time
import csv
from pathlib import Path

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