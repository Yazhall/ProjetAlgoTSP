import argparse
import json
import os
import time

from Brute_Force import (
    Tsp_solver,
    create_matrice,
    read_instance,
    write_instance,
    matrice2listeadjacente,
)


def ask_number(message, default):
    raw = input(f"{message} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print("Entrée invalide, utilisation de la valeur par défaut.")
        return default



def choose_method():
    methods = {
        "1": "bruteforce",
        "2": "b&b",
        "3": "nearest",
        "4": "cheapest",
        "5": "two-opt",
        "0": "back",
    }
    print("Choisissez une méthode :")
    for key, method in methods.items():
        print(f"{key}. {method}")
    choice = input("Votre choix [1]: ").strip()
    return methods.get(choice, "bruteforce")


def generate_instance():
    n = ask_number("Taille de l'instance n", 8)
    if n < 5:
        print("Les instances doivent avoir au moins 5 villes.")
        return None
    seed = ask_number("Seed aléatoire (int)", None)
    euclidian = input("Euclidienne ? (y/n) [y]: ").strip().lower() != "n"
    
    if euclidian:
        matrice = create_matrice(n, seed=seed, euclidian=euclidian)
    else:
        complete = input("Graphe complet ?\n certains algorithmes pourrais ne pas fonctionner ! (y/n) [y]: ").strip().lower() != "n"
        if not complete:
            print("Attention : graphe non complet, certaines méthodes peuvent échouer s'il n'existe pas de cycle hamiltonien.")
        force_triangle = input("Forcer l'inégalité triangulaire ? (y/n) [n]: ").strip().lower() == "y"
        matrice = create_matrice(n, seed=seed, euclidian=euclidian, complete_graph=complete, force_triangle=force_triangle)
    
    return matrice


def load_instance():
    name = input("Nom du fichier (sans extension) depuis saved/: ").strip()
    try:
        return read_instance(name)
    except FileNotFoundError:
        print("Fichier introuvable.")
        return None


def save_solution(path, cost, filename="solution.json"):
    payload = {"path": path, "cost": cost}
    save_path = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, filename), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)
    print(f"Solution sauvegardée dans saved/{filename}")


def run_solver(matrice, method, verbose=False, max_iter=None):
    tsp = Tsp_solver(matrice)
    if method == "bruteforce":
        if len(matrice) > 11:
            print("Brute force non autorisé au-delà de 11 villes.")
            return {"path": [], "total_dist": None}
        return tsp.bruteforce(verbose=verbose)
    if method == "b&b":
        if len(matrice) > 11:
            print("Branch and Bound non autorisé au-delà de 11 villes.")
            return {"path": [], "total_dist": None}
        return tsp.branch_and_bound(verbose=verbose)
    if method == "nearest":
        return tsp.nearest_neighbourg(verbose=verbose)
    if method == "cheapest":
        if len(matrice) > 100:
            print("Cheapest insertion non autorisé au-delà de 100 villes.")
            return {"path": [], "total_dist": None}
        return tsp.cheapest_insertion(verbose=verbose)
    if method == "two-opt":
        print("2-opt nécessite une solution initiale :")
        print("1. nearest neighbor")
        print("2. cheapest insertion")
        base_choice = input("Choix de la solution initiale [2]: ").strip()
        if base_choice == "1":
            heur = tsp.nearest_neighbourg(verbose=verbose)
        else:
            heur = tsp.cheapest_insertion(verbose=verbose)
        if not heur or not heur.get("path"):
            print("Impossible de générer la solution gloutonne initiale pour 2-opt.")
            return heur
        return tsp.two_opt_solver(
            heur["path"],
            verbose=verbose,
            max_iterations=max_iter if max_iter is not None else 1000,
        )
    raise ValueError("Méthode inconnue")


def print_result(result, verbose=False):
    if result is None:
        print("Aucun résultat.")
        return
    path = result.get("path")
    cost = result.get("total_dist")
    print("Tour:", path)
    print("Coût:", cost)
    if verbose:
        print("Taille du tour:", len(path) if path else 0)


def main():
    parser = argparse.ArgumentParser(description="Console Line Interface TSP")
    parser.parse_args()

    matrice = None
    while matrice is None:
        print("\nMenu principal :")
        print("1. Générer une instance")
        print("2. Charger une instance (CSV depuis saved/)")
        print("Other. Quitter")
        choice = ask_number("Votre choix", 1)
        if choice == 1:
            matrice = generate_instance()
            if matrice is None:
                continue
        elif choice == 2:
            matrice = load_instance()
        else:
            return
    again = True
    while again:
        method = choose_method()
        if method == "back":
            # retour au choix d'instance
            return main()
        verbose_choice = input("Mode verbeux ? (y/n) [n]: ").strip().lower() == "y"
        max_iter_choice = None
        if method == "two-opt":
            raw_iter = input("Limite d'itérations pour 2-opt (entrée vide pour défaut 1000): ").strip()
            if raw_iter != "":
                try:
                    max_iter_choice = int(raw_iter)
                except ValueError:
                    print("Entrée invalide, utilisation de 1000.")
                    max_iter_choice = 1000
            else:
                max_iter_choice = 1000
        start = time.perf_counter()
        result = run_solver(matrice, method, verbose=verbose_choice, max_iter=max_iter_choice)
        duration = time.perf_counter() - start
        print_result(result, verbose=verbose_choice)
        print(f"Temps écoulé: {duration:.4f}s")

        if result and result.get("path"):
            choice = input("Sauvegarder (s) la solution, (i) l'instance, (a) les deux, autre pour ignorer [none]: ").strip().lower()
            if choice in ("s", "a"):
                filename = input("Nom du fichier solution (sans extension, .json ajouté) [solution]: ").strip() or "solution"
                save_solution(result["path"], result["total_dist"], filename=f"{filename}.json")
            if choice in ("i", "a"):
                fname = input("Nom du fichier instance (sans extension, .csv ajouté) [last_generated]: ").strip() or "last_generated"
                write_instance(matrice, fname)

        again = input("Relancer avec une autre méthode sur la même instance ? (y/n) [n]: ").strip().lower()
        again = again == "y"


if __name__ == "__main__":
    main()
