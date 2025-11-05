test_bruteforce = [[0.0, 1.1, 0.79, 0.52, 0.39], [1.1, 0.0, 0.34, 1.0, 0.71], [0.79, 0.34, 0.0, 0.66, 0.41],
                   [0.52, 1.0, 0.66, 0.0, 0.5], [0.39, 0.71, 0.41, 0.5, 0.0]]


# test_bruteforce = [[0.0, 3.1, 0.79, 0.52, 0.39], [3.1, 0.0, 0.34, 1.0, 71], [0.79, 0.34, 0.0, 0.66, 0.41], [0.52, 1.0, 0.66, 0.0, 0.5], [0.39, 71, 0.41, 0.5, 0.0]]
#
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
        self.petite_distance = None
        self.meilleur_trajet = None

    # cherche le meilleur chemin,
    def search(
            self,
            ville_actuelle,  # position du voyageur
            ville_restante,  # liste de ville non visité
            distance_parcourue,  # distance déjà parcouru pour arrivé jusque ville_actuelle
            path
    ):

        # condition de sortie :
        if len(ville_restante) == 0:
            retour_départ = self.matrice[ville_actuelle][
                0]  # retour_depart c'est la distance pour retourné sur A  la ville ded épart ( pas de variable start)
            total_dist = distance_parcourue + retour_départ  # on ajoute la distance retour_départ pour aoir la distance de la "boucle complete"
            trajet = [0] + path + [
                0]  # on par de A ( donc 0) et come on fait une boucle pour retourné vers le départ A ( donc toujours 0) on ajoute au ville traversé (path) 0 avant et aprés

            if self.petite_distance is None or total_dist < self.petite_distance:
                self.petite_distance = total_dist
                self.meilleur_trajet = trajet
            print("trajet suivi:", trajet, "| distance parcouru:", round(total_dist, 2))
            return
        # TODO branch & bound
        if self.petite_distance is not None and self.petite_distance < distance_parcourue:
            print("calcule abort, trop long", round(distance_parcourue, 2))
            return

        i = 0
        while i < len(ville_restante):
            prochaine_ville = ville_restante[i]
            distance_etape = self.matrice[ville_actuelle][
                prochaine_ville]  # jsp comment l'appeler, c'est la distance entre la prochaine ville et la ville actuelle
            new_ville_restante = ville_restante[:i] + ville_restante[
                                                      i + 1:]  # enfaite ici on enleve juste la ville actuele de la liste ville restante
            new_path = path[
                       :]  # [:] pour faire une copie de path et non une référence, voir :  https://stackoverflow.com/questions/2612802/how-do-i-clone-a-list-so-that-it-doesnt-change-unexpectedly-after-assignment
            new_path.append(prochaine_ville)
            self.search(prochaine_ville, new_ville_restante, distance_parcourue + distance_etape, new_path)
            i += 1

    def bruteforce(self) -> tuple[float, list[int]]:

        if self.n == 0:
            print("La matrice de ville est vide")
            return (0.0, [])
        if self.n == 1:
            print("il n'y a qu'une ville dans la matrice fourni")
            return (0.0, [])

        # construie la liste des villes à visiter
        ville_a_voir = []
        j = 1
        while j < self.n:
            ville_a_voir.append(j)
            j += 1

        # construie la liste des villes à visiter (sans predndre en compte la ville de départ 0
        self.search(0, ville_a_voir, 0.0, [])

        # résumé final
        print("meilleur tour trouvé:", self.meilleur_trajet, "| distance:", round(self.petite_distance, 6))
        return (self.petite_distance, self.meilleur_trajet)


matrice = Tsp_solver(test_bruteforce)

resultat = matrice.bruteforce()
print(resultat)