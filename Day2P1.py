import random
from typing import List, Set, Dict

class Individual:
    def __init__(self, genes: List[int]):
        self.genes = genes
        self.value = None

    def __hash__(self):
        return hash(tuple(self.genes))

    def __eq__(self, other):
        return tuple(self.genes) == tuple(other.genes)

    def fitness(self, items: List[Dict[str, int]], max_knapsack_weight: int) -> float:
        """
        Calculate the fitness of the individual.

        Fitness is defined as the total value of the knapsack if the weight is within the limit,
        otherwise, the fitness is 0.

        Parameters:
        items (List[Dict[str, int]]): List of items where each item is represented as a dictionary with 'weight' and 'value' keys.
        max_knapsack_weight (int): Maximum weight that the knapsack can carry.

        Returns:
        float: The fitness value of the individual.
        """
        total_value = sum(
            int(bit) * item['value'] for item, bit in zip(items, self.genes)
        )
        total_weight = sum(
            int(bit) * item['weight'] for item, bit in zip(items, self.genes)
        )

        if total_weight <= max_knapsack_weight:
            return total_value

        return 0.0

def generate_initial_population(items: List[Dict[str, int]], count: int = 6) -> List[Individual]:
    """
    Generate an initial population of individuals for the genetic algorithm.

    Parameters:
    items (List[Dict[str, int]]): List of items where each item is represented as a dictionary with 'weight' and 'value' keys.
    count (int): The number of individuals to generate in the population. Default is 6.

    Returns:
    List[Individual]: A list of individuals representing the initial population.
    """
    # Your code here
    return [''.join(random.choices(["1", "0"], k=4)) for _ in range(count)]


def selection(population: List[Individual], items: List[Dict[str, int]], max_knapsack_weight: int) -> List[Individual]:
    """
    Select individuals from the population based on their fitness using tournament selection.

    Parameters:
    population (List[Individual]): The current population of individuals.
    items (List[Dict[str, int]]): List of items where each item is represented as a dictionary with 'weight' and 'value' keys.
    max_knapsack_weight (int): Maximum weight that the knapsack can carry.

    Returns:
    List[Individual]: A list of selected individuals to be parents.
    """
    # Your code here
    for i in population:
        i.value = i.fitness(items, max_knapsack_weight)
    pop = sorted(population, key=lambda x : x.value)
    return pop[:6]


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Perform crossover between two parents to create an offspring.

    Parameters:
    parent1 (Individual): The first parent individual.
    parent2 (Individual): The second parent individual.

    Returns:
    Individual: The offspring individual resulting from the crossover.
    """
    # Your code here
    return Individual(''.join(random.choices(list(parent1.genes + parent2.genes), k=4)))

def mutate(individual: Individual, mutation_rate: float) -> Individual:
    """
    Mutate an individual by flipping bits with a given mutation rate.

    Parameters:
    individual (Individual): The individual to mutate.
    mutation_rate (float): The mutation rate as a probability.

    Returns:
    Individual: The mutated individual.
    """
    # Your code here
    a = random.choices([0, 1], weights=[mutation_rate, 1 - mutation_rate], k=1)
    if a[0] == 0:
        b = random.choice([i for i in range(len(individual.genes))])
        s = list(individual.genes)
        s[b] = "1" if s[b] == "0" else "0"
        Individual.genes =  ''.join(s)
        return individual
    return individual
    

def next_generation(population: List[Individual], items: List[Dict[str, int]], max_knapsack_weight: int, mutation_rate: float) -> List[Individual]:
    """
    Create the next generation by selecting, crossing over, and mutating individuals from the current population.

    Parameters:
    population (List[Individual]): The current population of individuals.
    items (List[Dict[str, int]]): List of items where each item is represented as a dictionary with 'weight' and 'value' keys.
    max_knapsack_weight (int): Maximum weight that the knapsack can carry.
    mutation_rate (float): The mutation rate as a probability.

    Returns:
    List[Individual]: The next generation of individuals.
    """
    # Your code here
    j = []
    pop = selection(population)
    for i in range(len(pop), 2):
        j.append([mutate(crossover(pop[i], pop[i + 1]), mutation_rate) for _ in range(3)])
    return j



def genetic_algorithm(items: List[Dict[str, int]], max_knapsack_weight: int, population_size: int, generations: int, mutation_rate: float) -> Individual:
    """
    Run the genetic algorithm for a given number of generations and return the best solution found.

    Parameters:
    items (List[Dict[str, int]]): List of items where each item is represented as a dictionary with 'weight' and 'value' keys.
    max_knapsack_weight (int): Maximum weight that the knapsack can carry.
    population_size (int): The number of individuals in the population.
    generations (int): The number of generations to run the algorithm.
    mutation_rate (float): The mutation rate as a probability.

    Returns:
    Individual: The best solution found by the genetic algorithm.
    """
    # Your code here
    pass

# Example usage
items = [
    {'weight': 7, 'value': 5},
    {'weight': 2, 'value': 4},
    {'weight': 1, 'value': 7},
    {'weight': 9, 'value': 2}
]
max_weight = 10
population_size = 6
generations = 2 # CHange this to 10 and then 20.
mutation_rate = 0.01

# best_solution = genetic_algorithm(items, max_weight, population_size, generations, mutation_rate)
# a = generate_initial_population(items)
# print(mutate("1001", 0.9))
# print(crossover(a[0], a[1]))
# print(f"Best Solution: {best_solution.genes}")