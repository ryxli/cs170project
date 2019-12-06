import os
import sys
import cvxpy as cp
import numpy as np
import itertools

sys.path.append("..")
sys.path.append("../..")
import argparse
import utils

from student_utils import *

"""
======================================================================
  Complete the following function.
======================================================================
"""


def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """

    agraph, message = adjacency_matrix_to_graph(adjacency_matrix)
    edgelist = adjacency_matrix_to_edge_list(adjacency_matrix)

    shortestPaths = nx.floyd_warshall(agraph)

    tourHomes = []
    unvisited_homes = list.copy(list_of_homes[1:])
    current = 0
    while unvisited_homes:
        tourHomes += [current]
        homes = shortestPaths[current]
        length = float('inf')
        nextHome = None
        for i in range(len(list_of_locations)):
            if list_of_locations[i] in unvisited_homes:
                if homes[i] < length:
                    length = homes[i]
                    nextHome = i
        unvisited_homes.remove(list_of_locations[nextHome])
        current = nextHome
    tourHomes += [current]

    path = [0]
    predecessors, _ = nx.floyd_warshall_predecessor_and_distance(agraph)
    for i in range(len(tourHomes)-1):
        subpath = nx.reconstruct_path(tourHomes[i], tourHomes[i+1], predecessors)
        path += subpath[1:]
    path += nx.reconstruct_path(tourHomes[len(tourHomes)-1], 0, predecessors)[1:]

    print(path)

    drop = {}
    for p in path:
        drop[p] = []

    for i in range(len(list_of_locations)):
        if list_of_locations[i] in list_of_homes:
            closest = None
            length = float('inf')
            for p in path:
                if shortestPaths[p][i] < length:
                    length = shortestPaths[p][i]
                    closest = p
            drop[closest] += [i]

    print("before reducing ", cost_of_solution(agraph, path, drop))


    while (path):
        increase = []
        for i in path:
            inc = 0
            for j in range(len(list_of_locations)):
                if list_of_locations[j] in list_of_homes:
                    minDist = float('inf')
                    closest = None
                    excludeMinDist = float('inf')
                    excludeClosest = None
                    for p in path:
                        if shortestPaths[p][j] < minDist:
                            closest = p
                            minDist = shortestPaths[p][j]
                    exclude = list.copy(path)
                    exclude.remove(i)
                    for p in exclude:
                        if shortestPaths[p][j] < excludeMinDist:
                            excludeClosest = p
                            excludeMinDist = shortestPaths[p][j]
                    inc += excludeMinDist - minDist

            increase += [inc]

        maxS = float('-inf')
        maxI = None
        toRemove = []
        for i in range(len(path)-2):
            if path[i] == path[i+1]:
                toRemove += [i]
            s = 2 / 3 * shortestPaths[path[i]][path[i+1]] + 2 / 3 * shortestPaths[path[i+1]][path[i+2]] \
                - 2 / 3 * shortestPaths[path[i]][path[i+2]] - increase[i+1]
            if s >= maxS and (path[i], path[i+2]) in agraph.edges:
                maxS = s
                maxI = i+1
        for i in toRemove:
            if path[i] in path:
                path.remove(path[i])
        if maxS <= 0:
            break;
        if path[maxI] in path:
            path.remove(path[maxI])


    print(path)

    for i in range(len(path)-1):
        if adjacency_matrix[path[i]][path[i+1]] == 0:
            print("uh oh ", path[i], path[i+1])

    dropoffs = {}
    for p in path:
        dropoffs[p] = []

    for i in range(len(list_of_locations)):
        if list_of_locations[i] in list_of_homes:
            closest = None
            length = float('inf')
            for p in path:
                if shortestPaths[p][i] < length:
                    length = shortestPaths[p][i]
                    closest = p
            dropoffs[closest] += [i]

    print(cost_of_solution(agraph, path, dropoffs))

    """
    #ATTEMPT AT LINEAR PROGRAMMING
    numLoc = len(list_of_locations)
    numHom = len(list_of_homes)
    start = 0

    # INITIALIZE ALL VARIABLES AND CONSTANTS
    Z = [] #matrix of edges in graph, represents tour/cycle solution
    C = [] #matrix of travel distances between locations (shortest paths)

    for i in range(numLoc): #hacky hack to figure out what number is starting car location
        if starting_car_location == list_of_locations[i]:
            start = i
        z = []
        c = []
        for j in range(numLoc):
            z += [cp.Variable(1, integer = True)]
            c += [shortestPaths[i][j]]
        Z += [z]
        C += [c]

    X = [] #matrix of TA drop off locations (i: TA/home, j: dropoff location)
    D = [] #matrix of costs of dropping off TA at location ("")
    for i in range(numHom):
        x = []
        d = []
        for j in range(numLoc):
            x += [cp.Variable(1, integer = True)]
            d += [shortestPaths[j][i]]
        X += [x]
        D += [d]

    budget = 0 #calculate from walking distance for shortest paths
    for i in range(numHom):
        budget += shortestPaths[start][i]

    constraints = []

    #CONSTRAINT 1
    constraint1 = D[0][0] * X[0][0]
    for i in range(numHom):
        for j in range(numLoc):
            if i != 0 and j != 0:
                constraint1 += D[i][j] * X[i][j]
    constraints += [constraint1 <= budget]

    #CONSTRAINT 2
    for i in range(numHom):
        constraint2 = X[i][0]
        for j in range(numLoc):
            if j != 0:
                constraint2 += X[i][j]
        constraints += [constraint2 == 1]

    #CONSTRAINT 3
    subsets = []
    for i in range(numLoc):
        subsets += list(itertools.combinations(list(range(numLoc))[1:], i))
    for s in subsets:
        for i in range(numHom):
            constraint3 = X[i][0]
            for j in range(numLoc):
                if j != 0:
                    constraint3 += X[i][j]
            for j in s:
                notInS = [x for x in list(range(numLoc)) if x not in s]
                for k in notInS:
                    constraint3 += 0.5 * Z[j][k]
            constraints += [constraint3 >= 1]
    print("hello")

    #CONSTRAINT 4
    for i in range(numHom):
        for j in range(numLoc):
            constraints += [X[i][j] >= 0]
    #CONSTRAINT 5
    for i in range(numLoc):
        for j in range(numLoc):
            constraints += [Z[i][j] >= 0]

    #MINIMIZING OBJECTIVE
    exp = C[0][0] * Z[0][0]
    for i in range(numLoc):
        for j in range(numLoc):
            if i != 0 and j != 0:
                exp += C[i][j] * Z[i][j]
    objective = cp.Minimize(exp)

    problem = cp.Problem(objective, constraints)
    problem.solve()


    print(problem.status)

    print("optimal purchasing cost is: ", constraint1.value)
    #print("num of product 0 is: ", constraints[1].value)





    # The optimal dual variable (Lagrange multiplier) for
    # a constraint is stored in constraint.dual_value.


"""
"""
    TADropOffs = {}
    carPath = []

    #compute shortest distances to all homes
    unvisited_homes = list.copy(list_of_homes)
    currHome = starting_car_location
    if currHome in unvisited_homes:
        dropOffTA(TADropOffs, currHome, unvisited_homes, currHome)

    while unvisited_homes:
        carPath += [currHome]
        currLengths = []
        currPaths = []
        for home in unvisited_homes:
            length, path = nx.single_source_dijkstra(agraph, currHome, home, None, 'weight')
            currLengths += [length]
            currPaths += [path]

        index = a.index(min(a))
        shortestPath = currPaths[index]
        closestHome = shortestPath[len(currPaths)-1]


        nextLengths = []
        nextPaths = []
        for home in unvisited_homes:
            if closestHome != home:
                length, path = nx.single_source_dijkstra(agraph, closestHome, home, None, 'weight')
                nextLengths += [length]
                nextPaths += [path]

        if nextLengths and nextPaths:
            for i, home in unvisited_homes:
                if nextLengths[i] > currLengths[i]:
                    dropOffTA(TADropOffs, currHome, unvisited_homes, home)

        currHome = closestHome
        dropOffTA(TADropOffs, currHome, unvisited_homes, currHome)

    length, path = nx.single_source_dijkstra(agraph, currHome, starting_car_location, None, 'weight')
    carPath += path
    return carPath, TADropOffs

def dropOffTA(dropoffs, location, unvisited_homes, home):
    if location in dropoffs.keys():
        droppedOff = dropoffs[location]
    else:
        droppedOff = []
    dropoffs[location] = droppedOff + [home]
    unvisited_homes.remove(home)

"""


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""


def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ""
    for node in path:
        string += list_locs[node] + " "
    string = string.strip()
    string += "\n"

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + "\n"
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + " "
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + " "
        strDrop = strDrop.strip()
        strDrop += "\n"
        string += strDrop
    utils.write_to_file(path_to_file, string)


def solve_from_file(input_file, output_directory, params=[]):
    print("Processing", input_file)

    input_data = utils.read_file(input_file)
    (
        num_of_locations,
        num_houses,
        list_locations,
        list_houses,
        starting_car_location,
        adjacency_matrix,
    ) = data_parser(input_data)
    car_path, drop_offs = solve(
        list_locations,
        list_houses,
        starting_car_location,
        adjacency_matrix,
        params=params,
    )

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, "in")

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parsing arguments")
    parser.add_argument(
        "--all",
        action="store_true",
        help="If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file",
    )
    parser.add_argument(
        "input", type=str, help="The path to the input file or directory"
    )
    parser.add_argument(
        "output_directory",
        type=str,
        nargs="?",
        default=".",
        help="The path to the directory where the output should be written",
    )
    parser.add_argument(
        "params", nargs=argparse.REMAINDER, help="Extra arguments passed in"
    )
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
