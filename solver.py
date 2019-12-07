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
from collections import defaultdict


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

    gshPath, gshDropoffs = generalizedSavingsHeuristic(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
    trhPath, trhDropoffs = tourReductionHeuristic(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)

    gshCost, _ = cost_of_solution(agraph, gshPath, gshDropoffs)
    trhCost, _ = cost_of_solution(agraph, trhPath, trhDropoffs)

    if type(gshCost) == str:
        gshCost = float('inf')
    if type(trhCost) == str:
        trhCost = float('inf')
    if gshCost < trhCost:
        return gshPath, gshDropoffs
    else:
        return trhPath, trhDropoffs



def generalizedSavingsHeuristic(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    agraph, message = adjacency_matrix_to_graph(adjacency_matrix)
    edgelist = adjacency_matrix_to_edge_list(adjacency_matrix)

    SP = nx.floyd_warshall(agraph)

    path = [starting_car_location]
    loc_opt = ""
    best_cost = float("inf")
    for loc in list_of_locations:
        temp_sum = 0
        for home in list_of_homes:
            h = list_of_locations.index(home)
            l = list_of_locations.index(loc)
            temp_sum += SP[l][h]

        if temp_sum < best_cost:
            loc_opt = loc
            best_cost = temp_sum
    path.append(loc_opt)
    path.append(starting_car_location)

    costperiteminpath = NestedDict()
    decreaseincost = NestedDict()
    cont = True
    tadroppedoff = NestedDict()

    while(cont):
        for k in list_of_homes:
            minval = float("inf")
            for st in path:
                h = list_of_locations.index(k)
                l = list_of_locations.index(st)
                if SP[l][h] < minval:
                    minval = SP[l][h]
                    tadroppedoff[k] = l
                costperiteminpath[tuple(path)][k] = minval

        for k in list_of_homes:
            maxval = 0
            for p in list_of_locations:
                if not p in path:
                    h = list_of_locations.index(k)
                    l = list_of_locations.index(p)
                    val = costperiteminpath[tuple(path)][k] - SP[l][h]
                    decreaseincost[tuple(path)][p][k] = val if val>0 else 0

        savings = NestedDict()
        maxval = float("-inf")
        maxi = None
        maxj = None
        maxp = None
        for i in path:
            for j in path:
                for p in list_of_locations:
                    if not p in path:
                        tempsum = 0
                        for k in list_of_homes:
                            tempsum += decreaseincost[tuple(path)][p][k]
                        i_ = list_of_locations.index(i)
                        j_ = list_of_locations.index(j)
                        p_ = list_of_locations.index(p)
                        val = SP[i_][j_]-SP[i_][p_]-SP[p_][j_] + tempsum
                        savings[i][j][p] = val
                        if val > maxval:
                            maxval = val
                            maxi = i
                            maxj = j
                            maxp = p
        if (maxi != None) and savings[maxi][maxj][maxp] > 0:
            path.insert(path.index(maxi)+1, maxp)
        else:
            cont = False

    dropofflocations = defaultdict(list)
    for k,v in tadroppedoff.items():
        dropofflocations[v].append(list_of_locations.index(k))

    #print(path)

    path = [list_of_locations.index(i) for i in path]

    #print(path)

    count = 0
    fill_list = []
    for (index, thing) in enumerate(path[:-1]):
        current, next_ = thing, path[index + 1]
        if not (current, next_) in agraph.edges:
            fill = nx.dijkstra_path(agraph, current, next_)
            fill_list.append((index, fill))
    count = 0;
    ret_path = []
    for i, p in enumerate(path):
        if count < len(fill_list) and i == fill_list[count][0]:
            for e in fill_list[count][1][:-1]:
                ret_path.append(e)
            count += 1
        else:
            ret_path.append(p)

    #print(path)

    return path, dropofflocations

class NestedDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

def tourReductionHeuristic(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix):
    agraph, message = adjacency_matrix_to_graph(adjacency_matrix)
    edgelist = adjacency_matrix_to_edge_list(adjacency_matrix)

    shortestPaths = nx.floyd_warshall(agraph)

    tourHomes = []
    unvisited_homes = list.copy(list_of_homes)
    current = list_of_locations.index(starting_car_location)
    if current in unvisited_homes:
        unvisited_homes.remove(current)
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

    path = [list_of_locations.index(starting_car_location)]
    predecessors, _ = nx.floyd_warshall_predecessor_and_distance(agraph)
    for i in range(len(tourHomes)-1):
        subpath = nx.reconstruct_path(tourHomes[i], tourHomes[i+1], predecessors)
        path += subpath[1:]
    path += nx.reconstruct_path(tourHomes[len(tourHomes)-1], list_of_locations.index(starting_car_location), predecessors)[1:]

    #print(path)

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

    #print("before reducing ", cost_of_solution(agraph, path, drop))


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
            s = 2 / 3 * shortestPaths[path[i]][path[i+1]] + 2 / 3 * shortestPaths[path[i+1]][path[i+2]] \
                - 2 / 3 * shortestPaths[path[i]][path[i+2]] - increase[i+1]
            if s >= maxS and (path[i], path[i+2]) in agraph.edges:
                maxS = s
                maxI = i+1
        if maxS <= 0:
            break;
        if path[maxI] in path:
            path = path[:maxI] + path[maxI+1:]
        for i in range(len(path)-1):
            if path[i] == path[i+1]:
                path.remove(path[i])


    #print(path)

    dropoffs = {}

    for i in range(len(list_of_locations)):
        if list_of_locations[i] in list_of_homes:
            closest = None
            length = float('inf')
            for p in path:
                if shortestPaths[p][i] < length:
                    length = shortestPaths[p][i]
                    closest = p
            if closest in dropoffs.keys():
                dropoffs[closest] += [i]
            else:
                dropoffs[closest] = [i]

    #print(cost_of_solution(agraph, path, dropoffs))
    return path, dropoffs

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
