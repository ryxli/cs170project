# project-fa19
CS 170 Fall 2019 Project

solves an instance of the traveling purchasing problem https://en.wikipedia.org/wiki/Traveling_purchaser_problem
uses heuristic approximation based on “Approximate Algorithms for the Travelling Purchaser Problem” by Hoon Liong Ong

Run

python solver.py inputdirectory/ outputdirectory/
only for inputs 50.in

for inputs 100.in you can also do above, but it takes more time

for inputs 100.in and 200.in you might want to comment out
gshPath, gshDropoffs = generalizedSavingsHeuristic(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix)
in solver.py to make runspeed a lot faster
and just return tshPath, tshDropoffs

then run same command as above
