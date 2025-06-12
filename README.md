# Airfoil_optimiser
# Currently

-Using a genetic algorithm to optimise an aerofoil (maximise Cl/Cd) at a particular Reynolds number. 

-This program program will maximise L/D max and so may not always lead to increased performance at all angles of attack. 

-This program uses Hicks-Hennes functions to create pertubations on the aerofoils surface, the amplitude of which is governed by the genetic sequence.

-Currently the aerofoil coordinates must be manually input onto the program, because this program was to optimise one individual aerofoil, however, the program can be used on any other aerofoil. 

# Planned Additions

-Allow the program to take a file input of aerofoil coordinates to allow for more user friendly usage.

# NOTE

-The evaluation of airfoil performance in this program relies on the use of Xfoils 2D solver, whilst this software does produce accurate results they are not perfect and the program may struggle to produce effective results when faced woth extremely low values of Re.
