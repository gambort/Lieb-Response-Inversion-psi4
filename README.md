# Lieb-Response-Inversion-psi4
 Implementation of "Lieb-response" inversion in psi4

Requires psi4 (tested on version 1.6)

The main code is Invert-Lieb, which can (in theory) do an
inversion on any psi4 run, if provided with a reference density.

Run as:

Invert-Lieb.py --CCSD --Basis SOMEBASIS --DFA INITIALDFA -M XYZFILENAME



