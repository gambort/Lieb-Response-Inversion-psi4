# Lieb-Response-Inversion-psi4
 Implementation of "Lieb-response" inversion in psi4

Requires psi4 (tested on version 1.6)

The main code is Invert-Lieb, which can (in theory) invert a HF
density (for testing purposes) or CCSD density (--CCSD tag) for any
molecule psi4 can converge. The default options seem to work pretty
well except in cases of high symmetry.

Run as:

Invert-Lieb.py --CCSD --Basis SOMEBASIS --DFA INITIALDFA -M XYZFILENAME

(performs an inversion of a CCSD density for XYZFILENAME using
SOMEBASIS set and starting from INITIALDFA)

Invert-Lieb.py --help

(gives the full list of options)

* See the shell scripts for examples of usage.

* The other codes, Atom.py and Frac.py do more specialized jobs and are
for expert users only.

* Note, the Invert-Lieb.py code is provided to illustrate the working
of the library psi4Dens.LibInvert (e.g. see Frac.py for different use)


