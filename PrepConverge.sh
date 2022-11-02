for M in ketene ethylene glyoxal butadiene benzene ; do # benzoquinone
    echo ${M}
    ./Invert-Lieb.py -M ./QuestDB/${M}.xyz --ForceIP --CCSD --Basis cc-pvdz --CalcPot
    ./Invert-Lieb.py -M ./QuestDB/${M}.xyz --ForceIP --Basis cc-pvtz
done

for B in cc-pvdz cc-pvtz cc-pvqz; do
    for M in hydrogen_chloride hydrogen_sulfide water; do
	echo "${M} @ ${B}"
	./Invert-Lieb.py -M ./QuestDB/${M}.xyz --CCSD --Basis ${B} --NIter 500 --En_Exit 1e-8 --CalcPot
	./Invert-Lieb.py -M ./QuestDB/${M}.xyz --Basis ${B} --NIter 500 --En_Exit 1e-8 --CalcPot
    done
done