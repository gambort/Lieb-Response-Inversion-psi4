for M in ketene ethylene glyoxal butadiene benzene ; do # benzoquinone
    echo ${M}
    ./Invert-Lieb.py -M ./SysDB/${M}.xyz --CCSD --Basis cc-pvdz --CalcPot
    ./Invert-Lieb.py -M ./SysDB/${M}.xyz --Basis cc-pvtz --eps_Cut 1.0
done

for B in cc-pvdz cc-pvtz cc-pvqz; do
    for M in hydrogen_chloride hydrogen_sulfide water; do
	echo "${M} @ ${B}"
	./Invert-Lieb.py -M ./SysDB/${M}.xyz --CCSD --Basis ${B} --NIter 500 --En_Exit 1e-8 --CalcPot
	./Invert-Lieb.py -M ./SysDB/${M}.xyz --Basis ${B} --NIter 500 --En_Exit 1e-8 --CalcPot
    done
done
