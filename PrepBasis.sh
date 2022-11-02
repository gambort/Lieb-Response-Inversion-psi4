for B in cc-pvdz aug-cc-pvdz cc-pvtz aug-cc-pvtz cc-pvqz aug-cc-pvqz def2-tzvp def2-qzvp def2-qzvppd; do
    echo ${B}
    ./Invert-Lieb.py --CCSD  --Basis ${B} -M ./QuestDB/hydrogen_chloride.xyz --CalcPot
    ./Invert-Lieb.py --CCSD  --Basis ${B} -M ./QuestDB/hydrogen_chloride.xyz --DFA svwn --Calcdv
    ./Invert-Lieb.py --CCSD  --Basis ${B} -M ./SysDB/H2.mol --DFA pbe --Calcdv
    ./Invert-Lieb.py --CCSD  --Basis ${B} -M ./SysDB/Cl2.mol --DFA pbe --Calcdv
done
