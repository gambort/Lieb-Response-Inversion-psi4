./Frac.py > __LiH_Frac.out
tail -n4 __LiH_Frac.out

for M in ketene ethylene; do
    ./Frac.py -M ./SysDB/${M}.xyz > __${M}_Frac.out
    tail -n4 __${M}_Frac.out
done
