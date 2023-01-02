for S in lip li lim cp c cm fp f fm; do
    ./Atom.py --NIter 2000 --Basis aug-cc-pvqz-decon --NoSym -M ./SysDB/${S}.mol --Calcdv | grep Diff
done
