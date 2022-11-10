Opts="--DFA pbe0 --CCSD --Basis cc-pvdz --NIter 600 --a_Max 3.0 --eps_Cut 1.0"

# Closed shells
echo '***************************************************************'
echo DOING CLOSED SHELLS
echo '***************************************************************'
for M in ALKBDE10_2_cao ALKBDE10_7_mgo  \
			BH76_9_hf DC13_7_be4 G2RC_10_61 G2RC_10_62 \
			G2RC_10_66 G2RC_10_67 G2RC_20_1 G2RC_20_13 \
			G2RC_20_34 G2RC_20_68 W4-11_104_hnnn \
			W4-11_107_so3 W4-11_113_n2o W4-11_116_p4 \
			W4-11_128_s4-c2v W4-11_130_c2 \
			W4-11_136_foof W4-11_137_o3 W4-11_138_bn \
			W4-11_41_sif4 W4-11_46_alf3 \
			be ca he mg; do
    echo ${M}
    #rm Densities/Conv*${M}.mol*
    ./Invert-Lieb.py -M ./P30-5/${M}.mol ${Opts}
done

# Open shell
echo '***************************************************************'
echo DOING OPEN SHELLS
echo '***************************************************************'
for M in BH76_1_n2ohts BH76_9_hf2ts \
		       W4-11_120_t-hooo \
		       W4-11_124_no2 W4-11_134_fo2 W4-11_135_cloo \
		       SIE4x4_2_h2+_1.5 SIE4x4_3_h2+_1.75 \
		       he_plus SIE4x4_4_he2+_1.0 SIE4x4_5_he2+_1.25 \
		       SIE4x4_6_he2+_1.5 SIE4x4_7_he2+_1.75 \
		       W4-11_109_bn3pi he h; do
    echo ${M}
    #rm Densities/Conv*${M}.mol*
    ./Invert-Lieb.py -M ./P30-5/${M}.mol ${Opts}
done
	 
# NOTE W4-11_114_c-hooo does not work at ground state
echo '***************************************************************'
echo DOING SPECIAL CASES
echo '***************************************************************'
XOpts="--NIter 2000 --ForceCCSD --W_Cut 1.1 --a_Max 0.5"
for S in BH76_1_n2 G2RC_20_34 BH76_1_oh; do
    echo Special ${S}
    ./Invert-Lieb.py -M ./P30-5/${S}.mol ${Opts} ${XOpts} --EThermal 0.010
done
echo '***************************************************************'
echo DOING ATOMS
echo '***************************************************************'
XOpts="--NIter 2000 --ForceCCSD --W_Cut 1.1 --a_Max 1.5 --NoSym"
for S in b c n o f  al si p s cl; do
    echo Special ${S}
    ./Atom.py -M ./P30-5/${S}.mol ${Opts} ${XOpts} 
done
