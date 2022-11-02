for d in (0.38, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
    F = open("h2_%.2f.mol"%(d), "w")
    F.write("""0 1

H  0.000  0.000 %6.3f
@H 0.000  0.000  0.000
H  0.000  0.000 %6.3f
"""%(-d/2,d/2))
    F.close()


for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    F = open("lih_%.2f.xyz"%(d), "w")
    F.write("""2

Li  0.000  0.000 %6.3f
H   0.000  0.000 %6.3f
"""%(-d/2,d/2))
    F.close()


for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    F = open("heh_%.2f.mol"%(d), "w")
    F.write("""-1 1

He  0.000  0.000 %6.3f
H   0.000  0.000 %6.3f
"""%(-d/2,d/2))
    F.close()
