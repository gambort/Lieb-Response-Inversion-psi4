import numpy as np
import matplotlib.pyplot as plt
from NiceColours import *

kcal = 627.5

fig, ax = plt.subplots(1,1, figsize=(6,2))

q = np.linspace(0., 1., 11)
Ts = {}

# NOTE - This data is copied from Frac.py output

Ts['lih'] \
    = [ 8.02670, 8.00513, 7.98390, 7.96308,
        7.94273, 7.92291, 7.90374, 7.88537,
        7.86805, 7.85223, 7.83924]
Ts['ketene'] \
    = [ 151.52225, 151.49038, 151.45898, 151.42801,
        151.39756, 151.36756, 151.33807, 151.30900,
        151.28065, 151.25276, 151.22564]
Ts['ethylene'] \
    = [ 77.93843, 77.92930, 77.92057, 77.91222,
        77.90432, 77.89684, 77.88981, 77.88327,
        77.87730, 77.87191, 77.86724]

qp = np.linspace(0, 1, 81)
for KID, ID in enumerate(('LiH', 'ketene', 'ethylene')):
    Ts_S = Ts[ID.lower()]

    Ts_Lin = (1-q)*Ts_S[0] + q*Ts_S[-1]

    DTs = (Ts_S - Ts_Lin)*kcal
    DelTs = (Ts_S[-1] - Ts_S[0])*kcal

    p = np.polyfit(q, DTs, 4)
    DTs_Fit = np.polyval(p, qp)

    Col = NiceColour(KID)

    ax.plot(qp, DTs_Fit, color=Col)
    ax.scatter(q, DTs, color=Col)

    qT = (2*KID+2)/8
    AddBorder(
        ax.text( qT, -0.2, #np.interp(qT, qp, DTs_Fit)+0.1,
                 ID+"\n%.1f"%(DelTs),
                 color=Col, fontsize=14,
                 verticalalignment="top",
                 horizontalalignment="center",
                )
    )

ax.set_xlabel("Charge, $q$ [unitless]", fontsize=14)
ax.set_ylabel("$\\Delta T_s(q)$ [kcal]", fontsize=14)

ax.axis([0,1,-7,0])

fig.tight_layout(pad=0.1)

fig.savefig("FracCation.pdf")
plt.show()
