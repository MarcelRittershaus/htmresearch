import matplotlib.pyplot as plt
import numpy as np



# shape = (100000, 100000)
# rows = np.int_(np.round_(shape[0]*np.random.random(1000)))
# cols = np.int_(np.round_(shape[1]*np.random.random(1000)))
# vals = np.ones_like(rows)

#m = coo_matrix((vals, (rows, cols)), shape=shape)


num_t = 100
t = np.array(range(num_t))
y = np.array([np.random.random() for _ in t])

sdr_t = [1,1,20,50,90]
sdr = [1,100, 100,1000,61440]


plt.figure()
f, ax = plt.subplots(2, sharex=True)
ax[0].set_axis_bgcolor('black')
ax[0].plot(sdr_t, sdr, 's', color='white', ms=3)
ax[1].plot(t, y)

plt.show()