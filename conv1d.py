import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation
def rect(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)

nums = 1000
t = np.linspace(-2,2,nums)

rect1 = rect(t)
rect2 = rect(t)
plt.plot(t, rect1)
plt.show()
conv = signal.convolve(rect1,rect2,mode='same')  * (t[1] - t[0])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(-0.1, 1.2)
ax.set_xlim(-5, 5)

line_rect1, = ax.plot(t, rect1, label="Rectangle 1", color='blue')
line_rect2, = ax.plot([], [], label="Rectangle 2 (Sliding)", color='red')
line_convolution, = ax.plot(t, np.zeros_like(t), label="Convolution", color='orange')

def animate(i):
    shift = t[i] - t[len(t) // 2]
    rect2_shifted = rect(t - shift)

    overlap = rect1 * rect2_shifted

    line_rect2.set_data(t, rect2_shifted)
    line_convolution.set_data(t, signal.convolve(rect1, rect2, mode='same') * (t[1] - t[0]))

    return line_rect1, line_rect2, line_convolution
ani = FuncAnimation(fig, animate, frames=len(t), interval=20, blit=True)
ani.save('./animation.gif', fps=30)
plt.show()