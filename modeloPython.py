import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros del modelo
N = 4
omega = np.array([2.0] * N)  # frecuencia natural (rad/s)
K = 1.2                      # acoplamiento
A = 0.6                      # amplitud de oscilación (rad)
phi_0 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])  # desfase inicial entre patas

# Función del modelo de Kuramoto
def kuramoto(t, phi):
    dphi_dt = np.zeros(N)
    for i in range(N):
        dphi_dt[i] = omega[i] + (K / N) * sum(np.sin(phi[j] - phi[i]) for j in range(N))
    return dphi_dt

# Integrar el sistema
tiempo_total = 10  # segundos
dt = 0.01
t_eval = np.arange(0, tiempo_total, dt)
sol = solve_ivp(kuramoto, [0, tiempo_total], phi_0, t_eval=t_eval, method='RK45')
fases = sol.y % (2*np.pi)

# Calcular ángulos articulares como función de las fases
angulos = A * np.sin(fases)

# Estimar trayectoria del centro de masa (simplemente promediando senos)
centro_masa = np.mean(angulos, axis=0)

# 1. Gráfico: Fases de marcha
plt.figure(figsize=(8, 4))
plt.plot(sol.t, fases[0], label='Fase FL', color='orange')
plt.plot(sol.t, fases[1], label='Fase FR', color='red')
plt.plot(sol.t, fases[2], label='Fase BL', color='purple')
plt.plot(sol.t, fases[3], label='Fase BR', color='magenta')
plt.title('Fases de Marcha (Kuramoto)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Fase (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Gráfico: Ángulos articulares
plt.figure(figsize=(8, 4))
plt.plot(sol.t, angulos[0], label='Ángulo FL', color='orange')
plt.plot(sol.t, angulos[1], label='Ángulo FR', color='red')
plt.plot(sol.t, angulos[2], label='Ángulo BL', color='purple')
plt.plot(sol.t, angulos[3], label='Ángulo BR', color='magenta')
plt.title('Ángulos Articulares (Oscilación)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Gráfico: Centro de masa
plt.figure(figsize=(8, 4))
plt.plot(sol.t, centro_masa, label='Centro de Masa (estimado)', color='black')
plt.title('Trayectoria del Centro de Masa (Estimación)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Desplazamiento relativo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
