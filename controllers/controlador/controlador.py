from controller import Robot
import numpy as np
from scipy.integrate import solve_ivp

# Inicializar robot y tiempo de simulación
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = timestep / 1000.0  # en segundos

# Obtener motores
nombres_motores = ["motor_fl", "motor_fr", "motor_bl", "motor_br"] 
motores = [robot.getDevice(nombre) for nombre in nombres_motores]

# Activar control por velocidad
for motor in motores:
    motor.setPosition(float('inf'))
    motor.setVelocity(0.0)

# Parámetros Kuramoto
N = 4
omega = np.array([2.0] * N)   # frecuencia natural
K = 1.2                       # acoplamiento entre osciladores
A = 0.6                       # amplitud del movimiento (en rad)

# Fases iniciales para patrón de caminata secuencial
# Orden: FL, FR, BL, BR
phi_0 = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])

estado = phi_0.copy()
tiempo = 0.0

def kuramoto(t, phi):
    dphi_dt = np.zeros(N)
    for i in range(N):
        acoplamiento = sum(np.sin(phi[j] - phi[i]) for j in range(N))
        dphi_dt[i] = omega[i] + (K / N) * acoplamiento
    return dphi_dt

# Loop de simulación
while robot.step(timestep) != -1:
    # Resolver el sistema en el próximo paso
    resultado = solve_ivp(kuramoto, [tiempo, tiempo + dt], estado, method='RK45')
    estado = resultado.y[:, -1]
    tiempo += dt

    # Convertir fases a movimiento oscilatorio de caminata
    # Avanza y retrocede suavemente con sinusoidal
    posiciones = A * np.sin(estado)

    # Aplicar posiciones a motores
    for i in range(N):
        motores[i].setPosition(posiciones[i])
        motores[i].setVelocity(3.0)  # velocidad del motor 