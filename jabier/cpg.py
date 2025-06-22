import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =============================================
# 1. Modelo de Kuramoto extendido para CPG
# =============================================
def kuramoto_cpg(t, y, omega, K, phi, a, R):
    """
    Sistema de EDOs para el modelo CPG basado en Kuramoto
    Args:
        y: Vector de estado [theta_0, ..., theta_7, r_0, ..., r_7]
        omega: Frecuencias naturales (rad/s)
        K: Matriz de acoplamientos
        phi: Matriz de desfases
        a: Coeficientes de convergencia de amplitud
        R: Amplitudes deseadas
    """
    n = len(omega)
    theta = y[:n]
    r = y[n:2*n]
    
    dydt = np.zeros(2*n)
    
    # Ecuaciones para theta_i (fases)
    for i in range(n):
        coupling = 0.0
        for j in range(n):
            if i != j and K[i, j] != 0:  # Solo si hay acoplamiento
                coupling += K[i, j] * np.sin(theta[j] - theta[i] - phi[i, j])
        dydt[i] = omega[i] + coupling
    
    # Ecuaciones para r_i (amplitudes)
    for i in range(n):
        dydt[n+i] = a[i] * (R[i] - r[i])
    
    return dydt

# =============================================
# 2. Configuración de marcha tipo trote
# =============================================
def configure_trot_gait():
    n = 8  # 8 osciladores (4 patas x 2 articulaciones)
    
    # Frecuencia base (1 Hz)
    base_freq = 2 * np.pi * 1.0  # rad/s
    omega = base_freq * np.ones(n)
    
    # Matriz de acoplamientos (inicial: diagonal fuerte, laterales débiles)
    K = np.zeros((n, n))
    # Matriz de desfases
    phi = np.zeros((n, n))
    
    # Configuración de índices:
    # [0: LF_hip, 1: LF_knee, 2: RF_hip, 3: RF_knee,
    #  4: LH_hip, 5: LH_knee, 6: RH_hip, 7: RH_knee]
    
    # 1. Acoplamiento entre articulaciones de la misma pata
    for i in range(4):
        hip_idx = 2*i
        knee_idx = 2*i + 1
        K[knee_idx, hip_idx] = 5.0  # Rodilla sigue a cadera
        phi[knee_idx, hip_idx] = -np.pi/2  # 90° de retraso
    
    # 2. Acoplamiento entre patas diagonales (LF-RH, RF-LH)
    diagonal_pairs = [(0, 6), (6, 0), (2, 4), (4, 2)]  # Índices de cadera
    for i, j in diagonal_pairs:
        K[i, j] = 8.0
        phi[i, j] = 0  # En fase
    
    # 3. Acoplamiento entre patas del mismo lado
    same_side_pairs = [(0, 4), (4, 0), (2, 6), (6, 2)]  # LF-LH, RF-RH
    for i, j in same_side_pairs:
        K[i, j] = 8.0
        phi[i, j] = np.pi  # En oposición (180°)
    
    # Amplitudes deseadas (rad)
    R_hip = 0.7  # Caderas
    R_knee = 1.2  # Rodillas
    R = np.array([R_hip, R_knee] * 4)
    
    # Coeficientes de convergencia de amplitud
    a = 5.0 * np.ones(n)
    
    return omega, K, phi, a, R

# =============================================
# 3. Simulación y resolución numérica
# =============================================
def simulate_cpg():
    # Configurar parámetros
    omega, K, phi, a, R = configure_trot_gait()
    n = len(omega)
    
    # Condiciones iniciales
    theta0 = np.random.uniform(0, 2*np.pi, n)  # Fases aleatorias
    r0 = 0.1 * np.ones(n)  # Amplitudes iniciales pequeñas
    y0 = np.concatenate([theta0, r0])
    
    # Tiempo de simulación
    t_span = [0, 10]  # 10 segundos
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    
    # Resolver el sistema
    sol = solve_ivp(kuramoto_cpg, t_span, y0, args=(omega, K, phi, a, R), 
                    t_eval=t_eval, method='RK45')
    
    # Procesar resultados
    theta = sol.y[:n]
    r = sol.y[n:2*n]
    
    # Calcular ángulos articulares reales
    joint_angles = r * np.sin(theta)
    
    return t_eval, joint_angles, theta, r

# =============================================
# 4. Visualización de resultados
# =============================================
def plot_results(t, joint_angles, theta):
    plt.figure(figsize=(15, 10))
    
    # Ángulos articulares
    plt.subplot(2, 1, 1)
    for i in range(8):
        plt.plot(t, joint_angles[i], label=f'Joint {i}')
    plt.title('Ángulos Articulares')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Ángulo (rad)')
    plt.legend()
    plt.grid(True)
    
    # Diagramas de fase
    plt.subplot(2, 1, 2)
    for i in range(0, 8, 2):
        plt.plot(theta[i], theta[i+1], '.', label=f'Pata {i//2}')
    plt.title('Diagramas de Fase (Cadera vs Rodilla)')
    plt.xlabel('Fase Cadera (rad)')
    plt.ylabel('Fase Rodilla (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cpg_simulation.png')
    plt.show()

# =============================================
# 5. Cálculo de trayectoria del centro de masa
# =============================================
def calculate_com_trajectory(joint_angles):
    """Modelo cinemático simplificado para COM"""
    # Parámetros del robot (longitudes en metros)
    L_hip = 0.15
    L_knee = 0.2
    
    n_points = joint_angles.shape[1]
    com_trajectory = np.zeros((3, n_points))
    
    for t in range(n_points):
        com_sum = np.zeros(3)
        for leg in range(4):
            hip_ang = joint_angles[2*leg, t]
            knee_ang = joint_angles[2*leg+1, t]
            
            # Cinemática directa simplificada
            foot_pos = np.array([
                L_hip * np.sin(hip_ang) + L_knee * np.sin(hip_ang + knee_ang),
                0,
                L_hip * np.cos(hip_ang) + L_knee * np.cos(hip_ang + knee_ang)
            ])
            
            # Posición estimada de la cadera (asumimos altura constante)
            hip_pos = np.array([0, 0, 0.3])  # Altura aproximada
            
            # COM aproximado en medio de la cadera y el pie
            com_sum += (hip_pos + foot_pos) / 2
        
        com_trajectory[:, t] = com_sum / 4
    
    return com_trajectory

def plot_com_trajectory(com_traj):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(com_traj[0, :], com_traj[1, :], com_traj[2, :], 'b-', linewidth=2)
    ax.scatter(com_traj[0, 0], com_traj[1, 0], com_traj[2, 0], 
               c='g', s=100, label='Inicio')
    ax.scatter(com_traj[0, -1], com_traj[1, -1], com_traj[2, -1], 
               c='r', s=100, label='Fin')
    
    ax.set_title('Trayectoria del Centro de Masa (COM)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    plt.savefig('com_trajectory.png')
    plt.show()

# =============================================
# Ejecución principal
# =============================================
if __name__ == "__main__":
    # Simular CPG
    t, joint_angles, theta, r = simulate_cpg()
    
    # Guardar resultados para Webots
    np.save('joint_angles.npy', joint_angles)
    
    # Visualizar resultados
    plot_results(t, joint_angles, theta)
    
    # Calcular y visualizar trayectoria COM
    com_traj = calculate_com_trajectory(joint_angles)
    plot_com_trajectory(com_traj)