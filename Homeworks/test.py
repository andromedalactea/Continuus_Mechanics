import plotly.graph_objects as go
import numpy as np
from IPython.display import display

# Asumiendo que las funciones Lmol_dt y colision ya están definidas
ft = 500  # Factor de tiempo
L_mol, dt = Lmol_dt(V, N, rapideces, ft)  # Distancia molecular y tiempo de paso
f_d = .1  # Factor de distancia para colisión
N = 100  # Número de partículas
L = 10.0  # Lado de la caja
posiciones = np.random.rand(N, 3) * L  # Posiciones iniciales
velocidades = np.random.randn(N, 3)  # Velocidades iniciales

# Configuración para la visualización con Plotly
fig = go.FigureWidget(
    data=[go.Scatter3d(x=posiciones[:, 0], y=posiciones[:, 1], z=posiciones[:, 2], mode='markers')]
)
fig.update_layout(scene=dict(xaxis=dict(range=[0, L]), yaxis=dict(range=[0, L]), zaxis=dict(range=[0, L])))


# Función para actualizar la gráfica
def update_positions(posiciones, velocidades):
    scatter = fig.data[0]
    scatter.x = posiciones[:, 0]
    scatter.y = posiciones[:, 1]
    scatter.z = posiciones[:, 2]


# Bucle de simulación
def run_simulation(posiciones, velocidades, N, L, dt, f_d, L_mol, steps=500):
    for t in range(steps):
        # Actualizar posiciones
        posiciones += velocidades * dt

        # Chequear colisiones con las paredes
        for i in range(N):
            for j in range(3):
                if posiciones[i, j] < 0 or posiciones[i, j] > L:
                    velocidades[i, j] *= -1
                posiciones[i, j] = np.clip(posiciones[i, j], 0, L)

        # Chequear colisiones entre partículas
        for i in range(N):
            for j in range(i + 1, N):
                distancia = np.linalg.norm(posiciones[i] - posiciones[j])
                if distancia < f_d * L_mol:
                    velocidades[i], velocidades[j] = colision(posiciones[i], posiciones[j], velocidades[i], velocidades[j])

        # Actualizar la gráfica
        update_positions(posiciones, velocidades)


# Mostrar la figura antes de la animación
display(fig)

# Ejecutar la simulación
run_simulation(posiciones, velocidades, N, L, dt, f_d, L_mol)
