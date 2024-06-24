
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def W( x, y, z, h):
	"""Kernel Gaussiano
	"""
	r = np.sqrt(x**2 + y**2 + z**2)
	w = (1.0/(h*np.sqrt(np.pi)))**3*np.exp(-r**2/h**2)
	return w

def gradW( x, y, z, h ):
	"""Gradiente del kernel gaussiano
	"""
	r = np.sqrt(x**2 + y**2 + z**2)
	n = -2*np.exp(-r**2/h**2)/h**5/(np.pi)**(3/2)
	wx = n*x
	wy = n*y
	wz = n*z
	return wx, wy, wz

def vectores_relativos(ri, rj):
	"""Obtiene los vectores relativos entre partículas

  Recibe:
    ri: Matriz M x 3 con la posición de las partículas
    rj: Matriz N x 3 con la posición de otras partículas

  Devuelve:
    dx, dy, dz: Matrices M x N con las separaciones entre las partículas
	"""
	M = ri.shape[0]
	N = rj.shape[0]

	# Obtiene las componentes x,y,z de las posiciones
	rix = ri[:,0].reshape((M,1))
	riy = ri[:,1].reshape((M,1))
	riz = ri[:,2].reshape((M,1))
	rjx = rj[:,0].reshape((N,1))
	rjy = rj[:,1].reshape((N,1))
	rjz = rj[:,2].reshape((N,1))

	# Vectores relativos
	dx = rix - rjx.T
	dy = riy - rjy.T
	dz = riz - rjz.T

	return dx, dy, dz

def densidad(r, pos, m, h):
	"""Obtiene el valor del campo de densidad en la posición r

  Recibe:
    r: matriz M x 3 donde se quiere calcular el campo
    pos: matiz N x 3 donde están las partículas de SPH
    m: masa de las partículas de SPH
    h: longitud de suavizado o soporte del kernel

  Devuelve:
    rho: Vector M x 1 con las densidades
	"""
	M = r.shape[0]
	dx, dy, dz = vectores_relativos(r, pos);
	rho = np.sum(m*W(dx,dy,dz,h),1).reshape((M,1))

	return rho

def presion(rho, k, gamma):
	""" Ecuación de estado politrópica del medio

  Recibe:
    rho: valor o vector de densidades:
    k: constante de la ecuación de estado
    n: índice politrópico

  Devuelve:
    P: valor o vedtor de presiones
	"""
	P = k*rho**(1+1/gamma)
	return P

def aceleracion(pos, vel, m, h, k, gamma, phi, nu):
  """Calcula la aceleración de cada partícula

  Recibe:
    pos: matriz N x 3 con las posiciones de las partículas
    vel: matriz N x 3 con las velocidades de las partículas
    m: masa de cada partícula
    h: longitud de suavizado
    k, gamma: parametros de la ecuación de estado
    phi: constante del potencial externo
    nu: constante de viscosidad

  Devuelve:
    a: matriz N x 3 con las aceleraciones de las partículas
  """
  N = pos.shape[0]

  # Calcula primero las desnsidades
  rho = densidad(pos, pos, m, h)

  # Calcula las presiones
  P = presion(rho, k, gamma)

  # Vectores relativos para el calculo del gradiente
  dx, dy, dz = vectores_relativos(pos, pos)
  dWx, dWy, dWz = gradW(dx, dy, dz, h)

  # Contribución del gradiente de presión
  ax = -np.sum(m*(P/rho**2+P.T/rho.T**2)*dWx,1).reshape((N,1))
  ay = -np.sum(m*(P/rho**2+P.T/rho.T**2)*dWy,1).reshape((N,1))
  az = -np.sum(m*(P/rho**2+P.T/rho.T**2)*dWz,1).reshape((N,1))

  # Juntamos las aceleraciones
  a = np.hstack((ax,ay,az))

  # Contribución de la viscosidad
  a -= nu*vel

  # Potencial externo: asumimos un potencial del tipo oscilador armónico
  a -= phi * pos

  return a

# Genera posiciones y velocidades
def reinicia_condiciones_iniciales(N):
  pos = np.random.randn(N,3)
  vel = np.zeros_like(pos)
  return pos, vel

# Propiedades del medio
M = 2
N = 500
k = 0.1
gamma = 1 # Indice adiabatico
phi = 2 # Constante de la fuerza del medio
nu = 1 # Visosidad modelada

# Propiedades del SPH
m = M/N
h = 0.1

pos, vel = reinicia_condiciones_iniciales(N)

pos, vel = reinicia_condiciones_iniciales(N)

fig,ax = plt.subplots()


dt = 0.05
for i in tqdm(range(100)):
  plt.cla()
  a = aceleracion(pos, vel, m, h, k, gamma, phi, nu)
  vel += a*dt/2
  pos += vel*dt
  a = aceleracion(pos, vel, m, h, k, gamma, phi, nu)
  vel += a*dt/2

  rho = densidad(pos, pos, m, h)
  plt.scatter(pos[:,0],pos[:,1],c=rho,cmap='autumn', s=10, alpha=0.5)
  ax.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.2))
  ax.set_aspect('equal', 'box')
  ax.set_xticks([-1,0,1])
  ax.set_yticks([-1,0,1])
  ax.set_facecolor('black')
  ax.set_facecolor((.1,.1,.1))
  plt.pause(0.001)

plt.show()
