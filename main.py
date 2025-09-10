import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from math import pi

# --- Физические константы ---
mu0 = 4 * pi * 1e-7              # магнитная постоянная
e = 1.602176634e-19              # заряд электрона (Кл)
me = 9.10938356e-31              # масса электрона (кг)

# --- Параметры задачи ---
Rk = 5e-3       # радиус катода (м)
Ra = 15e-3      # радиус анода (м)
n = 2000        # число витков на единицу длины (1/м)

# диапазон напряжений
U1, U2 = 50.0, 500.0
num_U = 5

# параметры интегрирования
t_max = 2e-7    # общее время симуляции (с)
dt = 5e-9       # шаг дискретизации (с)

# сетка по току Ic
Ic_min, Ic_max = 0.0, 5.0
B_search_points = 30

# целевой радиус орбиты (середина между катодом и анодом)
r_target = Rk + (Ra - Rk) / 2.0

# --- Электрическое поле ---
def E_field(x, y, U):
    r = np.hypot(x, y)
    if r <= Rk:
        r = Rk + 1e-12
    Er_mag = -U / (r * np.log(Ra / Rk))
    return Er_mag * np.array([x/r, y/r])

# --- Уравнения движения ---
def eom(t, state, Bz, U):
    x, y, vx, vy = state
    Ex, Ey = E_field(x, y, U)
    q = -e
    ax = (q/me) * (Ex + vy * Bz)
    ay = (q/me) * (Ey - vx * Bz)
    return [vx, vy, ax, ay]

# --- Симуляция движения электрона ---
def simulate(U, Ic):
    Bz = mu0 * n * Ic
    x0 = Rk + 1e-6   # стартуем чуть вне катода
    y0, vx0, vy0 = 0.0, 0.0, 0.0
    state0 = [x0, y0, vx0, vy0]
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(eom, (0, t_max), state0, t_eval=t_eval, args=(Bz, U), rtol=1e-6, atol=1e-9)
    xs, ys = sol.y[0], sol.y[1]
    rs = np.hypot(xs, ys)
    return sol.t, xs, ys, rs, Bz

# --- Оценка радиуса орбиты ---
def radius_score(U, Ic):
    t, xs, ys, rs, Bz = simulate(U, Ic)
    mid = len(rs)//2
    r_mean = np.mean(rs[mid:])   # средний радиус на второй половине траектории
    return r_mean, Bz

# --- Подбор Ic для каждого U ---
Us = np.linspace(U1, U2, num_U)
Ic_vals = np.linspace(Ic_min, Ic_max, B_search_points)
Ic_for_U = np.full_like(Us, np.nan, dtype=float)

tolerance = 2e-4  # допуск по радиусу

for i, U in enumerate(Us):
    best_diff = 1e9
    best_Ic = None
    for Ic in Ic_vals:
        r_mean, Bz = radius_score(U, Ic)
        diff = abs(r_mean - r_target)
        if diff < best_diff:
            best_diff = diff
            best_Ic = Ic
        if diff <= tolerance:
            Ic_for_U[i] = Ic
            break
    if np.isnan(Ic_for_U[i]):
        Ic_for_U[i] = best_Ic

# --- Построение траектории для выбранного U ---
choice_idx = len(Us)//2
U_choice = Us[choice_idx]
Ic_choice = Ic_for_U[choice_idx]
t, xs, ys, rs, Bz = simulate(U_choice, Ic_choice)

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.plot(xs*1000, ys*1000, lw=0.8)
ax1.add_artist(plt.Circle((0,0), Rk*1000, color='k', fill=False, label='Катод'))
ax1.add_artist(plt.Circle((0,0), Ra*1000, color='r', fill=False, linestyle='--', label='Анод'))
ax1.set_xlabel('x (мм)')
ax1.set_ylabel('y (мм)')
ax1.set_title(f'Траектория: U={U_choice:.1f} В, Ic={Ic_choice:.3f} А, B={Bz:.2e} Т')
ax1.set_aspect('equal')
ax1.legend()
plt.savefig("/mnt/data/trajectory.png")

# --- Построение диаграммы Ic(U) ---
fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(Us, Ic_for_U, marker='o')
ax2.set_xlabel('U (В)')
ax2.set_ylabel('Ic (А)')
ax2.set_title('Зависимость Ic(U) для орбиты диаметром (Ra-Rk)')
ax2.grid(True)
plt.savefig("/mnt/data/diagram.png")

results_table = [(float(U), float(Ic)) for U, Ic in zip(Us, Ic_for_U)]
results_table
