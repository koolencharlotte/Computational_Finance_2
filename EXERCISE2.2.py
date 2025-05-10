import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

script_dir = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(script_dir, "fig")
os.makedirs(FIG_DIR, exist_ok=True)
# --- Discretization comparison functions ---
def heston_euler_vs_milstein(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, seed):
    N = int(T/dt)
    rng = np.random.default_rng(seed)
    S_eul = np.full(M, S0, dtype=float)
    V_eul = np.full(M, V0, dtype=float)
    sum_eul = np.zeros(M, dtype=float)
    S_mil = np.full(M, S0, dtype=float)
    V_mil = np.full(M, V0, dtype=float)
    sum_mil = np.zeros(M, dtype=float)
    for _ in range(N):
        sum_eul += S_eul
        sum_mil += S_mil
        Z_V = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)
        Z_S = rho*Z_V + np.sqrt(1-rho**2)*Z2
        # Euler step
        Vp = np.maximum(V_eul, 0)
        V_eul += kappa*(theta - Vp)*dt + xi*np.sqrt(Vp*dt)*Z_V
        S_eul *= np.exp((r-0.5*Vp)*dt + np.sqrt(Vp*dt)*Z_S)
        # Milstein step
        Vm = np.maximum(V_mil, 0)
        V_mil += kappa*(theta - Vm)*dt + xi*np.sqrt(Vm*dt)*Z_V + 0.25*xi**2*dt*(Z_V**2-1)
        S_mil *= np.exp((r-0.5*Vm)*dt + np.sqrt(Vm*dt)*Z_S + 0.5*Vm*dt*(Z_S**2-1))
    avg_eul = sum_eul / N
    avg_mil = sum_mil / N
    pay_eul = np.exp(-r*T)*np.maximum(avg_eul - K, 0)
    pay_mil = np.exp(-r*T)*np.maximum(avg_mil - K, 0)
    me_eul = pay_eul.mean()
    se_eul = pay_eul.std(ddof=1)/np.sqrt(M)
    me_mil = pay_mil.mean()
    se_mil = pay_mil.std(ddof=1)/np.sqrt(M)
    return me_eul, se_eul, me_mil, se_mil

# --- GBM benchmarks ---
def gbm_arithmetic_benchmark(M, S0, K, T, dt, r, sigma, seed):
    N = int(T/dt)
    rng = np.random.default_rng(seed)
    S = np.full(M, S0, dtype=float)
    sum_S = np.zeros(M, dtype=float)
    for _ in range(N):
        sum_S += S
        Z = rng.standard_normal(M)
        S *= np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    A = sum_S/N
    pay = np.exp(-r*T)*np.maximum(A - K, 0)
    return pay.mean(), pay.std(ddof=1)/np.sqrt(M)

def gbm_geometric_benchmark(M, S0, K, T, dt, r, sigma, seed):
    N = int(T/dt)
    rng = np.random.default_rng(seed)
    S = np.full(M, S0, dtype=float)
    sum_log = np.zeros(M, dtype=float)
    for _ in range(N):
        sum_log += np.log(S)
        Z = rng.standard_normal(M)
        S *= np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z)
    G = np.exp(sum_log/N)
    pay = np.exp(-r*T)*np.maximum(G - K, 0)
    return pay.mean(), pay.std(ddof=1)/np.sqrt(M)

# --- Analytical geometric-Asian price ---
def analytical_price(S0, r, T, K, sigma, N):
    sigma_tilde = sigma*np.sqrt((2*N+1)/(6*(N+1)))
    r_tilde = ((r-0.5*sigma**2) + sigma_tilde**2)/2
    d1 = (np.log(S0/K) + (r_tilde+0.5*sigma_tilde**2)*T)/(sigma_tilde*np.sqrt(T))
    d2 = d1 - sigma_tilde*np.sqrt(T)
    return (S0*np.exp((r_tilde-r)*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2))

# --- Control variate Monte Carlo ---
def control_variate_MC(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed):
    N = int(T/dt)
    rng = np.random.default_rng(seed)
    S_h = np.full(M, float(S0), dtype=float)
    V_h = np.full(M, float(V0), dtype=float)
    S_g = np.full(M, float(S0), dtype=float)
    sum_h = np.zeros(M, dtype=float)
    sum_g = np.zeros(M, dtype=float)
    for _ in range(N):
        sum_h += S_h
        sum_g += np.log(S_g)
        Z_V = rng.standard_normal(M)
        Z2 = rng.standard_normal(M)
        Z_S = rho*Z_V + np.sqrt(1-rho**2)*Z2
        V_prev = np.maximum(0.0, V_h)
        V_h += kappa*(theta - V_prev)*dt + xi*np.sqrt(V_prev*dt)*Z_V + 0.25*xi**2*dt*(Z_V**2-1)
        S_h *= np.exp((r-0.5*V_prev)*dt + np.sqrt(V_prev*dt)*Z_S + 0.5*V_prev*dt*(Z_S**2-1))
        S_g *= np.exp((r-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z_S)
    avg_h = sum_h / N
    avg_g = np.exp(sum_g / N)
    Y = np.exp(-r*T)*np.maximum(avg_h - K, 0.0)
    X = np.exp(-r*T)*np.maximum(avg_g - K, 0.0)
    plain = [np.mean(Y), np.std(Y)/np.sqrt(M), np.var(Y)]
    analy = analytical_price(S0, r, T, K, sigma, N)
    cv = [np.mean(Y + c*(analy - X)), np.std(Y + c*(analy - X))/np.sqrt(M), np.var(Y + c*(analy - X))]
    return {'plain': plain, 'control_var': cv, 'analy_price': analy, 'payoffs': (X, Y)}

def optimal_c(X, Y):
    cov = np.sum((Y - np.mean(Y))*(X - np.mean(X)))
    var = np.sum((X - np.mean(X))**2)
    return cov/var

# --- Sensitivity analysis helper ---
def heston_sensitivity(M, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed, parameter):
    grid = np.linspace(*parameter['range'], parameter['points'])
    results = {'plain': [], 'std_plain': [], 'cv': [], 'std_cv': []}
    for i, val in enumerate(grid):
        args = {parameter['name']: val}
        res = control_variate_MC(M, S0, V0,
                                 args.get('K', K), T, dt, r,
                                 args.get('rho', rho), kappa,
                                 theta, args.get('xi', xi), sigma, c, seed)
        results['plain'].append(res['plain'][0])
        results['std_plain'].append(res['plain'][1])
        results['cv'].append(res['control_var'][0])
        results['std_cv'].append(res['control_var'][1])
    return grid, results

# --- Main analysis script ---
if __name__ == "__main__":
    S0, V0, K, T = 100.0, 0.04, 100.0, 1.0
    r, rho, kappa, theta, xi = 0.05, -0.7, 2.0, 0.04, 0.3
    sigma = np.sqrt(theta); c = 1.0; seed = 42; dt = 1e-4

    # 1) Euler vs Milstein
    dts = np.logspace(-5, -1, 10)
    e_me, e_se, m_me, m_se = zip(*(
        heston_euler_vs_milstein(10000, S0, V0, K, T, dt, r, rho, kappa, theta, xi, seed)
        for dt in dts
    ))
    plt.figure(figsize=(5,6))
    plt.xscale('log')
    plt.errorbar(
        dts, m_me, yerr=m_se,
        fmt='o-', capsize=4, label='Milstein'
    )
    plt.errorbar(
        dts, e_me, yerr=e_se,
        fmt='v--', capsize=4, label='Euler'
    )
    plt.xlabel('dt')
    plt.ylabel('Option Price')
    plt.title('Euler vs Milstein (Std Error)')
    plt.legend()
    plt.grid(True, which='both', ls='--')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "euler_vs_milstein.png"))
    plt.show()

    # 2) GBM check
    e_me0, e_se0, m_me0, m_se0 = heston_euler_vs_milstein(10000, S0, V0, K, T, dt, r, rho, kappa, theta, 0.0, seed)
    gbm_arith_m, gbm_arith_se = gbm_arithmetic_benchmark(10000, S0, K, T, dt, r, sigma, seed)
    print("=== Arithmetic Average Check ===")
    print(f"Heston xi=0 - Euler: {e_me0:.4f}, SE={e_se0:.4f}")
    print(f"Heston xi=0 - Milstein: {m_me0:.4f}, SE={m_se0:.4f}")
    print(f"GBM benchmark: {gbm_arith_m:.4f}, SE={gbm_arith_se:.4f}\n")
    
    labels = ['Euler (ξ=0)', 'Milstein (ξ=0)', 'GBM arithmetic']
    means  = [e_me0,          m_me0,            gbm_arith_m]
    ses    = [e_se0,          m_se0,            gbm_arith_se]

    plt.figure(figsize=(6,4))
    plt.errorbar(labels, means, yerr=ses, fmt='o', capsize=5)
    plt.ylabel('Option Price')
    plt.title('Arithmetic Average: Heston vs GBM')
    plt.grid(axis='y', ls='--')
    plt.show()
    # 3) Analytical geometric-Asian price
    analytical = analytical_price(S0, r, T, K, sigma, int(T/dt))
    gbm_geo_m, gbm_geo_se = gbm_geometric_benchmark(
        10000, S0, K, T, dt, r, sigma, seed
    )
    print("—— Geometric Average Benchmark ——")
    print(f"Closed-form      : {analytical:.4f}")
    print(f"GBM geometric    : {gbm_geo_m:.4f} ± {gbm_geo_se:.4f}")
    print(f"Difference       : {abs(analytical - gbm_geo_m):.4f}")

    # 4) Control variate comparison
    res = control_variate_MC(10000, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed)
    plain_m, plain_se, _ = res['plain']
    cv_m,    cv_se,    _ = res['control_var']
    print(f"Plain MC = {res['plain'][0]:.4f} ± {res['plain'][1]:.4f}, Var={res['plain'][2]:.4e}")
    print(f"CV MC    = {res['control_var'][0]:.4f} ± {res['control_var'][1]:.4f}, Var={res['control_var'][2]:.4e}\n")

    labels = ['Plain MC', 'CV MC']
    means  = [plain_m,    cv_m]
    ses    = [plain_se,   cv_se]

    plt.figure(figsize=(5,4))
    plt.errorbar(labels, means, yerr=ses, fmt='s', capsize=5)
    plt.ylabel('Option Price')
    plt.title('Control Variate Comparison')
    plt.grid(axis='y', ls='--')
    plt.show()

    # 5) Variance Reduction Efficacy
    Ms = [5000, 10000, 50000, 100000]
    efficacy = {'M': [], 'plain': [], 'std_plain': [], 'cv': [], 'std_cv': []}

    for i, M_paths in enumerate(Ms):
        out = control_variate_MC(
            M_paths, S0, V0, K, T, dt,
            r, rho, kappa, theta, xi, sigma, c,
            seed + i
        )
        print(f"M = {M_paths}: Plain = {out['plain'][0]:.4f} ± {out['plain'][1]:.4f}, "
              f"CV = {out['control_var'][0]:.4f} ± {out['control_var'][1]:.4f}")
    
        efficacy['M'].append(M_paths)
        efficacy['plain'].append(out['plain'][0])
        efficacy['std_plain'].append(out['plain'][1])
        efficacy['cv'].append(out['control_var'][0])
        efficacy['std_cv'].append(out['control_var'][1])

    plt.figure(figsize=(6,6))
    plt.errorbar(
        efficacy['M'], efficacy['plain'],
        yerr=efficacy['std_plain'],
        fmt='o-', label='Plain MC', capsize=5
    )
    plt.errorbar(
    efficacy['M'], efficacy['cv'],
    yerr=efficacy['std_cv'],
    fmt='s--', label='Control Variate MC', capsize=5
)
    plt.xlabel('Number of Paths M')
    plt.ylabel('Option Price')
    plt.title('Variance Reduction Efficacy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "2.4(a).png"))
    plt.show()


    # 6) Sensitivity analysis with errorbars
    def plot_single_sensitivity(param, title, xlabel, filename):
        grid, res = heston_sensitivity(10000, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c, seed, param)
        print(f"--- Sensitivity: {title} ---")
        print(f"{xlabel} grid: {grid}")
        print(f"Plain MC prices: {res['plain']}")
        print(f"Plain MC SE    : {res['std_plain']}")
        print(f"CV    prices   : {res['cv']}")
        print(f"CV    SE       : {res['std_cv']}\n")
        plt.figure(figsize=(5, 4))
        plt.errorbar(
            grid - 0.02 * np.array(grid), res['plain'], yerr=res['std_plain'],
            fmt='o', label='Plain MC', capsize=4
        )
        plt.errorbar(
            grid + 0.02 * np.array(grid), res['cv'], yerr=res['std_cv'],
            fmt='x', label='CV MC', capsize=4
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, filename))
        plt.close()


    
    plot_single_sensitivity(
        {'name': 'xi', 'range': (0.1, 1.0), 'points': 5},
        'Price vs xi', 'xi', '2.4b_xi.png'
    )
    plot_single_sensitivity(
        {'name': 'rho', 'range': (-0.9, 0.5), 'points': 5},
        'Price vs rho', 'rho', '2.4b_rho.png'
    )
    plot_single_sensitivity(
        {'name': 'K', 'range': (95, 105), 'points': 5},
        'Price vs K', 'K', '2.4b_K.png'
    )


    # 7) Optimal c and plain comparison
    X, Y = res['payoffs']
    c_opt = optimal_c(X, Y)
    res_opt = control_variate_MC(10000, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, c_opt, seed)
    print(f"Optimal c* = {c_opt:.4f}")
    print(f"CV with c* = {res_opt['control_var'][0]:.4f} ± {res_opt['control_var'][1]:.4f}, Var={res_opt['control_var'][2]:.4e}")
    print(f"Plain MC   = {res_opt['plain'][0]:.4f} ± {res_opt['plain'][1]:.4f}, Var={res_opt['plain'][2]:.4e}")
    res_c1 = control_variate_MC(10000, S0, V0, K, T, dt, r, rho, kappa, theta, xi, sigma, 1.0, seed)
    c1_m, c1_se = res_c1['control_var'][0], res_c1['control_var'][1]
    print(f"CV with c=1 : {c1_m:.4f} ± {c1_se:.4f}, Var={res_c1['control_var'][2]:.4e}\n")
    
    labels = ['Plain MC', 'CV (c=1)', 'CV (c*)']
    means  = [res_opt['plain'][0], c1_m, res_opt['control_var'][0]]
    ses    = [res_opt['plain'][1], c1_se, res_opt['control_var'][1]]

    plt.figure(figsize=(6,4))
    plt.errorbar(labels, means, yerr=ses, fmt='d', capsize=5)
    plt.ylabel('Option Price')
    plt.title('Control Variate: c=1 vs c*')
    plt.grid(axis='y', ls='--')
    plt.show()