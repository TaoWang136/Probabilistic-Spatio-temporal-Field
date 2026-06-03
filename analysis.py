# 遍历所有的 node
import pickle
import os
import sys
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from matplotlib.cm import get_cmap
import pandas as pd
import matplotlib.ticker as ticker
# base_path = '/home/wangtao/下载/Trajectron-plus-plus-master/Data_gen/data_92_94'
# # 每个时间步取 data[0][0][t]
def load_pkl_by_timestep(path):
    with open(path, 'rb') as f:
        data = pickle.load(f) # shape: [12, ...]
        return data           # 返回 shape [12, ...]
        
# 添加 Trajectron++ 所在路径

sys.path.append('/home/wangtao/下载/Trajectron-ped-veh/Data_ana')
sys.path.append('/home/wangtao/下载/Trajectron-ped-veh/trajectron')
#from analysis import g       
     
x, y = np.mgrid[-10:3840/48.3:0.3,-10:2160/46.6:0.3]


# # mus_all = load_pkl_by_timestep(os.path.join(base_path, 'mus.pkl'))[0]      # [12, 25, 2]
# # sigmas_all = load_pkl_by_timestep(os.path.join(base_path, 'sigmas.pkl'))[0]   # [12, 25, 2]
# # log_pis_all = load_pkl_by_timestep(os.path.join(base_path, 'log_pis.pkl'))[0]  # [12, 25]
# # corrs_all = load_pkl_by_timestep(os.path.join(base_path, 'corrs.pkl'))[0]      # [12, 25]

# with open('/home/wangtao/下载/Trajectron-plus-plus-master/Data_gen/data_92_94/pre_full.pkl', 'rb') as f:
    # pre_full = pickle.load(f)
# with open('/home/wangtao/下载/Trajectron-plus-plus-master/Data_gen/data_92_94/tru.pkl', 'rb') as f:
    # tru = pickle.load(f)
# with open('/home/wangtao/下载/Trajectron-plus-plus-master/Data_gen/data_92_94/his.pkl', 'rb') as f:
    # his = pickle.load(f)
    

# 二维高斯函数
# def bivariate_gaussian(x, y, mu, sigma_x, sigma_y, rho):
    # norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
    # x_mu = x - mu[0]
    # y_mu = y - mu[1]
    # z_exp = ((x_mu ** 2) / sigma_x ** 2 +
             # (y_mu ** 2) / sigma_y ** 2 -
             # 2 * rho * x_mu * y_mu / (sigma_x * sigma_y)) / (2 * (1 - rho ** 2))
    # return norm * np.exp(-z_exp)
    
    
    
    
def bivariate_gaussian(x, y, mu, sigma_x, sigma_y, rho):
    sigma_x = max(sigma_x, 1e-3)
    sigma_y = max(sigma_y, 1e-3)
    rho = np.clip(rho, -0.999, 0.999)

    norm = 1.0 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho ** 2))
    x_mu = x - mu[0]
    y_mu = y - mu[1]
    z_exp = ((x_mu ** 2) / sigma_x ** 2 +
             (y_mu ** 2) / sigma_y ** 2 -
             2 * rho * x_mu * y_mu / (sigma_x * sigma_y)) / (2 * (1 - rho ** 2))
    return norm * np.exp(-z_exp)

def gmm_overlap_score(mus1, sigmas1, log_pis1, corrs1,
                      mus2, sigmas2, log_pis2, corrs2):
    """
    计算双 GMM 交互核
        S = Σ_m Σ_n  π_m^1 π_n^2  exp( -½ D_mn² )
    Parameters
    ----------
    mus1, mus2         : (K, 2)      - μ = (μx, μy)
    sigmas1, sigmas2   : (K, 2)      - σ = (σx, σy)
    corrs1, corrs2     : (K,)        - ρ
    log_pis1, log_pis2 : (K,)        - log π
    Returns
    -------
    S : float
    """
    # 转成 ndarray，确保为 float64 精度
    mus1     = np.asarray(mus1,     dtype=np.float64)
    mus2     = np.asarray(mus2,     dtype=np.float64)
    sigmas1  = np.asarray(sigmas1,  dtype=np.float64)
    sigmas2  = np.asarray(sigmas2,  dtype=np.float64)
    corrs1   = np.asarray(corrs1,   dtype=np.float64)
    corrs2   = np.asarray(corrs2,   dtype=np.float64)
    w1       = np.exp(np.asarray(log_pis1, dtype=np.float64))   # (25,)
    w2       = np.exp(np.asarray(log_pis2, dtype=np.float64))   # (25,)
    # ---------- (1) μ 差分 ----------
    delta = mus1[:, None, :] - mus2[None, :, :]     # shape (25, 25, 2)
    dx, dy = delta[..., 0], delta[..., 1]           # (25,25)
    # ---------- (2) Σ_i + Σ_j 的元素 ----------
    sx1, sy1 = sigmas1[:, 0], sigmas1[:, 1]         # (25,)
    sx2, sy2 = sigmas2[:, 0], sigmas2[:, 1]
    a = sx1[:, None]**2 + sx2[None, :]**2           # Σ₁₁ 元素 -> a  (25,25)
    c = sy1[:, None]**2 + sy2[None, :]**2           # Σ₂₂ 元素 -> c
    b = (corrs1[:, None] * sx1[:, None] * sy1[:, None] +
         corrs2[None, :] * sx2[None, :] * sy2[None, :])         # off-diag
    det = a * c - b**2                              # |Σ₁+Σ₂|  (25,25)
    # ---------- (3) Mahalanobis   dᵀ Σ⁻¹ d ----------
    # Σ⁻¹ = 1/det * [[ c, -b],[-b, a]]
    D2 = (c * dx**2 + a * dy**2 - 2 * b * dx * dy) / det   # (25,25)
    # ---------- (4) 权重外积 ----------
    W = w1[:, None] * w2[None, :]                   # (25,25)
    return np.sum(W * np.exp(-0.5 * D2))
    
def risk_change_rate(t, *, relative=False):
    """
    返回帧 t 相对于 t-1 的逐点风险变化。
    Parameters
    ----------
    t : int
        当前帧编号，必须 >= 1
    relative : bool, default False
        True  → (R_t - R_{t-1}) / R_{t-1}
        False → R_t - R_{t-1}
    Returns
    -------
    dR : ndarray, shape (m, n)
    """
    R_t   = spatiotemporal_r(t)
    R_tm1 = spatiotemporal_r(t - 1)
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            dR = (R_t - R_tm1) / R_tm1
    else:
        dR = (R_t - R_tm1)/0.4
    return dR


def local_spatial_moran(t, contiguity="queen"):
    """
    计算单帧 t 的 Local Moran's I 与伪 p 值。
    Returns
    -------
    I_sp : ndarray (m, n)  -- 局部空间莫兰指数
    p_sp : ndarray (m, n)  -- Monte-Carlo 伪 p 值
    """
    R = spatiotemporal_r(t)
    m, n = R.shape
    
    # 构造空间权重（rook=4 邻；queen=8 邻）
    W_sp = weights.lat2W(m, n, rook=(contiguity=="rook"))
    W_sp.transform = "r"        # 行标准化
    
    x = R.flatten(order="C")     # 1D 向量
    lm = Moran_Local(x, W_sp)
    
    I_sp = lm.Is.reshape(m, n)
    p_sp = lm.p_sim.reshape(m, n)
    return I_sp, p_sp


def local_space_time_moran(t, spatiotemporal_r, *, lag=1, contiguity="queen"):
    """
    计算指定时刻 t 的局部时空 Moran's I (Local ST-I)。
    """
    # 1. 基本参数
    R0 = spatiotemporal_r(t)
    m, n = R0.shape
    N    = m * n
    frames = list(range(t - lag, t + lag + 1))
    if min(frames) < 0:
        raise IndexError(f"t={t} 太小，不足以向前 lag={lag} 帧")

    # 2. 展平时空向量
    X_list = [spatiotemporal_r(tt).flatten(order="C") for tt in frames]
    x_all   = np.concatenate(X_list)

    # 3. 构造空间权重
    W_sp = weights.lat2W(m, n, rook=(contiguity=="rook"))
    W_sp.transform = "b"

    # 4. 构造时间权重
    T_win = len(frames)
    neigh_t = {
        i: [j for d in range(1, lag+1)
               for j in (i-d, i+d)
               if 0 <= j < T_win]
        for i in range(T_win)
    }
    w_t   = {i: [1.0]*len(neigh_t[i]) for i in neigh_t}
    W_tm  = weights.W(neigh_t, w_t, ids=list(range(T_win)))
    W_tm.transform = "b"

    # 5. 组合时空权重
    if hasattr(W_sp, "kronecker"):
        W_st = W_sp.kronecker(W_tm)
    else:
        spW  = W_sp.sparse
        spT  = W_tm.sparse
        spST = sp.kron(spT, spW, format="csr")
        # ★ 用 pandas.RangeIndex 作为 index
        index = pd.RangeIndex(start=0, stop=spST.shape[0], step=1)
        wsp   = WSP(spST, index=index)
        W_st  = util.WSP2W(wsp)
    W_st.transform = "r"

    # 6. 计算 Local ST-I
    lm    = Moran_Local(x_all, W_st)
    start = lag * N
    I_st  = lm.Is[start:start+N].reshape(m, n)
    p_st  = lm.p_sim[start:start+N].reshape(m, n)
    return I_st, p_st
    
    
    
def replace_outliers_custom(data, lower_pct=10, upper_pct=90, window_size=3):
    """
    替换离群值：低于下分位值的替换为0，高于上分位值的替换为局部均值。

    参数：
        data: np.ndarray，二维数组
        lower_pct: float，低于该百分位的值替换为0（默认10%）
        upper_pct: float，高于该百分位的值替换为局部均值（默认90%）
        window_size: int，用于计算局部均值的邻域窗口大小

    返回：
        替换后的新数组，维度与原始数据相同
    """
    data = np.array(data)
    lower = np.percentile(data, lower_pct)
    upper = np.percentile(data, upper_pct)

    # 定义局部替换函数
    def custom_filter(values):
        center = values[len(values) // 2]
        if center < lower:
            return 0
        elif center > upper:
            local_values = np.delete(values, len(values) // 2)
            return np.mean(local_values)
        else:
            return center

    # 应用局部处理
    filtered_data = generic_filter(data, custom_filter, size=window_size, mode='nearest')

    return filtered_data
    
    
# ---------------------------------------------------------------------
# 1) 低层采样函数：只处理单个时间步 (K,2)
# ---------------------------------------------------------------------
def sample_from_mog(mus, sigmas, corrs, log_pis, n_samples=100, rng=None):
    """
    从二维高斯混合模型(一个时间步)采 n_samples 个点.

    Parameters
    ----------
    mus : (K, 2) array
    sigmas : (K, 2) array     # 标准差
    corrs : (K,) array        # 相关系数 ρ
    log_pis : (K,) array      # 对数混合权重 log π
    n_samples : int
    rng : np.random.Generator or np.random.RandomState or None

    Returns
    -------
    samples : (n_samples, 2) array
    """
    # ---------- 兼容新旧随机源 ----------
    if rng is None:
        try:  # NumPy ≥ 1.17
            rng = np.random.default_rng()
            choice = rng.choice
            multivar = rng.multivariate_normal
        except AttributeError:  # NumPy < 1.17
            rng = np.random.RandomState()
            choice = rng.choice
            multivar = rng.multivariate_normal
    # ------------------------------------

    mus     = np.asarray(mus).reshape(-1, 2)      # (K,2)
    sigmas  = np.asarray(sigmas).reshape(-1, 2)   # (K,2)
    corrs   = np.asarray(corrs).ravel()           # (K,)
    log_pis = np.asarray(log_pis).ravel()         # (K,)

    # 1. soft-max 归一化权重
    pis = np.exp(log_pis - log_pis.max())
    pis = pis / pis.sum()

    K = pis.size
    comp_ids = choice(np.arange(K), size=n_samples, p=pis)
    samples = np.empty((n_samples, 2), dtype=float)

    # 2. 按分量批量采样
    for k in range(K):
        mask = comp_ids == k
        if not mask.any():
            continue

        mu = mus[k]
        sx, sy = sigmas[k]
        rho = float(np.clip(corrs[k], -0.999, 0.999))

        cov = np.array([[sx * sx,        rho * sx * sy],
                        [rho * sx * sy,  sy * sy     ]])

        samples[mask] = multivar(mu, cov, size=mask.sum())

    return samples


# ---------------------------------------------------------------------
# 2) 高层采样函数：一次性处理整个预测序列 (T,K,2)
# ---------------------------------------------------------------------
def sample_future_mog(mus_seq, sigmas_seq, corrs_seq, log_pis_seq,
                      n_samples=100, rng_seed=None):
    """
    对形状 (T, K, 2) 的预测参数，在每个未来时间步采样 n_samples 个点.

    Returns
    -------
    samples_list : list of length T
        samples_list[t] 形状 (n_samples, 2)
    """
    # 创建可重复的随机源
    if rng_seed is not None:
        try:
            rng = np.random.default_rng(rng_seed)
        except AttributeError:
            rng = np.random.RandomState(rng_seed)
    else:
        rng = None

    T = mus_seq.shape[0]
    samples_list = []

    for t in range(T):
        pts = sample_from_mog(mus_seq[t], sigmas_seq[t], corrs_seq[t],
                              log_pis_seq[t], n_samples=n_samples, rng=rng)
        samples_list.append(pts)

    return samples_list
    
    
    
BIG_TTC = 10_000_000
EPS     = 1e-9
def compute_ttc_x(row) -> float:
    # 1️⃣ 取出位置、方向、速度
    x_car, x_ped   = row['x_car'],   row['x_ped']
    dx_car, dx_ped = row['Δx_car'], row['Δx_ped']
    v_car, v_ped   = row['v_x_car'], row['v_x_ped']
    
    # 2️⃣ 判断左右位置（左 = x 较小）
    if x_car <= x_ped:
        x_left,  v_left,  dx_left  = x_car, v_car, dx_car
        x_right, v_right, dx_right = x_ped, v_ped, dx_ped
    else:
        x_left,  v_left,  dx_left  = x_ped, v_ped, dx_ped
        x_right, v_right, dx_right = x_car, v_car, dx_car

    gap = x_right - x_left         # 当前位置间距 (≥0)
    if gap < EPS:                  # 已经重叠，当作 TTC=0
        return 0.0
    
    # 3️⃣ 根据 4 种相对运动情形计算 TTC
    # ——① 都向右
    if dx_left > 0 and dx_right > 0:
        rel_speed = v_left - v_right          # 只有左边比右边快才会追尾
        return gap / rel_speed if rel_speed > EPS else BIG_TTC
    
    # ——② 都向左
    if dx_left < 0 and dx_right < 0:
        rel_speed = abs(v_right) - abs(v_left)  # 右边（后车）必须更快
        return gap / rel_speed if rel_speed > EPS else BIG_TTC
    
    # ——③ 迎面相向（左→右，右→左）
    if dx_left > 0 and dx_right < 0:
        rel_speed = abs(v_left) + abs(v_right)   # 速度相加
        return gap / rel_speed
    
    # ——④ 相背而行（左←，右→）：永远拉开
    return BIG_TTC

def compute_ttc_y(row) -> float:
    # 1. 取出位置、方向、速度
    y_car, y_ped   = row['y_car'],   row['y_ped']
    dy_car, dy_ped = row['Δy_car'], row['Δy_ped']   # >0 向下，<0 向上
    v_car, v_ped   = row['v_y_car'], row['v_y_ped'] # 同向号，与 dy_* 保持符号一致

    # 2. 判断上下位置（上 = y 较小）
    if y_car <= y_ped:
        y_up,  v_up,  dy_up  = y_car, v_car, dy_car
        y_down, v_down, dy_down = y_ped, v_ped, dy_ped
    else:
        y_up,  v_up,  dy_up  = y_ped, v_ped, dy_ped
        y_down, v_down, dy_down = y_car, v_car, dy_car

    gap = y_down - y_up
    if gap < EPS:          # 已经重叠
        return 0.0

    # 3. 四种情形
    # —① 同向向下
    if dy_up > 0 and dy_down > 0:
        rel_speed = abs(v_up) - abs(v_down)          # 只有上车更快才会追尾
        return gap / rel_speed if rel_speed > EPS else BIG_TTC

    # —② 同向向上
    if dy_up < 0 and dy_down < 0:
        rel_speed = abs(v_down) - abs(v_up)  # 下车（后车）必须更快
        return gap / rel_speed if rel_speed > EPS else BIG_TTC

    # —③ 迎面而行（上→下，下→上）
    if dy_up > 0 and dy_down < 0:
        rel_speed = abs(v_up) + abs(v_down)       # 方向相反，速度相加
        return gap / rel_speed

    # —④ 相背而行（上←下，下→下）：永远拉开
    return BIG_TTC


def adjust_ttc(row):
    ttc_x, ttc_y = row['TTC_x'], row['TTC_y']

    # 👉 如果二者没有同时小于 BIG_TTC，直接返回10000000
    if not (ttc_x < BIG_TTC and ttc_y < BIG_TTC):
        return np.inf

    # === 1️⃣ 先计算横向/纵向现距 & 相对速度 ===
    gap_x = abs(row['x_car'] - row['x_ped'])            # 横向间距 |Δx|
    gap_y = abs(row['y_car'] - row['y_ped'])            # 纵向间距 |Δy|

    # 相对速度直接用 *gap / TTC* 可保证与第一次计算一致
    v_rel_x = abs(row['x_v'])                   # 横向相对速度 (>0)
    v_rel_y = abs(row['y_v'])                 # 纵向相对速度 (>0)

    # === 2️⃣ 校正 TTC_y (迎面/同向纵向碰撞) ===
    if gap_x < v_rel_x * ttc_y + 2:                     # 横向距离足够小 → 有效
        ttc_y_corr = ttc_y
    else:                                               # 否则标为无效
        ttc_y_corr = BIG_TTC

    # === 3️⃣ 校正 TTC_x (迎面/同向横向碰撞) ===
    if gap_y < v_rel_y * ttc_x + 2:                     # 纵向距离足够小 → 有效
        ttc_x_corr = ttc_x
    else:
        ttc_x_corr = BIG_TTC

    # === 4️⃣ 返回修正后的最小值 ===
    return min(ttc_x_corr, ttc_y_corr)
    
    
    

def add_tdtc(df,
             eps_dir=1e-6,   # 判断“平行”与“零向量”的阈值
             eps_pos=1e-3):  # 判断“共线”时的垂距阈值
    """
    依据当前位置信息与运动方向，计算 TDTC 及冲突点坐标。
    Parameters
    ----------
    df : pd.DataFrame
        必含以下列：
        ['x_car','y_car','Δx_car','Δy_car','v_x_car','v_y_car',
         'x_ped','y_ped','Δx_ped','Δy_ped','v_x_ped','v_y_ped']
    eps_dir, eps_pos : float
        数值容忍阈值。

    Returns
    -------
    pd.DataFrame
        原 df 追加三列：['tdtc','x_int','y_int']
    """
    def _calc(row):
        # --- 基本向量 & 速度 ---
        p_car = np.array([row['x_car'], row['y_car']], dtype=float)
        p_ped = np.array([row['x_ped'], row['y_ped']], dtype=float)
        d_car = np.array([row['Δx_car'], row['Δy_car']], dtype=float)
        d_ped = np.array([row['Δx_ped'], row['Δy_ped']], dtype=float)

        # 速度标量
        v_car = np.hypot(row['v_x_car'], row['v_y_car'])
        v_ped = np.hypot(row['v_x_ped'], row['v_y_ped'])

        # 方向零向量 → 无法判断
        if v_car < eps_dir or v_ped < eps_dir \
           or np.hypot(*d_car) < eps_dir or np.hypot(*d_ped) < eps_dir:
            return pd.Series([np.nan, np.nan, np.nan],
                             index=['tdtc','x_int','y_int'])

        # ---------- 一、检测是否存在几何交叉 ----------
        A = np.column_stack((d_car, -d_ped))         # 2×2
        detA = np.linalg.det(A)

        if abs(detA) > eps_dir:                      # 非平行
            t_c, t_p = np.linalg.solve(A, p_ped - p_car)
            # 仅接受 **向前** 的交叉 (t>=0)
            if t_c >= 0 and t_p >= 0:
                int_pt = p_car + t_c * d_car
                # 走到交叉点的距离
                dist_car = t_c * np.linalg.norm(d_car)
                dist_ped = t_p * np.linalg.norm(d_ped)
                # 时间 = 距离 / 速度
                time_car = dist_car / v_car
                time_ped = dist_ped / v_ped
                tdtc = abs(time_car - time_ped)
                return pd.Series([tdtc, int_pt[0], int_pt[1]],
                                 index=['tdtc','x_int','y_int'])
        # ---------- 二、若平行，考虑跟驰 ----------
        # 交叉积接近零 → 平行
        if abs(np.cross(d_car, d_ped)) < eps_dir:
            dir_unit = d_car / (np.linalg.norm(d_car) + eps_dir)  # 单位方向
            proj = np.dot(p_ped - p_car, dir_unit)                # 投影距离
            # 垂距≈0 → 几乎同一直线
            perp = np.linalg.norm((p_ped - p_car) - proj * dir_unit)
            if perp < eps_pos:
                # 判断谁在前，谁在后
                if proj > 0:          # pedestrian 在 car 前方
                    distance = proj   # 车到人当前位置的距离
                    tdtc = distance / v_car
                else:                 # car 在 pedestrian 前方
                    distance = -proj  # 人到车当前位置的距离
                    tdtc = distance / v_ped
                return pd.Series([tdtc, np.nan, np.nan],
                                 index=['tdtc','x_int','y_int'])
        # ---------- 三、其余情况：无冲突 ----------
        return pd.Series([np.nan, np.nan, np.nan],
                         index=['tdtc','x_int','y_int'])

    return df.join(df.apply(_calc, axis=1))
    
dt = 0.4
def compute_acceleration(group, vx_col, vy_col):
    vx = group[vx_col].values
    vy = group[vy_col].values

    ax = np.insert(np.diff(vx), 0, 0) / dt
    ay = np.insert(np.diff(vy), 0, 0) / dt

    return pd.DataFrame({'a_x': ax, 'a_y': ay}, index=group.index)
def heading_rate(group, heading_col):
    angle = group[heading_col]
    # 差值并处理角度周期问题：使用 np.unwrap
    rate = np.insert(np.diff(np.unwrap(angle)), 0, 0) / dt
    return rate
     
    
def compute_act_row(row, t=0.4):
    # Step 1: 最短距离方向单位向量 u_δ
    rel_vec = np.array([row['rel_x'], row['rel_y']])
    distance = row['Distance']
    if distance == 0:
        return np.inf  # 避免除以0
    u_delta = rel_vec / distance
    # Step 2: 相对速度、加速度向量
    v_rel = np.array([row['v_x_car'] - row['v_x_ped'], row['v_y_car'] - row['v_y_ped']])
    a_rel = np.array([row['a_x_car'] - row['a_x_ped'], row['a_y_car'] - row['a_y_ped']])
    # Step 3: 投影到 u_delta 方向（点乘）
    rel_v_proj = np.dot(v_rel, u_delta)
    rel_a_proj = np.dot(a_rel, u_delta)
    # Step 4: 方向变化项（转向率差 × 距离）
    rel_theta_rate = row['heading_rate_car'] - row['heading_rate_ped']
    heading_term = rel_theta_rate * distance
    # Step 5: dδ/dt 计算
    d_delta_dt = rel_v_proj + rel_a_proj * t #+ heading_term
    # Step 6: ACT
    if d_delta_dt > 0:
        act = distance / d_delta_dt
    else:
        act = np.inf
    return act
    
    
    
    
    
    
    


def _cov_from_params(sx, sy, rho):
    """
    从 (σx, σy, ρ) 生成 2x2 协方差矩阵 Σ
    Σ = [[σx^2, ρ σx σy],
         [ρ σx σy, σy^2]]
    """
    return np.array([[sx*sx, rho*sx*sy],
                     [rho*sx*sy, sy*sy]], dtype=np.float64)

def _det2(a, b, c):
    """2x2 对称矩阵 [[a, b],[b, c]] 的行列式"""
    return a*c - b*b

def _inv2(a, b, c):
    """2x2 对称矩阵 [[a, b],[b, c]] 的解析逆和行列式"""
    det = _det2(a, b, c)
    inv = (1.0 / det) * np.array([[c, -b],
                                  [-b, a]], dtype=np.float64)
    return inv, det

def gaussian_overlap_closed_form(mu_i, sx_i, sy_i, rho_i,
                                 mu_j, sx_j, sy_j, rho_j,
                                 eps=1e-8):
    r"""
    单高斯间的 Bhattacharyya 系数（你的式 (1)）:
    R = |Σ_i|^{1/4} |Σ_j|^{1/4} / |Σ̄|^{1/2} * exp( -1/8 Δμᵀ Σ̄^{-1} Δμ ),
    其中 Σ̄ = (Σ_i + Σ_j)/2,  Δμ = μ_i - μ_j
    """
    # 数值稳定
    sx_i = max(float(sx_i), eps); sy_i = max(float(sy_i), eps)
    sx_j = max(float(sx_j), eps); sy_j = max(float(sy_j), eps)
    rho_i = float(np.clip(rho_i, -0.999, 0.999))
    rho_j = float(np.clip(rho_j, -0.999, 0.999))

    # Σ_i, Σ_j
    Si = _cov_from_params(sx_i, sy_i, rho_i)  # [[ai, bi],[bi, ci]]
    Sj = _cov_from_params(sx_j, sy_j, rho_j)  # [[aj, bj],[bj, cj]]

    # |Σ_i|, |Σ_j|
    ai, bi, ci = Si[0,0], Si[0,1], Si[1,1]
    aj, bj, cj = Sj[0,0], Sj[0,1], Sj[1,1]
    det_Si = _det2(ai, bi, ci)
    det_Sj = _det2(aj, bj, cj)

    # Σ̄ = (Σ_i + Σ_j)/2
    a_bar = 0.5*(ai + aj)
    b_bar = 0.5*(bi + bj)
    c_bar = 0.5*(ci + cj)

    # Σ̄^{-1} 与 |Σ̄|
    inv_Sbar, det_Sbar = _inv2(a_bar, b_bar, c_bar)

    # Δμ
    dmu = (np.asarray(mu_i, dtype=np.float64) - 
           np.asarray(mu_j, dtype=np.float64)).reshape(2)

    # 指数项：Δμᵀ Σ̄^{-1} Δμ
    quad = float(dmu.T @ inv_Sbar @ dmu)

    # 主体： |Σ_i|^{1/4} |Σ_j|^{1/4} / |Σ̄|^{1/2}
    # 加 max(eps, ·) 防止极小负误差
    pre = (max(det_Si, eps)**0.25) * (max(det_Sj, eps)**0.25) / (max(det_Sbar, eps)**0.5)

    R = pre * np.exp(-0.125 * quad)
    return float(R)

def top1_component_overlap(mus1, sigmas1, log_pis1, corrs1,
                           mus2, sigmas2, log_pis2, corrs2):
    """
    从两个 GMM 中各取“最大概率分量”（argmax log π），
    用闭式公式计算单高斯之间的重叠度 R。
    输入形状：
        mus*      : (K, 2)
        sigmas*   : (K, 2)       (σx, σy)
        log_pis*  : (K,)
        corrs*    : (K,)
    返回：
        R, idx1, idx2
    """
    mus1     = np.asarray(mus1,     dtype=np.float64)
    sigmas1  = np.asarray(sigmas1,  dtype=np.float64)
    log_pis1 = np.asarray(log_pis1, dtype=np.float64)
    corrs1   = np.asarray(corrs1,   dtype=np.float64)

    mus2     = np.asarray(mus2,     dtype=np.float64)
    sigmas2  = np.asarray(sigmas2,  dtype=np.float64)
    log_pis2 = np.asarray(log_pis2, dtype=np.float64)
    corrs2   = np.asarray(corrs2,   dtype=np.float64)

    # 取最大概率分量
    idx1 = int(np.argmax(log_pis1))
    idx2 = int(np.argmax(log_pis2))

    mu_i   = mus1[idx1]
    sx_i, sy_i = sigmas1[idx1]
    rho_i  = corrs1[idx1]

    mu_j   = mus2[idx2]
    sx_j, sy_j = sigmas2[idx2]
    rho_j  = corrs2[idx2]

    R = gaussian_overlap_closed_form(mu_i, sx_i, sy_i, rho_i,
                                     mu_j, sx_j, sy_j, rho_j)
    return R















def two_agent(gaussian_parameter,time_range, veh_name):
    # ===== 1) 同时保存原始字段和清洗后的字段 =====
    raw_fields = []    # 用来决定全局 vmin / vmax
    clean_fields = []  # 用来实际绘图
    for timestep in time_range:
        result_matrices = [1**idx*two_agent_conflict_p(gaussian_parameter,timestep, idx, veh_name)*0.4 for idx in range(16)]
        
        risk_field = sum(result_matrices)       # 原始场
        
        raw_fields.append(risk_field)
        # 去掉极端值（绘图用）
        risk_field_clean = replace_outliers_custom(risk_field, lower_pct=99)
        clean_fields.append(risk_field_clean)
    # ===== 2) 全局颜色范围：来自 8 个“原始”子图的全局最小值 & 最大值 =====
    vmin = min(np.nanmin(f) for f in raw_fields)
    vmax = max(np.nanmax(f) for f in raw_fields)
    print("全局颜色范围(原始数据) vmin, vmax =", vmin, vmax)
    # 显式定义一个统一的归一化
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # ===== 3) 预先计算坐标网格 =====
    X = x #* (3840 / (23.99 + 55.51)) + 23.99 * (3840 / (23.99 + 55.51))
    Y = y #* (2160 / (7.643 + 38.69)) + 7.64 * (2160 / (7.643 + 38.69))
    # ===== 4) 画 2x4 子图，所有子图共享同一个 norm 和 cmap =====
    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 6),
        constrained_layout=True)
    axes = axes.flatten()

    for i, (timestep, risk_field_clean) in enumerate(zip(time_range, clean_fields)):
        ax = axes[i]

        contour = ax.contour(
            X, Y,
            risk_field_clean,
            levels=10,
            cmap='jet',
            linewidths=3.0,      # 🔹线条粗细，例如 0.5, 1.0, 1.5, 2.0 等
            norm=norm       # 🔴 统一的归一化，和 colorbar 完全一致
        )
        ax.set_xlim(-10, 20)
        ax.set_ylim(0,10)
        ax.set_xlabel(f'({chr(97 + i)}) Frame {timestep}')
        ax.tick_params(axis='both')
        ax.clabel(
        contour,
        inline=True,          # 标签在线条上“镂空”出来
        fontsize=8,           # 标签字体大小
        fmt="%.2f" )
    # ===== 5) 构造一个专门给 colorbar 用的 mappable（只负责颜色映射，不画图）=====
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])  # 必须设置一个 array，哪怕是空的
    cbar = fig.colorbar(
        sm,
        ax=axes,                 # 8 个子图共享
        orientation='vertical',
        fraction=0.046,
        pad=0.02)

    cbar.set_label('R')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_formatter))
    plt.show()



def two_agent_grad(gaussian_parameter,time_range, veh_name, arrow_step=3):
    """
    只画风险场的梯度方向箭头，不叠加原来的等高线。
    time_range : 可迭代的时间步
    veh_name   : 例如 ['VEHICLE/406', 'PEDESTRIAN/658']
    arrow_step : 每隔多少个格点画一个箭头（越大越稀疏）
    """

    # ===== 1) 计算每个时刻的风险场（和 two_agent 一样的逻辑）=====
    clean_fields = []
    for timestep in time_range:
        result_matrices = [1**idx*two_agent_conflict_p(gaussian_parameter,timestep, idx, veh_name)*0.4 for idx in range(16)]
        risk_field = sum(result_matrices)       # 原始场
        # 去掉极端值（绘图用）
        risk_field_clean = replace_outliers_custom(risk_field, lower_pct=99)
        clean_fields.append(risk_field_clean)

    # ===== 2) 坐标网格（与你原来代码一致）=====
    # 假设 x, y 已经在外面定义为网格（shape 与 risk_field 一致）
    X = x #* (3840 / (23.99 + 55.51)) + 23.99 * (3840 / (23.99 + 55.51))
    Y = y #* (2160 / (7.643 + 38.69)) + 7.64 * (2160 / (7.643 + 38.69))

    # ===== 3) 2x4 子图，只画箭头 =====
    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 6),
        constrained_layout=True
    )
    axes = axes.flatten()

    for i, (timestep, risk_field_clean) in enumerate(zip(time_range, clean_fields)):
        ax = axes[i]

        # --- 计算梯度 ΔR ---
        # np.gradient 返回 [dR/dy, dR/dx]（axis=0 对应 y，axis=1 对应 x）
        dRy, dRx = np.gradient(risk_field_clean)

        # 子采样网格，避免箭头太密
        Xq = X[::arrow_step, ::arrow_step]
        Yq = Y[::arrow_step, ::arrow_step]
        U  = dRx[::arrow_step, ::arrow_step]
        V  = dRy[::arrow_step, ::arrow_step]

        # 只要方向 => 单位化向量
        mag = np.hypot(U, V)
        mag[mag == 0] = 1.0
        U_norm = U / mag
        V_norm = V / mag

        # 画箭头（方向 = 梯度方向，指向风险增加方向）
        ax.quiver(
            Xq, Yq,
            U_norm, V_norm,
            color='k',
            alpha=0.7,
            angles='xy',
            scale_units='xy',
            scale=1,    # 调这里可以调箭头长度
            width=0.01
        )

        # ax.set_xlim(1000, 2000)
        # ax.set_ylim(1160, 250)
        ax.set_xlabel(f'({chr(97 + i)}) Frame {timestep}')
        ax.tick_params(axis='both')
        ax.set_xlim(-10, 20)
        ax.set_ylim(0,10)
    # 如果 time_range 少于 8 个，把多余子图关掉
    for j in range(len(time_range), len(axes)):
        axes[j].axis('off')

    plt.show()
def risk_field_stats(gaussian_parameter, timestep, agent_name_a,agent_name_b):
    veh_name=[agent_name_a,agent_name_b]

    weights = 0.4 * (1 ** np.arange(16))           # 预先算好 16 个权重
    fields = [
        two_agent_conflict_p(gaussian_parameter, timestep, idx, veh_name)
        for idx in range(16)
    ]
    # 广播乘权重
    stacked = np.stack(fields, axis=0)                # (16, H, W)
    risk_field = np.tensordot(weights, stacked, axes=(0, 0))  # (H, W)

    R = np.asarray(risk_field, dtype=np.float64)
    max_val = float(np.nanmax(R))
    mean_val = float(np.nanmean(R))
    median_val = float(np.nanmedian(R))

    R_pos = np.maximum(R, 0.0)
    total = np.nansum(R_pos)
    if total == 0:
        return np.nan, max_val

    X = x
    Y = y

    x_c = float(np.nansum(R_pos * X) / total)
    y_c = float(np.nansum(R_pos * Y) / total)

    r2 = (X - x_c) ** 2 + (Y - y_c) ** 2
    sigma2 = float(np.nansum(R_pos * r2) / total)
    spread = float(np.sqrt(sigma2))

    return spread, max_val,mean_val,median_val

def cr(gaussian_parameter,time_step,pre_time_step,name_vid):
    f_n=[]
    for vid in name_vid:#['VEHICLE/48.0','VEHICLE/36.0']:
        z_total = np.zeros_like(x, dtype=np.float64)
        mus, sigmas, log_probs,corrs,_= gaussian_parameter(time_step,vid)   # (25, 2)
        probs=np.exp(log_probs)[pre_time_step]
        for m, (sx,sy), rho, p in zip(mus[pre_time_step], sigmas[pre_time_step], corrs[pre_time_step], probs):
            z_total += p * bivariate_gaussian(x, y, m, sx, sy, rho)
        f_n.append(z_total.reshape(-1))####n*100*4一共n个车辆，每个车辆计算出来的Z，每个点的概率密度数值。
    return np.array(f_n).transpose(1,0)##每个点的概率密度数值   n*100*4

def two_agent_conflict_p(gaussian_parameter,time_step, pre_time_step, name_vid):
    f = cr(gaussian_parameter,time_step, pre_time_step, name_vid)     # shape (N, 2)
    
    p_i = f[:, 0]                                  # 智能体1概率
    
    p_j = f[:, 1]                                  # 智能体2概率

    result = p_i * p_j                    

    return result.reshape(x.shape[0], x.shape[1])
    
def risk_field_properties(gaussian_parameter,
                          time_step,
                          veh_id,
                          ped_id,
                          veh_x, veh_y,
                          gamma=0.99,
                          dt=0.4,
                          T=16):
    """
    数值计算给定车辆-行人对在一个时间步上的:
      1) 风险场的质心 (centroid)
      2) 质心到车辆位置的距离
      3) 风险场的熵 (Shannon entropy)
    """

    agents = [veh_id, ped_id]

    # === 1) 计算时间折扣积分后的风险场 ===
    result_matrices = []
    for idx in range(T):
        R_t = two_agent_conflict_p(gaussian_parameter, time_step, idx, agents)
        result_matrices.append((gamma**idx) * R_t * dt)
    risk_field = np.sum(result_matrices, axis=0)   # shape ~ (Ny, Nx)
    risk_field = np.asarray(risk_field, dtype=float)
    risk_field = np.maximum(risk_field, 0.0)

    Ny, Nx = risk_field.shape

    # === 2) 在函数内部生成与 risk_field 同 shape 的 x,y 网格 ===
    # 你原来的写法：
    # x, y = np.mgrid[0:3840/48.3:1, 0:2160/46.6:1]
    # 这里假设这句生成的 grid 跟 risk_field 的 shape 一致
    x_grid, y_grid = x,y

    if x_grid.shape != risk_field.shape or y_grid.shape != risk_field.shape:
        raise ValueError(
            f"Grid shape {x_grid.shape} does not match risk_field shape {risk_field.shape}."
        )

    # === 3) 计算质心 ===
    total_mass = risk_field.sum()
    if total_mass <= 0:
        return np.nan, np.nan, np.nan, np.nan

    centroid_x = float((x_grid * risk_field).sum() / total_mass)
    centroid_y = float((y_grid * risk_field).sum() / total_mass)

    # === 4) 质心到车辆位置的距离 ===
    dx = centroid_x - veh_x
    dy = centroid_y - veh_y
    dist_to_vehicle = float(np.sqrt(dx*dx + dy*dy))

    # === 5) 场的熵 ===
    p = risk_field / total_mass
    mask = p > 0
    entropy = -float(np.sum(p[mask] * np.log(p[mask])))

    return centroid_x, centroid_y, dist_to_vehicle, entropy
    
    
    
    
    
    
def _to_image_coords(x, y):
    scale_x = 3840 / (23.99 + 55.51)
    offset_x = 23.99 * scale_x

    scale_y = 2160 / (7.643 + 38.69)
    offset_y = 7.64 * scale_y   # 保持你原来代码中的 7.64，不动

    x_img = x * scale_x + offset_x
    y_img = y * scale_y + offset_y
    return x_img, y_img
    
    

# x, y = np.mgrid[-10:3840/48.3:0.3,-10:2160/46.6:0.3]




# x, y = np.mgrid[-200:-50:1,-15:0:0.1]

def picture_single(tra_data,agent_A, agent_B, timestep,
                   label_A, label_B, label_A_1, label_B_1,
                   save_name=None):
    """
    输入一个时刻 timestep，只输出一张图像。
    """

    fig, ax = plt.subplots(figsize=(5, 4))
    # 你的数据获取
    tru_A, his_A, pre_A = tra_data(timestep, agent_A)
    tru_B, his_B, pre_B = tra_data(timestep, agent_B)

    # 如果没有预测，直接退出
    if pre_A is None or pre_B is None:
        ax.set_title(f"Frame {timestep} (无数据)")
        ax.axis('off')
        plt.show()
        return

    # 取重叠的未来时间长度
    T = min(pre_A.shape[1], pre_B.shape[1])
    reds  = plt.cm.Reds (np.linspace(1.0, 0.3, T))
    blues = plt.cm.Blues(np.linspace(1.0, 0.3, T))

    # === 新增：用于保存每个时间步的均值坐标（世界坐标系） ===
    mean_xA, mean_yA = [], []
    mean_xB, mean_yB = [], []

    # 画未来散点（第 j 个时间层）
    for j in range(T):
        # A 未来
        xA, yA = pre_A[..., j, 0].ravel(), pre_A[..., j, 1].ravel()
        xA_img, yA_img = _to_image_coords(xA, yA)
        ax.scatter(xA_img, yA_img, s=0.3, alpha=0.2, c=[reds[j]])

        # 记录 A 在该时间步的均值（世界坐标）
        mean_xA.append(xA.mean())
        mean_yA.append(yA.mean())

        # B 未来
        xB, yB = pre_B[..., j, 0].ravel(), pre_B[..., j, 1].ravel()
        xB_img, yB_img = _to_image_coords(xB, yB)
        ax.scatter(xB_img, yB_img, s=0.3, alpha=0.2, c=[blues[j]])

        # 记录 B 在该时间步的均值（世界坐标）
        mean_xB.append(xB.mean())
        mean_yB.append(yB.mean())

    # === 新增：把均值连成轨迹并画出来 ===
    mean_xA = np.array(mean_xA)
    mean_yA = np.array(mean_yA)
    mean_xB = np.array(mean_xB)
    mean_yB = np.array(mean_yB)

    mean_xA_img, mean_yA_img = _to_image_coords(mean_xA, mean_yA)
    mean_xB_img, mean_yB_img = _to_image_coords(mean_xB, mean_yB)

    # A、B 的均值轨迹线（使用实线，便于和历史虚线区分）
    mA, = ax.plot(mean_xA_img, mean_yA_img,
                  color='#ff8066', linewidth=1.5,
                  linestyle='-', label=f'Trajectory mean of pedestrian')
    mB, = ax.plot(mean_xB_img, mean_yB_img,
                  color='#0081cf', linewidth=1.5,
                  linestyle='-', label=f'Trajectory mean of vehicle')

    # 历史轨迹
    xA_his, yA_his = _to_image_coords(his_A[:, 0], his_A[:, 1])
    xB_his, yB_his = _to_image_coords(his_B[:, 0], his_B[:, 1])
    hA, = ax.plot(xA_his, yA_his,
                  color='#ff8066', alpha=1, linestyle='--', label=label_A)
    hB, = ax.plot(xB_his, yB_his,
                  color='#0081cf', alpha=1, linestyle='--', label=label_B)

    # 当前点
    xA_now, yA_now = _to_image_coords(his_A[-1, 0], his_A[-1, 1])
    xB_now, yB_now = _to_image_coords(his_B[-1, 0], his_B[-1, 1])

    pA = ax.scatter(xA_now, yA_now,
                    color='#ff8066', alpha=0.7, label=label_A_1)
    pB = ax.scatter(xB_now, yB_now,
                    color='#0081cf', alpha=0.7, marker='*', label=label_B_1)

    ax.set_xlabel(f'Frame {timestep}')
    # ax.set_xlim(1200, 2400)
    # ax.set_ylim(1250, 0)

    # 图例（现在会包含历史、当前点、以及均值轨迹）
    ax.legend(loc='lower right')

    # 保存或显示
    if save_name is not None:
        plt.savefig(save_name, dpi=300, format="png",
                    bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
    
    
    
    
def picture(tra_data,agent_A, agent_B, time_duration, label_A, label_B, label_A_1, label_B_1, save_name):
    fig, axes = plt.subplots(2, 4, figsize=(12, 4), gridspec_kw={'hspace': 0.34})
    axes = axes.ravel()  # ★ 关键：拉平成一维，axes[i] 才是一个 Axes
    # 可选：保护，避免时间步超过子图数量
    if len(time_duration) > len(axes):
        raise ValueError(f"time_duration 长度 {len(time_duration)} 超过子图数量 {len(axes)}")
    legend_handles, legend_labels = [], []
    for i, timestep in enumerate(time_duration):
        ax = axes[i]
        # 你的数据获取
        tru_A, his_A, pre_A = tra_data(timestep, agent_A)
        tru_B, his_B, pre_B = tra_data(timestep, agent_B)
        # 保护：如果某一侧没有预测，跳过避免 shape 错误
        if pre_A is None or pre_B is None:
            ax.set_title(f"Frame {timestep} (无数据)")
            ax.axis('off')
            continue
        # 取重叠的时间长度
        T = min(pre_A.shape[1], pre_B.shape[1])
        reds  = plt.cm.Reds (np.linspace(1.0, 0.3, T))
        blues = plt.cm.Blues(np.linspace(1.0, 0.3, T))
        # 画未来散点（第 j 个时间层）
        for j in range(T):
            ax.scatter(pre_A[..., j, 0].ravel(), pre_A[..., j, 1].ravel(), s=0.3, alpha=0.2, c=[reds[j]])
            ax.scatter(pre_B[..., j, 0].ravel(), pre_B[..., j, 1].ravel(), s=0.3, alpha=0.2, c=[blues[j]])
        # 历史轨迹与当前点（只在第一个子图加图例句柄）
        if i == 0:
            hA, = ax.plot(his_A[:, 0], his_A[:, 1], color='#ff8066', alpha=1, linestyle='--', label=label_A)
            hB, = ax.plot(his_B[:, 0], his_B[:, 1], color='#0081cf', alpha=1, linestyle='--', label=label_B)
            pA  = ax.scatter(his_A[-1, 0], his_A[-1, 1], color='#ff8066', alpha=0.7, label=label_A_1)
            pB  = ax.scatter(his_B[-1, 0], his_B[-1, 1], color='#0081cf', alpha=0.7, marker='*', label=label_B_1)
            legend_handles.extend([hA, hB, pA, pB])
            legend_labels.extend([label_A, label_B, label_A_1, label_B_1])
        else:
            ax.plot(his_A[:, 0], his_A[:, 1], color='#ff8066', alpha=1, linestyle='--')
            ax.plot(his_B[:, 0], his_B[:, 1], color='#0081cf', alpha=1, linestyle='--')
            ax.scatter(his_A[-1, 0], his_A[-1, 1], color='#ff8066', alpha=0.7)
            ax.scatter(his_B[-1, 0], his_B[-1, 1], color='#0081cf', alpha=0.7, marker='*')

        ax.set_xlabel(f'({chr(97 + i)}) Frame {timestep}')
        # ax.set_xlim(0, 3840/48)
        # ax.set_ylim(2160/48,0)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(218))
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(232))
    # 统一图例（如果需要）
    if legend_handles:
        fig.legend(handles=legend_handles,
                   labels=legend_labels,
                   loc='lower center',
                   ncol=4,
                   bbox_to_anchor=(0.5, -0.01))
    # 保存或显示
    # plt.savefig(save_name, dpi=300, format="png", bbox_inches='tight', pad_inches=0)
    plt.show()
    
    
BG_IMG_PATH = r'/home/wangtao/下载/Trajectron-ped-veh/Data_ana/SinD/SinD_data_tianjing/TJ.png'




W, H = 3840, 2160
X_L, X_R = 23.99, 55.51
Y_T, Y_B = 7.643, 38.69
car_cmap = get_cmap("Blues")    # 车辆用蓝色系
ped_cmap = get_cmap("Reds")     # 行人用红色系   


def plot_one(tra_data,tru,ax, time_step, num,show_legend=False):



    # 背景图
    if BG_IMG_PATH and os.path.exists(BG_IMG_PATH):
        img = plt.imread(BG_IMG_PATH)
        ax.imshow(img, extent=[0, W, 0, H], zorder=0,alpha=0.6)

    # 遍历所有智能体
    for agent in tru[time_step].keys():

        tru_data, his_data, pre_data = tra_data(time_step, agent)

        # 预测步长，用于生成渐变颜色
        if pre_data.size:
            pre_len = pre_data.shape[1]
            alphas = np.linspace(0.8, 0.1, pre_len)

        if agent.startswith("VEHICLE/"):   # 车辆
            base_cmap = car_cmap
        else:                               # 行人
            base_cmap = ped_cmap
        # 历史
        if his_data.size:
            ax.scatter(_tx(his_data[:, 0]), _ty(his_data[:, 1]),
                       color=base_cmap(0.8), s=3, zorder=2)
        # 预测（随步长变浅）
        if pre_data.size:
            for k in range(pre_len):  # 每一条预测轨迹单独画 + 渐变
                ax.scatter(_tx(pre_data[:, k, 0]), _ty(pre_data[:, k, 1]),
                           color=base_cmap(alphas[k]), s=0.6, zorder=1,alpha=0.5)
        # 未来真实
        # if tru_data.size:
        #     ax.scatter(_tx(tru_data[:, 0]), _ty(tru_data[:, 1]),
        #                color=base_cmap(1.0), s=5, label="Ground Truth", zorder=3)
    ax.set_xlabel(f"({chr(97 +num)}) Frame: {time_step}",)
    ax.set_aspect('equal')
    ax.set_xlim(0, W)
    ax.set_ylim(0,H)
    # 图例只在第一张子图显示
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=2)
        
        
def plot_one_FC(tra_data,tru,ax, time_step, num,show_legend=False):



    # 背景图

    img = plt.imread('/home/wangtao/下载/Trajectron-ped-veh/Data_ana/hepingmen_data/hp.jpg')
    ax.imshow(img, extent=[0, 3840, 2160, 0], zorder=0,alpha=0.6)

    # 遍历所有智能体
    for agent in tru[time_step].keys():

        tru_data, his_data, pre_data = tra_data(time_step, agent)

        # 预测步长，用于生成渐变颜色
        if pre_data.size:
            pre_len = pre_data.shape[1]
            alphas = np.linspace(0.8, 0.1, pre_len)

        if agent.startswith("VEHICLE/"):   # 车辆
            base_cmap = car_cmap
        else:                               # 行人
            base_cmap = ped_cmap
        # 历史
        if his_data.size:
            ax.scatter(his_data[:, 0]*56.5, his_data[:, 1]*56.5,
                       color=base_cmap(0.8), s=3, zorder=2)
        # 预测（随步长变浅）
        if pre_data.size:
            for k in range(pre_len):  # 每一条预测轨迹单独画 + 渐变
                ax.scatter(pre_data[:, k, 0]*56.5, pre_data[:, k, 1]*56.5,
                           color=base_cmap(alphas[k]), s=0.6, zorder=1,alpha=0.5)
        # 未来真实
        # if tru_data.size:
        #     ax.scatter(_tx(tru_data[:, 0]), _ty(tru_data[:, 1]),
        #                color=base_cmap(1.0), s=5, label="Ground Truth", zorder=3)
    ax.set_xlabel(f"({chr(97 +num)}) Frame: {time_step}",)
    ax.set_aspect('equal')
    ax.set_xlim(0, 3840)
    ax.set_ylim(2160, 0)
    # 图例只在第一张子图显示
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', ncol=2)       
        
        
        
def plot_range_grid(tra_data,tru,start, step=3, count=8):
    frame_list = list(range(start, start + step * count, step))
    fig, axes = plt.subplots(2,4, figsize=(9,3), constrained_layout=True)
    axes = axes.flatten()
    for i, (ax, t) in enumerate(zip(axes, frame_list)):
        plot_one(tra_data,tru,ax, str(t), i,show_legend=(i == 0))  # ← 只有第一个图 show_legend=True
    for j in range(len(frame_list), len(axes)):  # 若不足 8 张隐藏
        axes[j].axis('off')
    plt.savefig('/home/wangtao/下载/Trajectron-ped-veh/Data_ana/TJ_ped_car.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_range_grid_FC(tra_data,tru,start, step=3, count=8):
    frame_list = list(range(start, start + step * count, step))
    fig, axes = plt.subplots(2,4, figsize=(9,3), constrained_layout=True)
    axes = axes.flatten()
    for i, (ax, t) in enumerate(zip(axes, frame_list)):
        plot_one_FC(tra_data,tru,ax, str(t), i,show_legend=(i == 0))  # ← 只有第一个图 show_legend=True
    for j in range(len(frame_list), len(axes)):  # 若不足 8 张隐藏
        axes[j].axis('off')
        
    plt.savefig('/home/wangtao/下载/Trajectron-ped-veh/Data_ana/tianjin_.png', dpi=300, bbox_inches='tight')
    plt.show()
    



    
    
    
    
    
    
    
def _tx(x):
    return x * (W / (X_L + X_R)) + X_L * (W / (X_L + X_R))
def _ty(y):
    return y * (H / (Y_T + Y_B)) + 7.64 * (H / (Y_T + Y_B))
# 车辆 & 行人颜色梯度 colormap   
 
x, y = np.mgrid[-10:3840/48.3:0.3,-10:2160/46.6:0.3]   
    

    
    
def f_x_n(time_step,pre_time_step):
    f_n=[]
    for vid in car_name_per_timestep[time_step]:
        z_total = np.zeros_like(x, dtype=np.float64)
        mus, sigmas, log_probs,corrs,_= gaussian_parameter(time_step,vid)   # (25, 2)
        probs=np.exp(log_probs)[pre_time_step]
        for m, (sx,sy), rho, p in zip(mus[pre_time_step], sigmas[pre_time_step], corrs[pre_time_step], probs):
            z_total += p * bivariate_gaussian(x, y, m, sx, sy, rho)
        f_n.append(z_total.reshape(-1))####n*100*4一共n个车辆，每个车辆计算出来的Z，每个点的概率密度数值。
    return np.array(f_n).transpose(1,0)##每个点的概率密度数值   n*100*4

def conflict_p(time_step,pre_time_step):
    sum_f = f_x_n(time_step,pre_time_step).sum(axis=1)  
    #print(f_x_n(time_step,pre_time_step).shape)# 每行的总和，shape (2,)
    sum_sq = (f_x_n(time_step,pre_time_step)**2).sum(axis=1)  # 每行的平方和，shape (2,)
    # 应用公式
    result = 0.5 * (sum_f**2 - sum_sq)
    return result.reshape(x.shape[0],x.shape[1])

def decimal_formatter(x, pos):
    return f'{x:.1e}'  # 科学计数法，保留1位小数
# 创建画布和子图

def picture_with_field(tra_data, gaussian_parameter,
                       x1, x2, y1, y2,
                       agent_A, agent_B, time_duration,
                       label_A, label_B, label_A_1, label_B_1,
                       save_name=None):
    """
    在同一张图里，2x4 子图：
      - 背景：两个智能体轨迹分布产生的场强等高线
      - 前景：两个智能体的历史轨迹 + 当前点 + 未来分布散点
      - 右侧：统一颜色柱子

    如果某个时间步计算或绘图出错，在对应子图画一个空图，并标注出错信息。
    """

    # ===== 1) 先算每个时间步的场强（原始 + 清洗），单步报错不影响其它时间步 =====
    raw_fields = []    # 用来决定全局 vmin / vmax（可能有 None）
    clean_fields = []  # 用来实际绘图（可能有 None）

    veh_name = [agent_A, agent_B]

    for timestep in time_duration:
        try:
            result_matrices = [
                two_agent_conflict_p(gaussian_parameter, timestep, idx, veh_name)
                for idx in range(16)
            ]
            risk_field = sum(result_matrices)   # 原始场
            raw_fields.append(risk_field)

            # 去极值后的场（绘图用）
            risk_field_clean = replace_outliers_custom(risk_field, lower_pct=99)
            clean_fields.append(risk_field_clean)
        except Exception as e:
            print(f"[warning] 计算时间步 {timestep} 的场强失败: {e}")
            raw_fields.append(None)
            clean_fields.append(None)

    # ===== 2) 全局颜色范围：只用成功计算的“原始”子图 =====
    valid_raw = [f for f in raw_fields if f is not None]
    if len(valid_raw) == 0:
        print("[error] 所有时间步的场强都计算失败，无法设置颜色范围。")
        # 画一张空图，直接返回
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("No valid risk field")
        ax.axis('off')
        plt.show()
        return

    vmin = min(np.nanmin(f) for f in valid_raw)
    vmax = max(np.nanmax(f) for f in valid_raw)
    print("全局颜色范围(原始数据) vmin, vmax =", vmin, vmax)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ===== 3) 预先计算坐标网格 + 坐标变换系数 =====
    sx = 3840.0 / (23.99 + 55.51)
    ox = 23.99 * sx
    sy = 2160.0 / (7.643 + 38.69)
    oy = 7.64 * sy

    X = x * sx + ox
    Y = y * sy + oy

    def to_px_x(arr):
        return arr * sx + ox

    def to_px_y(arr):
        return arr * sy + oy

    # ===== 4) 建立 2x4 子图 =====
    fig, axes = plt.subplots(
        2, 4,
        figsize=(12, 3),
        constrained_layout=True
    )
    axes = axes.ravel()

    if len(time_duration) > len(axes):
        raise ValueError(f"time_duration 长度 {len(time_duration)} 超过子图数量 {len(axes)}")

    legend_handles, legend_labels = [], []

    # ===== 5) 每个子图：先画场强等高线，再叠加轨迹分布 =====
    for i, (timestep, risk_field_clean) in enumerate(zip(time_duration, clean_fields)):
        ax = axes[i]

        # 如果场强计算失败，就画空子图
        if risk_field_clean is None:
            ax.set_title(f"({chr(97 + i)}) Frame {timestep}\n(field error)")
            ax.axis('off')
            continue

        try:
            # --- 场强等高线 ---
            contour = ax.contour(
                X, Y,
                risk_field_clean,
                levels=10,
                cmap='jet',
                norm=norm,
                linewidths=1.5
            )
            ax.clabel(
                contour,
                inline=True,
                fmt="%.2f"
            )

            # --- 轨迹数据 ---
            tru_A, his_A, pre_A = tra_data(timestep, agent_A)
            tru_B, his_B, pre_B = tra_data(timestep, agent_B)

            # 如果没有预测，画空背景 + 文本提示
            if (pre_A is None) or (pre_B is None):
                ax.set_title(f"({chr(97 + i)}) Frame {timestep}\n(no prediction)")
                ax.set_xlim(0, 3840)
                ax.set_ylim(2160, 0)
                ax.grid(alpha=0.3)
                continue

            # 未来轨迹散点（带时间渐变颜色）
            T = min(pre_A.shape[1], pre_B.shape[1])
            reds  = plt.cm.Reds (np.linspace(1.0, 0.3, T))
            blues = plt.cm.Blues(np.linspace(1.0, 0.3, T))

            for j in range(T):
                ax.scatter(
                    to_px_x(pre_A[..., j, 0].ravel()),
                    to_px_y(pre_A[..., j, 1].ravel()),
                    s=3, alpha=0.2, c=[reds[j]]
                )
                ax.scatter(
                    to_px_x(pre_B[..., j, 0].ravel()),
                    to_px_y(pre_B[..., j, 1].ravel()),
                    s=3, alpha=0.2, c=[blues[j]]
                )

            # 历史轨迹 + 当前点（只在第一个子图收集图例）
            if i == 0:
                hA, = ax.plot(to_px_x(his_A[:, 0]), to_px_y(his_A[:, 1]),
                              color='#ff8066', alpha=1, linestyle='--', label=label_A)
                hB, = ax.plot(to_px_x(his_B[:, 0]), to_px_y(his_B[:, 1]),
                              color='#0081cf', alpha=1, linestyle='--', label=label_B)
                pA  = ax.scatter(to_px_x(his_A[-1, 0]), to_px_y(his_A[-1, 1]),
                                 color='#ff8066', alpha=0.7, label=label_A_1)
                pB  = ax.scatter(to_px_x(his_B[-1, 0]), to_px_y(his_B[-1, 1]),
                                 color='#0081cf', alpha=0.7, marker='*', label=label_B_1)
                legend_handles.extend([hA, hB, pA, pB])
                legend_labels.extend([label_A, label_B, label_A_1, label_B_1])
            else:
                ax.plot(to_px_x(his_A[:, 0]), to_px_y(his_A[:, 1]),
                        color='#ff8066', alpha=1, linestyle='--')
                ax.plot(to_px_x(his_B[:, 0]), to_px_y(his_B[:, 1]),
                        color='#0081cf', alpha=1, linestyle='--')
                ax.scatter(to_px_x(his_A[-1, 0]), to_px_y(his_A[-1, 1]),
                           color='#ff8066', alpha=0.7)
                ax.scatter(to_px_x(his_B[-1, 0]), to_px_y(his_B[-1, 1]),
                           color='#0081cf', alpha=0.7, marker='*')

            # 轴范围（可按需要裁剪）
            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
            ax.set_xlabel(f'({chr(97 + i)}) Frame {timestep}')

        except Exception as e:
            # 该子图出错，画空图并标注
            print(f"[warning] 绘制时间步 {timestep} 子图时出错: {e}")
            ax.clear()
            ax.set_title(f"({chr(97 + i)}) Frame {timestep}\n(plot error)")
            ax.axis('off')
            continue

    # ===== 6) 统一图例 =====
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='center left',
            ncol=4,
            bbox_to_anchor=(0.1, 1.05)
        )

    # ===== 7) 统一颜色柱子 =====
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes,
        orientation='vertical',
        fraction=0.046,
        pad=0.008
    )
    cbar.set_label(r'$\mathcal{R(\mathbf{x})}$')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_formatter))

    # ===== 8) 保存 / 显示 =====
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
    else:
        plt.show()




def picture_single_with_field_one_timestep(
    tra_data, gaussian_parameter,
    x1, x2, y1, y2,
    agent_A, agent_B, timestep,  # 单个时刻
    label_A, label_B, label_A_1, label_B_1,
    save_name=None,
    vmin=None, vmax=None,        # 可选：用于跨帧统一色标
    n_levels=10,
    show_tru=True,               # ✅ 新增：是否画 Tru
    label_A_tru="Future truth (A)",
    label_B_tru="Future truth (B)"
):
    """
    单时刻单图：
      - 背景：两智能体未来分布产生的场强等高线（jet）
      - 前景：历史轨迹 + 当前点 + 未来预测散点 + ✅未来真值轨迹(tru)
      - 右侧：colorbar
    """

    veh_name = [agent_A, agent_B]

    # ===== 1) 计算场强 =====
    try:
        mats = [two_agent_conflict_p(gaussian_parameter, timestep, idx, veh_name) for idx in range(16)]
        risk_field_raw = sum(mats)
        risk_field = replace_outliers_custom(risk_field_raw, lower_pct=99)  # 你原来的清洗方式
    except Exception as e:
        print(f"[error] 计算时间步 {timestep} 场强失败: {e}")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(f"Frame {timestep}\n(field error)")
        ax.axis("off")
        plt.show()
        return

    # ===== 2) 颜色范围 =====
    if vmin is None:
        vmin = np.nanmin(risk_field_raw)
    if vmax is None:
        vmax = np.nanmax(risk_field_raw)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # ===== 3) 坐标变换（按你原来的像素映射） =====
    sx = 3840.0 / (23.99 + 55.51)
    ox = 23.99 * sx
    sy = 2160.0 / (7.643 + 38.69)
    oy = 7.64 * sy

    X = x * sx + ox
    Y = y * sy + oy

    def to_px_x(arr): return arr * sx + ox
    def to_px_y(arr): return arr * sy + oy

    # ===== 4) 单图绘制 =====
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)

    try:
        contour = ax.contour(
            X, Y, risk_field,
            levels=n_levels,
            cmap="jet",
            norm=norm,
            linewidths=1.5
        )
        ax.clabel(contour, inline=True, fmt="%.2f")

        # --- 轨迹数据 ---
        tru_A, his_A, pre_A = tra_data(timestep, agent_A)
        tru_B, his_B, pre_B = tra_data(timestep, agent_B)

        # --- 预测散点 ---
        if (pre_A is None) or (pre_B is None):
            ax.set_title(f"Frame {timestep} (no prediction)")
        else:
            T = min(pre_A.shape[1], pre_B.shape[1])
            reds  = plt.cm.Reds (np.linspace(1.0, 0.3, T))
            blues = plt.cm.Blues(np.linspace(1.0, 0.3, T))

            for j in range(T):
                ax.scatter(
                    to_px_x(pre_A[..., j, 0].ravel()),
                    to_px_y(pre_A[..., j, 1].ravel()),
                    s=3, alpha=0.2, c=[reds[j]],
                    label="Pred (A)" if j == 0 else None
                )
                ax.scatter(
                    to_px_x(pre_B[..., j, 0].ravel()),
                    to_px_y(pre_B[..., j, 1].ravel()),
                    s=3, alpha=0.2, c=[blues[j]],
                    label="Pred (B)" if j == 0 else None
                )

        # ✅ --- Tru 真值未来轨迹（新增） ---
        if show_tru:
            if tru_A is not None and hasattr(tru_A, "shape") and tru_A.size:
                ax.scatter(
                    to_px_x(tru_A[:, 0]), to_px_y(tru_A[:, 1]),
                    s=28, alpha=0.8, marker="^", color="r",
                    label=label_A_tru
                )
            if tru_B is not None and hasattr(tru_B, "shape") and tru_B.size:
                ax.scatter(
                    to_px_x(tru_B[:, 0]), to_px_y(tru_B[:, 1]),
                    s=28, alpha=0.8, marker="D", color="b",
                    label=label_B_tru
                )

        # --- 历史轨迹 + 当前点 ---
        hA, = ax.plot(
            to_px_x(his_A[:, 0]), to_px_y(his_A[:, 1]),
            color="#ff8066", alpha=1, linestyle="--", label=label_A
        )
        hB, = ax.plot(
            to_px_x(his_B[:, 0]), to_px_y(his_B[:, 1]),
            color="#0081cf", alpha=1, linestyle="--", label=label_B
        )
        pA = ax.scatter(
            to_px_x(his_A[-1, 0]), to_px_y(his_A[-1, 1]),
            color="#ff8066", alpha=0.9, s=35, label=label_A_1
        )
        pB = ax.scatter(
            to_px_x(his_B[-1, 0]), to_px_y(his_B[-1, 1]),
            color="#0081cf", alpha=0.9, s=45, marker="*", label=label_B_1
        )

        # 图例去重（避免 Pred(A/B) 多次）
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l and l not in uniq:
                uniq[l] = h
        #ax.legend(uniq.values(), uniq.keys(), loc="upper left", frameon=False, fontsize=9)

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_xlabel(f"Frame {timestep}")

    except Exception as e:
        print(f"[error] 绘图失败 (frame={timestep}): {e}")
        ax.clear()
        ax.set_title(f"Frame {timestep}\n(plot error)")
        ax.axis("off")

    # ===== 5) colorbar =====
    sm = ScalarMappable(norm=norm, cmap="jet")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.02)
    cbar.set_label(r'$\mathcal{R}(\mathbf{x})$')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(decimal_formatter))

    # ===== 6) 保存/显示 =====
    if save_name is not None:
        plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.show()

    return fig, ax
    
    
    
    
    
    
    
    

def _to_numpy(x):
    """兼容 torch / numpy"""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _softmax(logits, axis=-1):
    logits = np.asarray(logits, dtype=float)
    m = np.max(logits, axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def gmm_weighted_mean_trajectory(mus, log_pis):
    """
    mus:      (H, K, 2)
    log_pis:  (H, K)
    return:   traj_mean (H, 2)
    """
    mus = _to_numpy(mus)
    log_pis = _to_numpy(log_pis)
    w = _softmax(log_pis, axis=1)                      # (H, K)
    traj = np.sum(w[:, :, None] * mus, axis=1)         # (H, 2)
    return traj

def segment_intersection(p, p2, q, q2, eps=1e-12):
    """
    线段 p->p2 与 q->q2 的相交检测
    返回 (hit, point, t, u)
      t: 交点在 p->p2 上的参数(0~1)
      u: 交点在 q->q2 上的参数(0~1)
    """
    p = np.asarray(p, dtype=float); p2 = np.asarray(p2, dtype=float)
    q = np.asarray(q, dtype=float); q2 = np.asarray(q2, dtype=float)

    r = p2 - p
    s = q2 - q

    def cross2(a, b):
        return a[0]*b[1] - a[1]*b[0]

    rxs = cross2(r, s)
    q_p = q - p

    # 平行或近似平行：这里简单认为无交点（需要处理共线重叠的话可以再扩展）
    if abs(rxs) < eps:
        return False, None, None, None

    t = cross2(q_p, s) / rxs
    u = cross2(q_p, r) / rxs

    if (0.0 - eps) <= t <= (1.0 + eps) and (0.0 - eps) <= u <= (1.0 + eps):
        pt = p + t * r
        # 裁剪到[0,1]避免数值误差
        t = float(np.clip(t, 0.0, 1.0))
        u = float(np.clip(u, 0.0, 1.0))
        return True, pt, t, u

    return False, None, None, None

def conflict_score_from_mean_intersection(gaussian_parameter,t, veh1, veh2):
    
    """
    输出：
      - 若两条未来均值轨迹存在交叉：返回 1 / (abs(idx1-idx2)*dt_step)
      - 若不存在：返回 0
    """
    dt_step=0.4    
    H=16
    mus1, sig1, logpis1, corrs1, _ = gaussian_parameter(t, veh1)
    mus2, sig2, logpis2, corrs2, _ = gaussian_parameter(t, veh2)

    traj1 = gmm_weighted_mean_trajectory(mus1, logpis1)  # (H,2)
    traj2 = gmm_weighted_mean_trajectory(mus2, logpis2)  # (H,2)

    H1 = min(H, traj1.shape[0])
    H2 = min(H, traj2.shape[0])
    traj1 = traj1[:H1]
    traj2 = traj2[:H2]

    best = None  # 存最小时间差的那个交点（更合理）

    # 遍历所有线段对，找交点
    for i in range(H1 - 1):
        p, p2 = traj1[i], traj1[i + 1]
        for j in range(H2 - 1):
            q, q2 = traj2[j], traj2[j + 1]
            hit, pt, ti, uj = segment_intersection(p, p2, q, q2)
            if not hit:
                continue

            # 交点对应的“连续时间索引”
            idx1_cont = i + ti
            idx2_cont = j + uj

            # 题目说“最近的时间步长索引”：取四舍五入到最近整数
            idx1 = int(np.clip(np.round(idx1_cont), 0, H1 - 1))
            idx2 = int(np.clip(np.round(idx2_cont), 0, H2 - 1))

            delta = abs(idx1 - idx2) * dt_step

            # 选择 delta 最小的交点（若多个交点）
            if best is None or delta < best[0]:
                best = (delta, idx1, idx2, pt)

    if best is None:
        return 0.0

    delta = best[0]
    if delta <= 0:
        # 同一步到达交点：理论上倒数为无穷大；你也可以改成一个很大的数或返回0
        return float("inf")

    return 1.0 / delta
