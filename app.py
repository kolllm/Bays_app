import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# 设置支持中文的字体列表（按优先级排列），并解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面基础配置
st.set_page_config(page_title="贝叶斯理论交互演示", layout="wide")
st.title("🧠 形象理解贝叶斯理论 (Bayesian Inference)")
st.markdown("核心思想：**用数据更新我们的信念**。先验概率 + 新证据 = 后验概率")

# --- 第一部分：公式拆解 ---
st.header("1. 贝叶斯公式拆解")
col1, col2 = st.columns([1, 1])

with col1:
    st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
    st.markdown("""
    - **$P(A|B)$ (后验概率)**：在观察到证据 $B$ 后，假设 $A$ 成立的概率。
    - **$P(A)$ (先验概率)**：在看到证据前，我们对 $A$ 的初始信念。
    - **$P(B|A)$ (似然度)**：如果假设 $A$ 是真的，观察到证据 $B$ 的概率。
    - **$P(B)$ (边际概率)**：无论假设是否成立，观察到证据 $B$ 的总概率。
    """)

with col2:
    st.info("💡 **直觉理解**：贝叶斯定理就是一个‘修正器’。它告诉我们，当出现新的线索时，我们该如何合理地调整自己最初的猜测。")

st.divider()

# --- 第二部分：经典应用 - 医学检测与假阳性悖论 ---
st.header("2. 经典应用：医学检测的‘假阳性悖论’")
st.caption("为什么高准确率不等于高患病率？让我们通过具体人数来算一算！")

c1, c2, c3 = st.columns(3)
with c1:
    prior = st.slider("疾病基础发病率 (先验概率)", 0.001, 0.1, 0.01, 0.001, format="%.3f")
with c2:
    sensitivity = st.slider("检测敏感度 (真阳性率)", 0.5, 1.0, 0.99, 0.01)
with c3:
    false_positive = st.slider("假阳性率 (健康人误报)", 0.01, 0.1, 0.05, 0.01)

# 为了方便学生理解，我们将概率转化为 10万人的绝对数量
population = 100000
sick_people = int(population * prior)
healthy_people = population - sick_people

true_positives = int(sick_people * sensitivity)
false_positives = int(healthy_people * false_positive)
total_positives = true_positives + false_positives

posterior = true_positives / total_positives if total_positives > 0 else 0

# 显示结果与详细拆解
res_c1, res_c2 = st.columns([1, 2])
with res_c1:
    st.metric("检测为阳性时，实际患病的概率 (后验概率)", f"{posterior:.2%}")
    if posterior < 0.5 and sensitivity > 0.9:
        st.warning("⚠️ **反直觉现象**：尽管检测非常准确，但由于疾病太罕见，绝大多数阳性结果其实都是健康的假警报！这就是贝叶斯定理揭示的真相。")

with res_c2:
    st.subheader("🔍 详细推导过程 (以 10万人 为例)")
    st.markdown(f"""
    **第一步：计算真实患者产生的阳性数**  
    {sick_people:,} 名真实患者 × {sensitivity * 100}% 敏感度 = **{true_positives:,} 人** (真阳性)

    **第二步：计算健康人群产生的阳性数**  
    {healthy_people:,} 名健康人 × {false_positive * 100}% 误诊率 = **{false_positives:,} 人** (假阳性)

    **第三步：计算后验概率**  
    在所有 **{total_positives:,} 名** 拿到阳性报告的人中，真正患病的只有 {true_positives:,} 人。  
    `实际患病概率 = {true_positives:,} / {total_positives:,} ≈ {posterior:.2%}`
    """)

st.divider()

# --- 第三部分：动态更新可视化 (从先验到后验) ---
st.header("3. 动态信念更新：抛硬币实验")
st.markdown("假设我们有一枚硬币，想知道它正面朝上的真实概率。随着抛掷次数增加，我们的信念分布会越来越集中。")

s_c1, s_c2 = st.columns([1, 2])

with s_c1:
    true_prob = st.slider("设定硬币真实的正面概率", 0.0, 1.0, 0.5, 0.01)
    n_flips = st.slider("模拟抛掷次数", 10, 1000, 100, 10)

    # 生成模拟数据
    np.random.seed(42)
    heads = np.sum(np.random.rand(n_flips) < true_prob)
    tails = n_flips - heads

    # 设置 Beta 分布参数 (先验设为均匀分布 a=1, b=1)
    a_prior, b_prior = 1, 1
    a_post = a_prior + heads
    b_post = b_prior + tails

with s_c2:
    x = np.linspace(0, 1, 1000)
    prior_pdf = beta.pdf(x, a_prior, b_prior)
    posterior_pdf = beta.pdf(x, a_post, b_post)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, prior_pdf, 'b--', label=f'先验分布 (Prior)', linewidth=2)
    ax.fill_between(x, prior_pdf, alpha=0.1, color='blue')

    ax.plot(x, posterior_pdf, 'r-', label=f'后验分布 (Posterior)', linewidth=2)
    ax.fill_between(x, posterior_pdf, alpha=0.2, color='red')

    ax.axvline(true_prob, color='green', linestyle=':', label=f'真实概率 ({true_prob})', linewidth=2)

    ax.set_xlabel("正面朝上的概率", fontsize=12)
    ax.set_ylabel("概率密度", fontsize=12)
    ax.set_title(f"抛掷 {n_flips} 次，观察到 {heads} 次正面，{tails} 次反面", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.success(f"📊 **数据说话**：经过 {n_flips} 次实验，我们的信念（红色区域）已经强烈收敛到了真实概率附近！这就是贝叶斯学习的魅力。")

    st.markdown(f"""
    **🔄 更新机制解析：**
    - **先验信念**：一开始我们一无所知，认为任何概率都有可能（蓝色平缓曲线）。
    - **收集证据**：我们抛了 `{n_flips}` 次硬币，得到了 `{heads}` 正 `{tails}` 反。
    - **后验更新**：数学上，我们将这些数据加到了 Beta 分布的参数中。观察红色的曲线，它变得非常尖锐，说明我们现在的结论不再盲目，而是高度确信真实概率就在绿色虚线附近！
    """)

st.divider()

# --- 第四部分：进阶思考 - 为什么不能只用数据拟合？ ---
st.header("4. 进阶思考：为什么不能只用数据拟合？")
st.markdown("让我们来看看，当数据量极少时，纯数据拟合与贝叶斯推断的巨大差异！")

c_f1, c_f2 = st.columns([1, 1])

with c_f1:
    # 模拟极小样本情况
    small_flips = st.slider("设定极小的抛掷次数", 1, 20, 5, 1)
    np.random.seed(42)
    small_heads = np.sum(np.random.rand(small_flips) < true_prob)

    # 频率学派的结果 (最大似然估计 MLE)
    freq_prob = small_heads / small_flips if small_flips > 0 else 0

    st.subheader("📊 频率学派 (纯数据拟合)")
    st.metric("计算出的正面概率", f"{freq_prob:.2%}")
    st.caption(f"逻辑：{small_flips} 次里有 {small_heads} 次正面，所以概率是 {freq_prob:.2%}。")
    st.info("⚠️ **痛点**：如果只抛了 3 次出现 2 次正面，它会告诉你概率是 66.7%！这显然违背了我们认为‘硬币通常是公平的’这一常识。")

with c_f2:
    # 贝叶斯学派的结果
    a_post_small = 1 + small_heads
    b_post_small = 1 + (small_flips - small_heads)
    bayes_mean = a_post_small / (a_post_small + b_post_small)

    st.subheader("🧠 贝叶斯学派 (结合常识的动态更新)")
    st.metric("更新后的期望概率", f"{bayes_mean:.2%}")
    st.caption(f"逻辑：虽然出现了 {small_heads} 次正面，但因为我们初始一无所知(先验)，少量的数据不足以让结论产生剧烈偏移。")
    st.success("✅ **优势**：贝叶斯不仅给出了一个合理的折中值，还保留了极大的不确定性（宽容度）。随着后续数据增多，它才会慢慢向真实概率靠拢。")

st.warning("💡 **总结**：纯数据拟合适合处理工厂流水线这种**条件稳定、数据海量**的场景；而贝叶斯理论更适合现实世界中充满未知、需要结合历史经验和实时线索来**动态决策**的复杂环境。")