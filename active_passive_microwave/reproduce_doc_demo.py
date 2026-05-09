from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "active_passive_microwave_demo"


def density_to_dielectric(rho_g_cm3: np.ndarray) -> np.ndarray:
    """Simple empirical snow/firn dielectric model used for the demo."""
    return (1.0 + 0.845 * rho_g_cm3) ** 2


def make_profiles(depth_m: float = 4000.0, dz_m: float = 5.0) -> pd.DataFrame:
    z = np.arange(0.0, depth_m + dz_m, dz_m)

    mean_density = 0.35 + (0.917 - 0.35) * (1.0 - np.exp(-z / 650.0))
    layering = (
        0.025 * np.sin(2.0 * np.pi * z / 120.0) * np.exp(-z / 900.0)
        + 0.012 * np.sin(2.0 * np.pi * z / 37.0) * np.exp(-z / 350.0)
    )
    density = np.clip(mean_density + layering, 0.30, 0.917)
    dielectric = density_to_dielectric(density)

    temperature = 235.0 + 28.0 * (z / depth_m) ** 1.35

    return pd.DataFrame(
        {
            "depth_m": z,
            "density_g_cm3": density,
            "dielectric": dielectric,
            "temperature_k": temperature,
        }
    )


def passive_brightness_temperature(
    z: np.ndarray,
    temperature_k: np.ndarray,
    density_g_cm3: np.ndarray,
    frequency_ghz: float,
) -> tuple[float, np.ndarray]:
    dz = np.gradient(z)
    absorption = (1.0 / 1450.0) * (frequency_ghz / 1.413) ** 1.55
    absorption = absorption * (0.55 + 0.65 * density_g_cm3 / 0.917)
    optical_depth = np.cumsum(absorption * dz)
    weight = absorption * np.exp(-optical_depth)
    tb = np.trapz(temperature_k * weight, z) / np.trapz(weight, z)
    return float(tb), weight / np.max(weight)


def radar_echo_power(
    z: np.ndarray,
    dielectric: np.ndarray,
    density_g_cm3: np.ndarray,
    frequency_ghz: float,
) -> np.ndarray:
    dz = np.gradient(z)
    refractive_index = np.sqrt(dielectric)
    reflection = np.diff(refractive_index) / (refractive_index[:-1] + refractive_index[1:])
    reflection_power = reflection**2
    absorption = (1.0 / 2600.0) * (frequency_ghz / 0.435) ** 1.25
    absorption = absorption * (0.65 + 0.50 * density_g_cm3[:-1] / 0.917)
    two_way_loss = np.exp(-2.0 * np.cumsum(absorption * dz[:-1]))
    power = reflection_power * two_way_loss
    power = power / np.max(power)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def temperature_profile(z: np.ndarray, surface_k: float, basal_k: float) -> np.ndarray:
    return surface_k + (basal_k - surface_k) * (z / z.max()) ** 1.35


def radar_attenuation_proxy(z: np.ndarray, temperature_k: np.ndarray, density_g_cm3: np.ndarray) -> float:
    dz = np.gradient(z)
    temp_factor = np.exp((temperature_k - 250.0) / 22.0)
    density_factor = 0.6 + 0.5 * density_g_cm3 / 0.917
    return float(np.trapz(temp_factor * density_factor, z) / z.max())


def run_joint_inversion(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    z = profiles["depth_m"].to_numpy()
    rho = profiles["density_g_cm3"].to_numpy()
    t_true = profiles["temperature_k"].to_numpy()
    freqs = np.array([0.435, 0.75, 1.0, 1.413])

    tb_true = np.array([passive_brightness_temperature(z, t_true, rho, f)[0] for f in freqs])
    tb_obs = tb_true + rng.normal(0.0, 0.7, size=len(freqs))
    attenuation_true = radar_attenuation_proxy(z, t_true, rho)
    attenuation_obs = attenuation_true + rng.normal(0.0, 0.015)

    surface_grid = np.linspace(228.0, 242.0, 57)
    basal_grid = np.linspace(250.0, 270.0, 81)
    rows = []
    for surface in surface_grid:
        for basal in basal_grid:
            t_model = temperature_profile(z, surface, basal)
            tb_model = np.array([passive_brightness_temperature(z, t_model, rho, f)[0] for f in freqs])
            attenuation_model = radar_attenuation_proxy(z, t_model, rho)
            passive_cost = np.sum(((tb_obs - tb_model) / 0.7) ** 2)
            radar_cost = ((attenuation_obs - attenuation_model) / 0.015) ** 2
            rows.append(
                {
                    "surface_k": surface,
                    "basal_k": basal,
                    "passive_cost": passive_cost,
                    "joint_cost": passive_cost + radar_cost,
                }
            )

    posterior = pd.DataFrame(rows)
    passive_map = posterior.loc[posterior["passive_cost"].idxmin()].copy()
    joint_map = posterior.loc[posterior["joint_cost"].idxmin()].copy()
    estimates = pd.DataFrame(
        [
            {
                "method": "true",
                "surface_k": float(t_true[0]),
                "basal_k": float(t_true[-1]),
            },
            {
                "method": "passive_only",
                "surface_k": passive_map["surface_k"],
                "basal_k": passive_map["basal_k"],
            },
            {
                "method": "passive_plus_radar",
                "surface_k": joint_map["surface_k"],
                "basal_k": joint_map["basal_k"],
            },
        ]
    )
    tb_table = pd.DataFrame(
        {
            "frequency_ghz": freqs,
            "tb_true_k": tb_true,
            "tb_observed_k": tb_obs,
        }
    )
    return estimates, tb_table


def save_figures(profiles: pd.DataFrame, estimates: pd.DataFrame, tb_table: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    z = profiles["depth_m"].to_numpy()
    rho = profiles["density_g_cm3"].to_numpy()
    eps = profiles["dielectric"].to_numpy()
    temp = profiles["temperature_k"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    axes[0].plot(rho, z)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Density (g/cm3)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Firn/ice density")
    axes[1].plot(eps, z, color="tab:orange")
    axes[1].set_xlabel("Relative dielectric constant")
    axes[1].set_title("Dielectric from density")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "density_dielectric_profile.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for freq in tb_table["frequency_ghz"]:
        _, weight = passive_brightness_temperature(z, temp, rho, float(freq))
        ax.plot(weight, z, label=f"{freq:.3g} GHz")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized contribution")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Passive microwave depth response")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "passive_depth_response.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for freq in [0.435, 1.0]:
        power_db = radar_echo_power(z, eps, rho, freq)
        ax.plot(power_db, z[:-1], label=f"{freq:.3g} GHz")
    ax.invert_yaxis()
    ax.set_xlabel("Normalized echo power (dB)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Active radar echo from dielectric layering")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "radar_echo_profile.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(temp, z, label="true")
    for _, row in estimates[estimates["method"] != "true"].iterrows():
        t_est = temperature_profile(z, float(row["surface_k"]), float(row["basal_k"]))
        ax.plot(t_est, z, "--", label=row["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Toy temperature-profile inversion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "temperature_inversion.png", dpi=200)
    plt.close(fig)


def write_report(estimates: pd.DataFrame, tb_table: pd.DataFrame) -> None:
    estimates_path = REPORT_DIR / "temperature_inversion_estimates.csv"
    tb_path = REPORT_DIR / "passive_tb_observations.csv"
    estimates.to_csv(estimates_path, index=False)
    tb_table.to_csv(tb_path, index=False)

    report_path = REPORT_DIR / "reproduction_notes.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# 主被动微波冰盖探测简化复现\n\n")
        handle.write("这个 demo 复现的是文档里的机理链路，不是完整 CLEM 多次散射模型。\n\n")
        handle.write("## 复现内容\n\n")
        handle.write("1. 构造 0-4000 m 冰盖密度和温度廓线。\n")
        handle.write("2. 用经验关系从密度计算相对介电常数。\n")
        handle.write("3. 用简化深度权重模拟 P-L 波段被动亮温贡献。\n")
        handle.write("4. 用介电常数层间变化模拟主动雷达回波剖面。\n")
        handle.write("5. 用被动亮温和雷达衰减约束做一个玩具版温度廓线反演。\n\n")
        handle.write("## 结果文件\n\n")
        for name in [
            "density_dielectric_profile.png",
            "passive_depth_response.png",
            "radar_echo_profile.png",
            "temperature_inversion.png",
        ]:
            handle.write(f"![{name}]({name})\n\n")
        handle.write("## 和当前海冰雪深数据的关系\n\n")
        handle.write(
            "你当前的数据主要是海冰/积雪厚度表格样本和 18-89 GHz 亮温，"
            "可以用于被动微波雪深反演 baseline；"
            "但文档讨论的是 P-L 波段冰盖深层主被动联合探测，"
            "需要雷达回波、低频辐射亮温、冰芯/钻孔等数据。"
            "所以当前数据不能直接完整复现这篇文档，只能作为被动亮温反演部分的类比。\n"
        )


def main() -> None:
    profiles = make_profiles()
    estimates, tb_table = run_joint_inversion(profiles)
    save_figures(profiles, estimates, tb_table)
    write_report(estimates, tb_table)
    print(f"Saved demo outputs to: {REPORT_DIR}")
    print(estimates)


if __name__ == "__main__":
    main()
