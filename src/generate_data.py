import os
import numpy as np
import pandas as pd

def generate_telemetry(num_timesteps: int = 10000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    
    # 1. Base time axis (simulating readings every 1 minute)
    time_steps = np.arange(num_timesteps)
    
    # 2. Baseline cyclical behavior (e.g., daily cycle with 1440 mins/day)
    diurnal_cycle = 20 * np.sin(2 * np.pi * time_steps / 1440)
    
    # 3. CPU Utilization (%): Baseline ~35% + diurnal variation + Gaussian jitter
    cpu_noise = np.random.normal(0, 3, num_timesteps)
    cpu = 35 + diurnal_cycle + cpu_noise
    
    # 4. Memory Utilization (%): Baseline ~45% + slight correlation with traffic
    mem_noise = np.random.normal(0, 1.5, num_timesteps)
    mem = 45 + (diurnal_cycle * 0.3) + mem_noise
    
    # 5. Network Ingress (MB/s): Traffic wave + jitter
    net_noise = np.random.normal(0, 2, num_timesteps)
    net_in = 50 + (diurnal_cycle * 0.8) + net_noise
    net_in = np.clip(net_in, 0, None)
    
    # 6. Labels: 0 = Normal, 1 = Anomaly
    labels = np.zeros(num_timesteps, dtype=int)
    
    # --- Inject Realistic Failure Modes ---
    
    # Anomaly 1: Memory Leak (Timesteps 2500 -> 3200)
    # Gradual accumulation without releasing
    leak_len = 700
    leak_slope = np.linspace(0, 48, leak_len)
    mem[2500:3200] += leak_slope
    labels[3000:3200] = 1  # Flag the critical threshold
    
    # Anomaly 2: Traffic Surge / DDoS Attack (Timesteps 6000 -> 6150)
    # Massive CPU spike accompanied by network saturation
    cpu[6000:6150] += 50
    net_in[6000:6150] += 80
    labels[6000:6150] = 1
    
    # Anomaly 3: Thread Deadlock / CPU Throttle (Timesteps 8200 -> 8350)
    # CPU pinned near 100% while network collapses to near zero
    cpu[8200:8350] = 98 + np.random.normal(0, 0.5, 150)
    net_in[8200:8350] = 2 + np.random.normal(0, 0.5, 150)
    labels[8200:8350] = 1

    # Keep metrics within physically viable bounds (0% to 100% for CPU/Mem)
    cpu = np.clip(cpu, 1.0, 100.0)
    mem = np.clip(mem, 1.0, 100.0)
    
    df = pd.DataFrame({
        "timestamp": time_steps,
        "cpu_pct": np.round(cpu, 2),
        "mem_pct": np.round(mem, 2),
        "net_in_mbps": np.round(net_in, 2),
        "is_anomaly": labels
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_telemetry(num_timesteps=12000)
    output_path = os.path.join("data", "server_metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at '{output_path}' with {len(df)} records.")