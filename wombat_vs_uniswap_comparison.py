import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output folder
output_dir = "wombat_uniswap_comparison"
os.makedirs(output_dir, exist_ok=True)

# Load data
with open("data_export.json") as f:
    wombat_data = json.load(f)

with open("final_uniswap_realistic_simulation.json") as f:
    uniswap_data = json.load(f)["UniswapV2"]

# Compute average coverage ratios for Wombat
avg_coverage_ratios = {token: float(np.mean(vals)) for token, vals in wombat_data["coverage_ratio"].items()}

# Prepare comparison summary
comparison = {
    "Liquidity Fragmentation": {
        "Explanation": "Liquidity fragmentation refers to how capital is distributed across the pool's price range. Wombat uses dynamic rebalancing to maintain even coverage, while Uniswap splits liquidity across price ranges due to x*y=k model.",
        "Wombat": {
            "Coverage Ratios (Mean)": avg_coverage_ratios,
            "Dynamic Rebalancing": True,
            "Fragmentation Control": "Adaptive via coverage ratio + repegging",
            "Liquidity Range": "Wide, dynamically adjusted"
        },
        "Uniswap": {
            "Coverage Ratios": {"BTC": 1.0, "ETH": 1.0, "USDT": 1.0},
            "Dynamic Rebalancing": False,
            "Fragmentation Control": "None (constant product formula)",
            "Liquidity Range": "Fixed range, no self-adjustment"
        },
        "Winner": "Wombat – better capital spread and dynamic response to price"
    },
    "Computation Cost (Gas / Smart Contract Load)": {
        "Explanation": "Computation cost includes operation count (proxy for gas) and how much state data is fetched/updated in each transaction. Wombat’s dynamic operations increase cost but provide tighter peg.",
        "Wombat": {
            "Operation Counters": wombat_data["operation_counter"],
            "State Data Accessed": "Coverage ratios, fee multipliers, price tweaks, rebase targets",
            "Complexity": "High – several real-time adjustments per swap",
            "Gas Optimization": "Partially optimized, multi-step"
        },
        "Uniswap": {
            "Operation Counters": {
                "swap_quote": "~1",
                "calculate_fee": "0.3% fee per swap",
                "tweak_price": "N/A",
                "repeg": "Not supported"
            },
            "State Data Accessed": "Token reserves (x, y), swap fee",
            "Complexity": "Low – constant function only",
            "Gas Optimization": "Highly gas efficient, simple swap logic"
        },
        "Winner": "Uniswap – much simpler and gas efficient"
    },
    "Capital Efficiency": {
        "Explanation": "Capital efficiency measures how well liquidity earns yield with minimal slippage and loss. Wombat mitigates impermanent loss via repeg and smoother pricing.",
        "Wombat": {
            "LP APY": wombat_data["apy_lp"],
            "Impermanent Loss Mitigation": "Yes – via dynamic coverage, smoothing, repeg",
            "Slippage": "Low – Oracle pegged, smoother curve"
        },
        "Uniswap": {
            "LP APY": {
                "BTC": uniswap_data["BTC"]["Relative LP Gain %"],
                "ETH": uniswap_data["ETH"]["Relative LP Gain %"],
                "USDT": uniswap_data["USDT"]["Relative LP Gain %"]
            },
            "Impermanent Loss %": {
                "BTC": uniswap_data["BTC"]["Impermanent Loss %"],
                "ETH": uniswap_data["ETH"]["Impermanent Loss %"],
                "USDT": uniswap_data["USDT"]["Impermanent Loss %"]
            },
            "Slippage": {
                "BTC": uniswap_data["BTC"]["Average Slippage"],
                "ETH": uniswap_data["ETH"]["Average Slippage"],
                "USDT": uniswap_data["USDT"]["Average Slippage"]
            }
        },
        "Winner": "Wombat – better IL protection and lower slippage overall"
    }
}

# Save comparison JSON
comparison_path = os.path.join(output_dir, "wombat_vs_uniswap_comparison.json")
with open(comparison_path, "w") as f:
    json.dump(comparison, f, indent=2)

# Visualization 1: Uniswap LP vs HODL
labels = []
values = []
colors = []

for token in ["BTC", "ETH", "USDT"]:
    if token in uniswap_data:
        labels.extend([f'{token} LP', f'{token} HODL'])
        values.extend([
            uniswap_data[token]["LP Final Value"],
            uniswap_data[token]["HODL Final Value"]
        ])
        colors.extend(['blue', 'gray'])

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=colors)
plt.title("Uniswap LP vs HODL Final Values")
plt.ylabel("USDT Value")
plt.tight_layout()
plot1_path = os.path.join(output_dir, "uniswap_lp_vs_hodl.png")
plt.savefig(plot1_path)
plt.close()

# Visualization 2: Wombat Coverage Ratios
tokens = list(avg_coverage_ratios.keys())
coverage_vals = list(avg_coverage_ratios.values())

plt.figure(figsize=(8, 5))
plt.bar(tokens, coverage_vals, color='purple')
plt.title("Wombat Average Coverage Ratios")
plt.ylabel("Average Coverage Ratio")
plt.tight_layout()
plot2_path = os.path.join(output_dir, "wombat_coverage_ratios.png")
plt.savefig(plot2_path)
plt.close()

# Visualization 3: Wombat LP APY
apy_tokens = list(wombat_data["apy_lp"].keys())
apy_vals = list(wombat_data["apy_lp"].values())

plt.figure(figsize=(8, 5))
plt.bar(apy_tokens, apy_vals, color='orange')
plt.title("Wombat LP APY per Token")
plt.ylabel("APY")
plt.tight_layout()
plot3_path = os.path.join(output_dir, "wombat_lp_apy.png")
plt.savefig(plot3_path)
plt.close()

output_dir, comparison_path, plot1_path, plot2_path, plot3_path
