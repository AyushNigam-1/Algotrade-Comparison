import json
import pandas as pd
import matplotlib.pyplot as plt

price_data = pd.read_csv("BTC_ETH_price.csv")

drop_btc_eth = price_data.dropna()

btc_price_cex = drop_btc_eth["BTC"].tolist()
eth_price_cex = drop_btc_eth["ETH"].tolist()

# The name of your JSON file
filename = "data_export.json"

# Open the file and read the JSON data
with open(filename, "r") as file:
    variables = json.load(file)

# Now `data` is a Python dictionary that contains the data from the JSON file
# Assuming 'variables' is a dictionary obtained from json.loads() or similar.
btc_apy = variables.get("btc_apy", None)
eth_apy = variables.get("eth_apy", None)
usdt_apy = variables.get("usdt_apy", None)
average_aggregated_value = variables.get("average_aggregated_value", None)
impermanent_loss = variables.get("impermanent_loss", None)
impermanent_loss_percent = variables.get("impermanent_loss_percent", None)
uniswapv2_impermanent_loss_percent = variables.get(
    "uniswapv2_impermanent_loss_percent", None
)
btc_amm_hourly = variables.get("btc_amm_hourly", None)
btc_oracle_hourly = variables.get("btc_oracle_hourly", None)
btc_scale_hourly = variables.get("btc_scale_hourly", None)
btc_scale_new_hourly = variables.get("btc_scale_new_hourly", None)
btc_fee_rate = variables.get("btc_fee_rate", None)
btc_sigma = variables.get("btc_sigma", None)
btc_fee_sigma_u_records = variables.get("btc_fee_sigma_u_records", None)
btc_utilization = variables.get("btc_utilization", None)
btc_cov_ratio = variables.get("btc_cov_ratio", None)
eth_cov_ratio = variables.get("eth_cov_ratio", None)
usdt_cov_ratio = variables.get("usdt_cov_ratio", None)
eth_fee_rate = variables.get("eth_fee_rate", None)
usdt_fee_rate = variables.get("usdt_fee_rate", None)
hodl_value = variables.get("hodl_value", None)
lp_value = variables.get("lp_value", None)
btc_price_cex_trimmed = variables.get("btc_price_cex_trimmed", None)
eth_price_cex_trimmed = variables.get("eth_price_cex_trimmed", None)
eth_amm_hourly = variables.get("eth_amm_hourly", None)
eth_oracle_hourly = variables.get("eth_oracle_hourly", None)
eth_scale_hourly = variables.get("eth_scale_hourly", None)
eth_scale_new_hourly = variables.get("eth_scale_new_hourly", None)
r_star_repeg = variables.get("r_star_repeg", None)
ema_t_dict = variables.get("ema_t_dict", None)

graph = True
if graph == True:
    # Assuming BTC_price_scale_records and BTC_price_oracle_records are lists containing the data points
    # Create an array of time_steps (assuming the records represent values at each time step)

    # Reduce the dimension to only showing the closing day
    interval = 24
    hourly = False

    time_steps = range(len(btc_amm_hourly))

    # Plot the data and set labels
    plt.plot(time_steps, btc_scale_hourly, label="BTC.price_scale", color="purple")
    plt.plot(
        time_steps,
        btc_scale_new_hourly,
        label="BTC.price_new_scale",
        color="purple",
        linestyle="dashed",
    )
    plt.plot(time_steps, btc_oracle_hourly, label="BTC.price_oracle", color="red")
    plt.plot(time_steps, btc_amm_hourly, label="BTC.price_last", color="grey")

    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("BTC price")
    plt.title("AMM Hourly BTC price_scale/oracle/last")
    plt.grid(True)
    plt.legend(loc="lower left")

    # Show the plot
    plt.savefig("autosave_amm_repeg.png")

    plt.show()
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    # Create an array of time_steps (assuming the records represent values at each time step)
    time_steps = range(len(impermanent_loss))

    # Create a figure and axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot impermanent_loss on the left y-axis
    ax1.plot(time_steps, impermanent_loss, label="Impermanent Loss", color="blue")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Impermanent Loss", color="blue")

    # Create a second y-axis on the right side
    ax2 = ax1.twinx()

    # Plot impermanent_loss_percent on the right y-axis
    ax2.plot(
        time_steps, impermanent_loss_percent, label="Impermanent Loss %", color="green"
    )
    ax2.set_ylabel("Impermanent Loss %", color="green")

    # Set title and legend
    plt.title("Impermanent Loss and Impermanent Loss % Over Time")
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")

    # Show the plot
    plt.grid(True)
    plt.savefig("autosave_IL.png")
    plt.show()
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    # Create an array of time_steps (assuming the records represent values at each time step)
    time_steps = range(len(impermanent_loss_percent))

    # Plot the data and set labels
    plt.plot(
        time_steps,
        impermanent_loss_percent,
        label="Wombat Impermanent Loss",
        color="purple",
    )

    plt.plot(
        time_steps,
        uniswapv2_impermanent_loss_percent,
        label="Uniswap v2 Impermanent Loss",
        color="red",
    )

    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("Impermanent loss %")
    plt.title("Wombat v.s. Uniswap v2 Impermanent Loss")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig("autosave_wombat_uni.png")
    plt.show()

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    # Create an array of time_steps (assuming the records represent values at each time step)
    time_steps = range(len(impermanent_loss_percent))

    # Plot the data and set labels
    plt.plot(
        time_steps,
        impermanent_loss_percent,
        label="Wombat Impermanent Loss",
        color="purple",
    )

    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("Impermanent loss %")
    plt.title("Wombat Impermanent Loss with fee")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.savefig("autosave_wombat_IL.png")
    plt.show()

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    # Plot ETH repegging trajectory

    import matplotlib.pyplot as plt

    # Assuming BTC_price_scale_records and BTC_price_oracle_records are lists containing the data points
    # Create an array of time_steps (assuming the records represent values at each time step)
    time_steps = range(len(btc_amm_hourly))

    size = 1  # min = 1 with 8741
    btc_price_cex_trimmed = btc_price_cex[::size]
    eth_price_cex_trimmed = eth_price_cex[::size]

    # Plot the data and set labels
    plt.plot(time_steps, btc_amm_hourly, label="BTC.price_last", color="purple")
    plt.plot(
        time_steps,
        btc_price_cex_trimmed,
        label="BTC CEX",
        color="red",
        linestyle="dashed",
    )

    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("BTC price")
    plt.title("BTC CEX v.s. BTC.price_last")
    plt.grid(True)
    plt.legend(loc="upper left")

    # # Show the plot
    plt.show()

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################

    # Assuming BTC.h_rate_records, BTC.sigma_T_records, BTC.utilization_T_records, and BTC.imbalance_records are lists containing the data points
    # Create an array of time_steps (assuming the records represent values at each time step)
    time_steps = range(len(btc_fee_rate))

    # Create a figure with three subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # Plot BTC.h_rate_records on the left y-axis of all subplots
    ax1.plot(time_steps, btc_fee_rate, label="BTC fee", color="purple")
    ax2.plot(time_steps, btc_fee_rate, label="BTC fee", color="purple")
    # ax3.plot(time_steps, btc_fee_rate, label="BTC fee", color="purple")

    # Set common labels and title
    fig.text(0.5, 0.04, "Time Step (hours)", ha="center")
    fig.text(0.04, 0.5, "BTC fee", va="center", rotation="vertical")
    fig.suptitle("BTC fee and Other Records")

    # Plot BTC.sigma_T_records on the right y-axis of the first subplot
    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_steps, btc_sigma, label="BTC sigma_T", color="blue")
    ax1_twin.set_ylabel("BTC sigma_T", color="blue")
    ax1_twin.grid(True)
    # Plot BTC.imbalance_records on the right y-axis of the second subplot
    # ax2.plot(time_steps, BTC.cov_ratio_records, label='BTC.cor_ratio_records', color='green')
    ax2_twin = ax2.twinx()

    ax2_twin.plot(time_steps, btc_cov_ratio, label="BTC cov", color="orange")
    ax2_twin.plot(time_steps, usdt_cov_ratio, label="USDT cov", color="green")

    ax2_twin.set_ylabel("BTC-USDT imbalance", color="orange")
    ax2_twin.grid(True)
    ax2.legend()

    # Show legends for each subplot
    ax1_twin.legend(loc="upper left")
    ax2_twin.legend(loc="upper left")
    # ax3_twin.legend(loc="upper left")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("autosave_fee_para.png")
    # Show the plot
    plt.show()

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    time_steps_btc = range(len(btc_fee_rate))
    time_steps_eth = range(len(eth_fee_rate))
    time_steps_usdt = range(len(usdt_fee_rate))

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    # Plot BTC.h_rate_records on the left y-axis of all subplots
    ax1.plot(time_steps_btc, btc_fee_rate, label="BTC h_rate", color="purple")
    ax2.plot(time_steps_eth, eth_fee_rate, label="ETH h_rate", color="purple")
    ax3.plot(time_steps_usdt, usdt_fee_rate, label="USDT h_rate", color="purple")

    # Set common labels and title
    fig.text(0.5, 0.04, "Time Step (hours)", ha="center")
    fig.text(0.04, 0.5, "h_rate", va="center", rotation="vertical")
    fig.suptitle("h_rate and Other Records")

    # Plot BTC.sigma_T_records on the right y-axis of the first subplot
    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        time_steps_btc,
        btc_cov_ratio,
        label="BTC cov ratio",
        color="blue",
        linestyle="dashed",
    )
    ax1_twin.set_ylabel("BTC cov_ratio", color="blue")
    ax1_twin.grid(True)
    # Plot BTC.imbalance_records on the right y-axis of the second subplot

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        time_steps_eth,
        eth_cov_ratio,
        label="ETH cov ratio",
        color="blue",
        linestyle="dashed",
    )
    ax2_twin.set_ylabel("ETH cov_ratio", color="blue")
    ax2_twin.grid(True)
    ax2.legend()

    # Plot BTC.utilization_T_records on the right y-axis of the third subplot
    ax3_twin = ax3.twinx()
    ax3_twin.plot(
        time_steps_usdt,
        usdt_cov_ratio,
        label="USDT cov ratio",
        color="blue",
        linestyle="dashed",
    )
    ax3_twin.set_ylabel("USDT cov_ratio", color="blue")
    ax3_twin.grid(True)
    ax2.legend()

    # Show legends for each subplot
    ax1_twin.legend(loc="upper left")
    ax2_twin.legend(loc="upper left")
    ax3_twin.legend(loc="upper left")

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("autosave_fee_cov.png")
    # Show the plot
    plt.show()

    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    ######################################################################################################
    time_steps = range(len(btc_cov_ratio))

    # Plot the data and set labels
    plt.plot(
        time_steps,
        btc_cov_ratio,
        label="BTC cov_ratio ",
        color="purple",
        linestyle="dashed",
    )

    plt.plot(
        time_steps,
        usdt_cov_ratio,
        label="USDT cov_ratio ",
        color="green",
        linestyle="dashed",
    )
    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("coverage ratio")
    plt.title("Price change")
    plt.grid(True)
    plt.legend(loc="upper left")

    # Create a second y-axis on the right side
    ax2 = plt.twinx()

    size = 1  # min = 1 with 8741
    btc_price_cex_trimmed = btc_price_cex[::size]

    btc_percent_change = [
        (btc_price_cex_trimmed[value] - btc_price_cex_trimmed[0])
        / btc_price_cex_trimmed[0]
        * 100
        for value in range(len(btc_price_cex_trimmed))
    ]

    # Plot impermanent_loss_percent on the right y-axis
    plt.plot(time_steps, btc_percent_change, label="BTC price change", color="black")

    ax2.set_ylabel("BTC price change", color="black")

    # Set title and legend
    plt.legend(loc="lower left")
    ax2.legend(loc="lower right")

    # Show the plot
    plt.grid(True)
    plt.show()

    ################################################################################################################################

    btc_percent_change_trailing = [
        (btc_price_cex_trimmed[value] - btc_price_cex_trimmed[value - 24])
        / btc_price_cex_trimmed[0]
        * 100
        for value in range(len(btc_price_cex_trimmed))
    ]

    time_steps = range(len(btc_sigma))

    # Plot the data and set labels
    plt.plot(
        time_steps,
        btc_percent_change_trailing,
        label="BTC CEX",
        color="black",
        linestyle="dashed",
    )

    # Set labels and title
    plt.xlabel("Time Step(hours)")
    plt.ylabel("BTC price")
    plt.title("Sigma and price")
    plt.grid(True)
    plt.legend(loc="upper left")

    # Create a second y-axis on the right side
    ax2 = plt.twinx()

    # Plot impermanent_loss_percent on the right y-axis
    ax2.plot(
        time_steps,
        btc_sigma,
        label="sigma_T ",
        color="blue",
    )

    ax2.set_ylabel("sigma")

    # Set title and legend
    plt.legend(loc="lower left")
    ax2.legend(loc="lower right")

    # Show the plot
    plt.grid(True)
    plt.savefig("autosave_fee_sigma_btc.png")
    plt.show()
