import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import json
import ContractVolatile as trinity

price_data = pd.read_csv("BTC_ETH_price.csv")

drop_btc_eth = price_data.dropna()

btc_price_cex = drop_btc_eth["BTC"].tolist()
eth_price_cex = drop_btc_eth["ETH"].tolist()

start_time = time.time()

# POOL setup, Mary deposit as the protocol, Another external LP as Others and Peter act as arbitrague and swap around.
# The goal of historical stimuation is to observe the Imperanent loss of "Others".
# Create the Pool
Trinity = trinity.Pool()

# Create the Users
Peter = trinity.Users("Peter")  # Peter is the user
Mary = trinity.Users("Mary")  # Mary is the arbitrageur
Others = trinity.Users("Others")  # This is another LP

"""
Block begin user inputs.
Change all inputs, including tokens, parameters setup here
"""

# Create all the tokens here
BTC = trinity.Tokens("BTC")  # *** To be modified ***
ETH = trinity.Tokens("ETH")  # *** To be modified ***
USDT = trinity.Tokens("USDT")  # *** To be modified ***

# Printing parameters
print_update = False
print_action = False
no_fee = False
single_token = False
graph = True
trinity.no_fee = no_fee
trinity.print_action = print_action
trinity.print_update = print_update

# User wallet, BTC=30000, ETH=2000, Denote Peter to be arbitraguer swapper, Mary to be first depositor, Other to be an active LP
Peter.wallet = {
    "BTC": 10000,
    "ETH": 150000,
    "USDT": 300000000,
}  # *** To be modified ***
Mary.wallet = {"BTC": 10000, "ETH": 150000, "USDT": 300000000}  # *** To be modified ***
Others.wallet = {
    "BTC": 10000,
    "ETH": 150000,
    "USDT": 300000000,
}  # *** To be modified ***

Peter.lp_token = {"BTC": 0, "ETH": 0, "USDT": 0}
Mary.lp_token = {"BTC": 0, "ETH": 0, "USDT": 0}
Others.lp_token = {"BTC": 0, "ETH": 0, "USDT": 0}

"""
Block begin initialization after user inputs.
"""
# Setting Initial token price
price_list = [
    "price_scale",
    "price_scale_new",
    "price_oracle",
    "price_market",
    "price_last",
    "price_last_average",
    "price_oracle_average",
]
price_list_records = [
    "price_scale_records",
    "price_scale_new_records",
    "price_oracle_records",
    "price_last_records",
]

# Set initial price index
for price_index in price_list:
    setattr(BTC, price_index, btc_price_cex[0])
    setattr(ETH, price_index, eth_price_cex[0])

for price_index_records in price_list_records:
    setattr(BTC, price_index_records, [btc_price_cex[0]])
    setattr(ETH, price_index_records, [eth_price_cex[0]])

# Initial Deposit
init_pool_liquidity_stable = 1500000  # *** To be modified ***
init_pool_liquidity_btc = init_pool_liquidity_stable / btc_price_cex[0]
init_pool_liquidity_eth = init_pool_liquidity_stable / eth_price_cex[0]

Mary.deposit(Trinity, BTC, init_pool_liquidity_btc, ETH, USDT)
Mary.deposit(Trinity, ETH, init_pool_liquidity_eth, BTC, USDT)
Mary.deposit(Trinity, USDT, init_pool_liquidity_stable, BTC, ETH)

impermanent_loss_btc = []
impermanent_loss_eth = []
impermanent_loss_usdt = []

def arb_bot(
    first_token,
    USDT,
    second_token,
    first_token_cex,
    second_token_cex,
    tolerance,
    t,
    cex_count,
    ema_t_dict,
):
    epislon_division = 2

    if first_token.price_last > first_token_cex[t]:
        while True:
            diff = first_token.price_last - first_token_cex[t]

            epsilon_first_token_adjustment = (
                diff / first_token.price_scale / epislon_division
            )
            if diff < tolerance:
                ema_t_dict[t].append(USDT.ema_t)
                USDT.ema_t = 1
                break

            Peter.swap_volatile_repeg(
                Trinity, first_token, epsilon_first_token_adjustment, USDT, second_token
            )

            # Recording
            first_token.price_last_dict_records[t].append(first_token.price_last)
            first_token.price_last_average_records[t] = np.mean(
                first_token.price_last_dict_records[t]
            )

            first_token.price_last_average = np.mean(
                first_token.price_last_dict_records[t]
            )

            first_token.volume_dict_records[t].append(first_token.volume)
            first_token.volume_hourly_records[t] = np.sum(
                first_token.volume_dict_records[t]
            )

            USDT.volume_dict_records[t].append(USDT.volume)
            USDT.volume_hourly_records[t] = np.sum(USDT.volume_dict_records[t])
            USDT.price_last_average_records[t] = USDT.price_last

            second_token.price_last_dict_records[t].append(second_token.price_last)
            second_token.price_last_average_records[t] = np.mean(
                second_token.price_last_dict_records[t]
            )

            second_token.price_last_average = np.mean(
                second_token.price_last_dict_records[t]
            )

            second_token.volume_dict_records[t].append(second_token.volume)
            second_token.volume_hourly_records[t] = np.sum(
                second_token.volume_dict_records[t]
            )
            USDT.ema_t += 1
    else:
        while True:
            diff = first_token_cex[t] - first_token.price_last
            epsilon_usdt_adjustment = diff / epislon_division
            if diff < tolerance:
                ema_t_dict[t].append(USDT.ema_t)
                USDT.ema_t = 1
                break
            Peter.swap_volatile_repeg(
                Trinity, USDT, epsilon_usdt_adjustment, first_token, second_token
            )

            # Recording
            first_token.price_last_dict_records[t].append(first_token.price_last)
            first_token.price_last_average_records[t] = np.mean(
                first_token.price_last_dict_records[t]
            )

            first_token.price_last_average = np.mean(
                first_token.price_last_dict_records[t]
            )

            first_token.volume_dict_records[t].append(first_token.volume)
            first_token.volume_hourly_records[t] = np.sum(
                first_token.volume_dict_records[t]
            )

            USDT.volume_dict_records[t].append(USDT.volume)
            USDT.volume_hourly_records[t] = np.sum(USDT.volume_dict_records[t])
            USDT.price_last_average_records[t] = USDT.price_last

            second_token.price_last_dict_records[t].append(second_token.price_last)
            second_token.price_last_average_records[t] = np.mean(
                second_token.price_last_dict_records[t]
            )

            second_token.price_last_average = np.mean(
                second_token.price_last_dict_records[t]
            )

            second_token.volume_dict_records[t].append(second_token.volume)
            second_token.volume_hourly_records[t] = np.sum(
                second_token.volume_dict_records[t]
            )
            USDT.ema_t += 1

    if single_token == False:
        # Second token
        if second_token.price_last > second_token_cex[t]:
            while True:  # Checking second token and arb
                diff = second_token.price_last - second_token_cex[t]

                epsilon_second_token_adjustment = (
                    diff / second_token.price_scale / epislon_division
                )

                if diff < tolerance:
                    ema_t_dict[t].append(USDT.ema_t)
                    USDT.ema_t = 1
                    break

                Peter.swap_volatile_repeg(
                    Trinity,
                    second_token,
                    epsilon_second_token_adjustment,
                    USDT,
                    first_token,
                )

                # Recording
                first_token.price_last_dict_records[t].append(first_token.price_last)
                first_token.price_last_average_records[t] = np.mean(
                    first_token.price_last_dict_records[t]
                )

                first_token.volume_dict_records[t].append(first_token.volume)
                first_token.volume_hourly_records[t] = np.sum(
                    first_token.volume_dict_records[t]
                )

                USDT.volume_dict_records[t].append(USDT.volume)
                USDT.volume_hourly_records[t] = np.sum(USDT.volume_dict_records[t])
                USDT.price_last_average_records[t] = USDT.price_last

                # Also need to record other token
                second_token.price_last_dict_records[t].append(second_token.price_last)
                second_token.price_last_average_records[t] = np.mean(
                    second_token.price_last_dict_records[t]
                )

                second_token.volume_dict_records[t].append(second_token.volume)
                second_token.volume_hourly_records[t] = np.sum(
                    second_token.volume_dict_records[t]
                )

                USDT.ema_t += 1
        else:
            while True:  # Checking second token and arb
                diff = second_token_cex[t] - second_token.price_last
                epsilon_usdt_adjustment = diff / epislon_division

                if diff < tolerance:
                    ema_t_dict[t].append(USDT.ema_t)
                    USDT.ema_t = 1
                    break

                Peter.swap_volatile_repeg(
                    Trinity, USDT, epsilon_usdt_adjustment, second_token, first_token
                )

                # Recording
                first_token.price_last_dict_records[t].append(first_token.price_last)
                first_token.price_last_average_records[t] = np.mean(
                    first_token.price_last_dict_records[t]
                )

                first_token.volume_dict_records[t].append(first_token.volume)
                first_token.volume_hourly_records[t] = np.sum(
                    first_token.volume_dict_records[t]
                )

                USDT.volume_dict_records[t].append(USDT.volume)
                USDT.volume_hourly_records[t] = np.sum(USDT.volume_dict_records[t])
                USDT.price_last_average_records[t] = USDT.price_last

                # Also need to record other token
                second_token.price_last_dict_records[t].append(second_token.price_last)
                second_token.price_last_average_records[t] = np.mean(
                    second_token.price_last_dict_records[t]
                )

                second_token.volume_dict_records[t].append(second_token.volume)
                second_token.volume_hourly_records[t] = np.sum(
                    second_token.volume_dict_records[t]
                )

                USDT.ema_t += 1
            return


def historic_arbs_new(
    target_token, other_token1, other_token2, btc_price_cex, eth_price_cex
):
    import numpy as np
    import random

    inf = 1e10
    others_percentage = 0.2
    deposit_amount = target_token.cash * others_percentage
    Others.deposit(Trinity, target_token, deposit_amount, other_token1, other_token2)

    btc_price_cex_trimmed = btc_price_cex[:500]
    eth_price_cex_trimmed = eth_price_cex[:500]
    n = len(btc_price_cex_trimmed)

    for token in [BTC, ETH, USDT]:
        token.price_last_average_records = [1e10] * n
        token.volume_hourly_records = [1e10] * n
        token.price_last_dict_records = [[] for _ in range(n)]
        token.volume_dict_records = [[] for _ in range(n)]
        token.hf_lp_records = getattr(token, 'hf_lp_records', 0)

    impermanent_loss = []
    impermanent_loss_percent = []
    uniswapv2_impermanent_loss_percent = []
    hodl_value = []
    lp_value = []
    ema_t_dict = {}
    cex_count = 0
    random.seed(1234)
    tolerance = 5

    btc_amm_hourly, eth_amm_hourly = [], []
    btc_oracle_hourly, eth_oracle_hourly = [], []
    btc_scale_hourly, eth_scale_hourly = [], []
    btc_scale_new_hourly, eth_scale_new_hourly = [], []
    r_star_repeg = []

    btc_fee_rate, eth_fee_rate, usdt_fee_rate = [], [], []
    btc_cov_ratio, eth_cov_ratio, usdt_cov_ratio = [], [], []
    btc_sigma, btc_utilization = [], []

    btc_slippage_hourly, eth_slippage_hourly, usdt_slippage_hourly = [], [], []
    impermanent_loss_btc, impermanent_loss_eth, impermanent_loss_usdt = [], [], []

    btc_initial_liability = BTC.liability or 1
    eth_initial_liability = ETH.liability or 1
    usdt_initial_liability = USDT.liability or 1

    for t in range(n):
        ema_t_dict[t] = []

        u = random.randint(0, 1)
        if single_token:
            u = 0

        if u == 0:
            arb_bot(BTC, USDT, ETH, btc_price_cex_trimmed, eth_price_cex_trimmed, tolerance, t, cex_count, ema_t_dict)
        else:
            arb_bot(ETH, USDT, BTC, eth_price_cex_trimmed, btc_price_cex_trimmed, tolerance, t, cex_count, ema_t_dict)

        hodl_value_current = deposit_amount * target_token.price_last
        init_cash = Others.wallet[target_token.name]

        Others.withdraw_volatile(Trinity, target_token, inf, other_token1, other_token2)
        withdraw_amount = Others.wallet[target_token.name] - init_cash
        Others.deposit(Trinity, target_token, withdraw_amount, other_token1, other_token2)

        withdraw_amount_current = withdraw_amount * target_token.price_last
        impermanent_loss_current = withdraw_amount_current - hodl_value_current

        usdt_price = 1.0

# Calculate IL as percentage relative to LP value (withdraw_amount_current)
        il_percent_usdt = (impermanent_loss_current / withdraw_amount_current) * 100

        # Optionally track absolute token amount loss
        il_tokens_usdt = impermanent_loss_current / usdt_price

        # Store percent IL in the list instead of raw ratio
        impermanent_loss_usdt.append(il_percent_usdt)
        il_btc = impermanent_loss_current / btc_price_cex_trimmed[t]
        il_eth = impermanent_loss_current / eth_price_cex_trimmed[t]

        impermanent_loss_btc.append(il_btc)
        impermanent_loss_eth.append(il_eth)

        impermanent_loss_percent_current = (
            impermanent_loss_current / withdraw_amount_current * 100
        )

        k_btc = btc_price_cex_trimmed[t] / btc_price_cex_trimmed[0]
        k_eth = eth_price_cex_trimmed[t] / eth_price_cex_trimmed[0]
        if single_token:
            k_eth = 0
        k = max(k_btc, k_eth)
        uni_il = (2 * np.sqrt(k) / (1 + k) - 1) * 100

        uniswapv2_impermanent_loss_percent.append(uni_il)
        hodl_value.append(hodl_value_current)
        lp_value.append(withdraw_amount_current)
        impermanent_loss.append(impermanent_loss_current)
        impermanent_loss_percent.append(impermanent_loss_percent_current)
        cex_count += 1
        print(cex_count)
        btc_amm_hourly.append(BTC.price_last)
        eth_amm_hourly.append(ETH.price_last)
        btc_oracle_hourly.append(BTC.price_oracle)
        btc_scale_hourly.append(BTC.price_scale)
        btc_scale_new_hourly.append(BTC.price_scale_new)
        eth_oracle_hourly.append(ETH.price_oracle)
        eth_scale_hourly.append(ETH.price_scale)
        eth_scale_new_hourly.append(ETH.price_scale_new)
        r_star_repeg.append(trinity.equilCovRatio_repeg(BTC, ETH, USDT))

        btc_fee_rate.append(BTC.h_rate_records[-1])
        eth_fee_rate.append(ETH.h_rate_records[-1])
        usdt_fee_rate.append(USDT.h_rate_records[-1])
        btc_cov_ratio.append(BTC.cov_ratio())
        eth_cov_ratio.append(ETH.cov_ratio())
        usdt_cov_ratio.append(USDT.cov_ratio())
        btc_sigma.append(BTC.sigma_T_records[-1])
        btc_utilization.append(BTC.utilization_T_records[-1])

        btc_slip = abs(BTC.price_last - btc_price_cex_trimmed[t]) / btc_price_cex_trimmed[t] * 100
        eth_slip = abs(ETH.price_last - eth_price_cex_trimmed[t]) / eth_price_cex_trimmed[t] * 100
        usdt_slip = abs(USDT.price_last - usdt_price) / usdt_price * 100

        btc_slippage_hourly.append(btc_slip)
        eth_slippage_hourly.append(eth_slip)
        usdt_slippage_hourly.append(usdt_slip)

    btc_apy = BTC.hf_lp_records / btc_initial_liability
    eth_apy = ETH.hf_lp_records / eth_initial_liability
    usdt_apy = USDT.hf_lp_records / usdt_initial_liability

    total_ema = sum([sum(v) for v in ema_t_dict.values()])
    average_aggregated_value = total_ema / len(ema_t_dict)
    return (
        hodl_value,
        lp_value,
        impermanent_loss,
        impermanent_loss_percent,
        uniswapv2_impermanent_loss_percent,
        deposit_amount,
        btc_price_cex_trimmed,
        eth_price_cex_trimmed,
        btc_amm_hourly,
        eth_amm_hourly,
        btc_oracle_hourly,
        btc_scale_hourly,
        btc_scale_new_hourly,
        eth_oracle_hourly,
        eth_scale_hourly,
        eth_scale_new_hourly,
        r_star_repeg,
        ema_t_dict,
        btc_fee_rate,
        btc_cov_ratio,
        eth_fee_rate,
        eth_cov_ratio,
        usdt_fee_rate,
        usdt_cov_ratio,
        btc_sigma,
        btc_utilization,
        btc_slippage_hourly,
        eth_slippage_hourly,
        usdt_slippage_hourly,
        impermanent_loss_btc,
        impermanent_loss_eth,
        impermanent_loss_usdt,
        btc_apy,
        eth_apy,
        usdt_apy,
        average_aggregated_value
    )
    # return {
    #     "hodl_value": hodl_value,
    #     "lp_value": lp_value,
    #     "impermanent_loss": impermanent_loss,
    #     "impermanent_loss_percent": impermanent_loss_percent,
    #     "uniswapv2_impermanent_loss_percent": uniswapv2_impermanent_loss_percent,
    #     "deposit_amount": deposit_amount,
    #     "btc_price_cex_trimmed": btc_price_cex_trimmed,
    #     "eth_price_cex_trimmed": eth_price_cex_trimmed,
    #     "btc_amm_hourly": btc_amm_hourly,
    #     "eth_amm_hourly": eth_amm_hourly,
    #     "btc_oracle_hourly": btc_oracle_hourly,
    #     "btc_scale_hourly": btc_scale_hourly,
    #     "btc_scale_new_hourly": btc_scale_new_hourly,
    #     "eth_oracle_hourly": eth_oracle_hourly,
    #     "eth_scale_hourly": eth_scale_hourly,
    #     "eth_scale_new_hourly": eth_scale_new_hourly,
    #     "r_star_repeg": r_star_repeg,
    #     "ema_t_dict": ema_t_dict,
    #     "btc_fee_rate": btc_fee_rate,
    #     "btc_cov_ratio": btc_cov_ratio,
    #     "eth_fee_rate": eth_fee_rate,
    #     "eth_cov_ratio": eth_cov_ratio,
    #     "usdt_fee_rate": usdt_fee_rate,
    #     "usdt_cov_ratio": usdt_cov_ratio,
    #     "btc_sigma": btc_sigma,
    #     "btc_utilization": btc_utilization,
    #     "btc_slippage_hourly": btc_slippage_hourly,
    #     "eth_slippage_hourly": eth_slippage_hourly,
    #     "usdt_slippage_hourly": usdt_slippage_hourly,
    #     "impermanent_loss_btc": impermanent_loss_btc,
    #     "impermanent_loss_eth": impermanent_loss_eth,
    #     "impermanent_loss_usdt": impermanent_loss_usdt,
    #     "btc_apy": btc_apy,
    #     "eth_apy": eth_apy,
    #     "usdt_apy": usdt_apy,
    #     "average_aggregated_value": average_aggregated_value
    # }





# Run the function
(
 hodl_value,
        lp_value,
        impermanent_loss,
        impermanent_loss_percent,
        uniswapv2_impermanent_loss_percent,
        deposit_amount,
        btc_price_cex_trimmed,
        eth_price_cex_trimmed,
        btc_amm_hourly,
        eth_amm_hourly,
        btc_oracle_hourly,
        btc_scale_hourly,
        btc_scale_new_hourly,
        eth_oracle_hourly,
        eth_scale_hourly,
        eth_scale_new_hourly,
        r_star_repeg,
        ema_t_dict,
        btc_fee_rate,
        btc_cov_ratio,
        eth_fee_rate,
        eth_cov_ratio,
        usdt_fee_rate,
        usdt_cov_ratio,
        btc_sigma,
        btc_utilization,
        btc_slippage_hourly,
        eth_slippage_hourly,
        usdt_slippage_hourly,
        impermanent_loss_btc,
        impermanent_loss_eth,
        impermanent_loss_usdt,
        btc_apy,
        eth_apy,
        usdt_apy,
        average_aggregated_value
) = historic_arbs_new(USDT, BTC, ETH, btc_price_cex, eth_price_cex)

# State observations
print("-------------- The BTC-USDT price -------------------")
percentage_chage = (
    (BTC.price_last - BTC.price_last_records[0]) / BTC.price_last_records[0]
) * 100

print(
    f"The price change from {BTC.price_last_records[0]} to {BTC.price_last} ({percentage_chage})"
)
print({"Asset: BTC": BTC.cash, "ETH": ETH.cash, "USDT": USDT.cash})
print({"Liability: BTC": BTC.liability, "ETH": ETH.liability, "USDT": USDT.liability})
print(
    {
        "hair cut: BTC": BTC.hf_lp_records,
        "ETH": ETH.hf_lp_records,
        "USDT": USDT.hf_lp_records,
    }
)
print(
    {
        "Coverage ratio: BTC": BTC.cov_ratio(),
        "ETH": ETH.cov_ratio(),
        "USDT": USDT.cov_ratio(),
    }
)

# For Trimmed data analysis
Multiplier = 1
print(f"BTC APY = {BTC.hf_lp_records/BTC.liability * Multiplier}")
print(f"ETH APY = {ETH.hf_lp_records/ETH.liability * Multiplier}")
print(f"USDT APY = {USDT.hf_lp_records/USDT.liability * Multiplier}")

# Fee APY
# btc_apy = BTC.hf_lp_records / BTC.liability
# eth_apy = ETH.hf_lp_records / ETH.liability
# usdt_apy = USDT.hf_lp_records / USDT.liability


# Finding the average ema
total_values = 0
aggregated_values = []

for key, values in ema_t_dict.items():
    aggregated_value = sum(values)
    aggregated_values.append(aggregated_value)
    total_values += aggregated_value

average_aggregated_value = total_values / len(ema_t_dict)
print("The average swapping frequency is: ", average_aggregated_value)

print("-------------- END  ----------------")
elapsed_time = time.time() - start_time
print(f"Elapsed time is, {elapsed_time}, seconds")

# Organize the variables into a dictionary
data_to_export = {
    "btc_apy": btc_apy,
    "eth_apy": eth_apy,
    "usdt_apy": usdt_apy,
    "average_aggregated_value": average_aggregated_value,
    "impermanent_loss": impermanent_loss,
    "impermanent_loss_percent": impermanent_loss_percent,
    "uniswapv2_impermanent_loss_percent": uniswapv2_impermanent_loss_percent,
    "btc_amm_hourly": btc_amm_hourly,
    "btc_oracle_hourly": btc_oracle_hourly,
    "btc_scale_hourly": btc_scale_hourly,
    "btc_scale_new_hourly": btc_scale_new_hourly,
    "btc_fee_rate": btc_fee_rate,
    "btc_sigma": btc_sigma,
    "btc_fee_sigma_u_records": BTC.Fee_sigma_u_records,
    "btc_utilization": btc_utilization,
    "btc_cov_ratio": btc_cov_ratio,
    "eth_cov_ratio": eth_cov_ratio,
    "usdt_cov_ratio": usdt_cov_ratio,
    "eth_fee_rate": eth_fee_rate,
    "usdt_fee_rate": usdt_fee_rate,
    "hodl_value": hodl_value,
    "lp_value": lp_value,
    "btc_price_cex_trimmed": btc_price_cex_trimmed,
    "eth_price_cex_trimmed": eth_price_cex_trimmed,
    "eth_amm_hourly": eth_amm_hourly,
    "eth_oracle_hourly": eth_oracle_hourly,
    "eth_scale_hourly": eth_scale_hourly,
    "eth_scale_new_hourly": eth_scale_new_hourly,
    "r_star_repeg": r_star_repeg,
    "ema_t_dict": ema_t_dict,
}
output = {
    "apy_lp": {
        "BTC": round(btc_apy, 4),
        "ETH": round(eth_apy, 4),
        "USDT": round(usdt_apy, 4),
    },
    "coverage_ratio": {
        "BTC": btc_cov_ratio,
        "ETH": eth_cov_ratio,
        "USDT": usdt_cov_ratio,
    },
    "slippage": {
        "BTC":btc_slippage_hourly,
        "ETH": eth_slippage_hourly,
        "USDT": usdt_slippage_hourly,
    },
    "impermanent_loss_percent": {
        "BTC": impermanent_loss_btc,
        "ETH": impermanent_loss_eth, 
        "USDT":impermanent_loss_usdt,
    },
    "operation_counter": trinity.operation_counters,
    "repeg_count": trinity.operation_counters.get("repeg", 0),
}
print("Computation Cost (Operation Counts):")
print(trinity.operation_counters)


# Serialize the dictionary to a JSON formatted string and write it to a file
with open("data_export.json", "w") as json_file:
    json.dump(output, json_file, indent=4)
