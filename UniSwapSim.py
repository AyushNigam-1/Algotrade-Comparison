import pandas as pd
import numpy as np
import json

df = pd.read_csv("BTC_ETH_price.csv")
btc_prices = df["BTC"].values
eth_prices = df["ETH"].values

usdt_prices = np.clip(
    np.full_like(btc_prices, 1.0) + np.random.normal(0, 0.0001, size=len(btc_prices)),  # tighter noise for USDT
    0.999, 1.001
)

fee_rate = 0.003
steps = len(btc_prices)
initial_liquidity_token = 10  # smaller initial token liquidity for better realism
initial_liquidity_usdt = 10000  # fixed USDT liquidity pool side

def impermanent_loss(price_ratio):
    return 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1

class UniswapPool:
    def __init__(self, token_reserve, usdt_reserve, token_prices):
        self.token_reserve = token_reserve
        self.usdt_reserve = usdt_reserve
        self.k = token_reserve * usdt_reserve
        self.token_prices = token_prices
        self.lp_token_supply = np.sqrt(token_reserve * usdt_reserve)  # LP tokens = sqrt(k)
        self.lp_tokens_held = self.lp_token_supply
        self.initial_token_amount = token_reserve  # Save initial token amount for HODL calculation
        self.lp_values = []
        self.hodl_values = []
        self.slippages = []
        self.volumes = []
        self.fees_collected = []

    def get_amm_price(self):
        return self.usdt_reserve / self.token_reserve

    def trade(self, real_price, max_slippage=0.01):
        amm_price = self.get_amm_price()
        price_gap = real_price / amm_price

        # No trade if price gap within slippage tolerance
        if 1 - max_slippage < price_gap < 1 + max_slippage:
            self.slippages.append(0)
            self.volumes.append(0)
            self.fees_collected.append(0)
            return

        max_trade_ratio = 0.005  # max 0.5% of reserves traded per step

        if price_gap > 1 + max_slippage:
            max_usdt_in = max_trade_ratio * self.usdt_reserve
            trade_factor = min(price_gap - 1 - max_slippage, max_trade_ratio)  # scaled to gap
            amount_usdt_in = max_usdt_in * (trade_factor / max_trade_ratio)
            fee = amount_usdt_in * fee_rate
            amount_in_with_fee = amount_usdt_in - fee
            new_usdt_reserve = self.usdt_reserve + amount_usdt_in
            new_token_reserve = self.k / (self.usdt_reserve + amount_in_with_fee)
            token_out = self.token_reserve - new_token_reserve
            effective_price = amount_usdt_in / token_out
            slippage = abs(1 - (effective_price / real_price))
            self.usdt_reserve = new_usdt_reserve
            self.token_reserve = new_token_reserve
            self.k = self.token_reserve * self.usdt_reserve
            self.slippages.append(slippage)
            self.volumes.append(amount_usdt_in)
            self.fees_collected.append(fee)

        elif price_gap < 1 - max_slippage:
            max_token_in = max_trade_ratio * self.token_reserve
            trade_factor = min(1 - max_slippage - price_gap, max_trade_ratio)
            amount_token_in = max_token_in * (trade_factor / max_trade_ratio)
            fee = amount_token_in * fee_rate
            amount_in_with_fee = amount_token_in - fee
            new_token_reserve = self.token_reserve + amount_in_with_fee
            new_usdt_reserve = self.k / new_token_reserve
            usdt_out = self.usdt_reserve - new_usdt_reserve
            effective_price = usdt_out / amount_token_in
            slippage = abs(1 - (effective_price / real_price))
            self.token_reserve = new_token_reserve
            self.usdt_reserve = new_usdt_reserve
            self.k = self.token_reserve * self.usdt_reserve
            self.slippages.append(slippage)
            self.volumes.append(usdt_out)
            fee_usdt = fee * real_price  # convert token fee to USDT equivalent for consistency
            self.fees_collected.append(fee_usdt)

    def calculate_lp_value(self, current_price):
        lp_share = self.lp_tokens_held / self.lp_token_supply
        token_amount = lp_share * self.token_reserve
        usdt_amount = lp_share * self.usdt_reserve
        return token_amount * current_price + usdt_amount

    def calculate_hodl_value(self, current_price):
        return self.initial_token_amount * current_price

btc_pool = UniswapPool(initial_liquidity_token, initial_liquidity_usdt, btc_prices)
eth_pool = UniswapPool(initial_liquidity_token, initial_liquidity_usdt, eth_prices)
usdt_pool = UniswapPool(initial_liquidity_usdt, initial_liquidity_usdt, usdt_prices)

for i in range(1, steps):
    btc_pool.trade(btc_prices[i])
    eth_pool.trade(eth_prices[i])
    usdt_pool.trade(usdt_prices[i])
    btc_pool.lp_values.append(btc_pool.calculate_lp_value(btc_prices[i]))
    btc_pool.hodl_values.append(btc_pool.calculate_hodl_value(btc_prices[i]))
    eth_pool.lp_values.append(eth_pool.calculate_lp_value(eth_prices[i]))
    eth_pool.hodl_values.append(eth_pool.calculate_hodl_value(eth_prices[i]))
    usdt_pool.lp_values.append(usdt_pool.calculate_lp_value(usdt_prices[i]))
    usdt_pool.hodl_values.append(usdt_pool.calculate_hodl_value(usdt_prices[i]))

def summarize(pool, token):
    final_lp = pool.lp_values[-1]
    final_hodl = pool.hodl_values[-1]
    price_ratio = pool.token_prices[-1] / pool.token_prices[0]
    il = impermanent_loss(price_ratio)
    rel_gain = (final_lp - final_hodl) / final_hodl
    avg_slip = np.mean(np.abs(pool.slippages))
    total_volume = float(np.sum(pool.volumes))
    total_fees = float(np.sum(pool.fees_collected))

    # Annualize APY assuming daily steps (365 days per year)
    steps = len(pool.lp_values)
    apy = (1 + rel_gain) ** (365 / steps) - 1

    return {
        "LP Final Value": final_lp,
        "HODL Final Value": final_hodl,
        "Impermanent Loss %": il * 100,
        "Relative LP Gain %": rel_gain * 100,
        "Annualized APY %": apy * 100,
        "Average Slippage": avg_slip,
        "Total Volume": total_volume,
        "Total Fees Collected": total_fees
    }

results = {
    "UniswapV2": {
        "BTC": summarize(btc_pool, "BTC"),
        "ETH": summarize(eth_pool, "ETH"),
        "USDT": summarize(usdt_pool, "USDT")
    }
}

with open("final_uniswap_realistic_simulation.json", "w") as f:
    json.dump(results, f, indent=2)
